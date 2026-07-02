use super::types::VctxMemoryConfig;
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Debug)]
struct MemoryBlock {
    block_id: String,
    title: String,
    content: String,
    conclusion: String,
    keywords: String,
    score: f64,
}

pub fn apply_request_memory(
    body: &mut Value,
    config: &VctxMemoryConfig,
    session_id: &str,
    app_type: &str,
) {
    if !config.enabled || body_already_has_vctx(body) {
        return;
    }

    let Some(query) = extract_query_text(body) else {
        return;
    };

    let db_path = Path::new(&config.db_path);
    if !db_path.exists() {
        return;
    }

    let memory = match recall_memory(db_path, &query, config) {
        Ok(memory) => memory,
        Err(err) => {
            log::warn!("[VCTX] recall skipped: {err}");
            return;
        }
    };

    if memory.is_empty() {
        return;
    }

    inject_memory(body, &memory, session_id, app_type);
}

pub fn maybe_checkpoint_response(
    body_bytes: &[u8],
    config: &VctxMemoryConfig,
    session_id: &str,
    app_type: &str,
    model: &str,
) {
    if !config.enabled {
        return;
    }

    let db_path = Path::new(&config.db_path);
    let Ok(value) = serde_json::from_slice::<Value>(body_bytes) else {
        return;
    };
    let text = extract_response_text(&value);
    let text = text.trim();
    if text.chars().count() < config.checkpoint_min_chars {
        return;
    }

    if let Err(err) = archive_checkpoint(db_path, text, session_id, app_type, model) {
        log::warn!("[VCTX] checkpoint skipped: {err}");
    }
}

fn recall_memory(
    db_path: &Path,
    query: &str,
    config: &VctxMemoryConfig,
) -> Result<String, rusqlite::Error> {
    let terms = tokenize(query);
    if terms.is_empty() {
        return Ok(String::new());
    }

    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT block_id, title, content, conclusion, keywords, importance, access_count
         FROM blocks
         ORDER BY importance DESC, access_count DESC, created_at DESC
         LIMIT 200",
    )?;
    let rows = stmt.query_map([], |row| {
        let title: String = row.get::<_, Option<String>>(1)?.unwrap_or_default();
        let content: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
        let conclusion: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
        let keywords: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
        let importance = row.get::<_, Option<f64>>(5)?.unwrap_or(1.0);
        let access_count = row.get::<_, Option<i64>>(6)?.unwrap_or(0);
        let searchable = format!("{title} {conclusion} {keywords}").to_lowercase();
        let hits = terms
            .iter()
            .filter(|term| searchable.contains(term.as_str()))
            .count() as f64;
        let score = hits + importance * 0.05 + (access_count as f64).min(20.0) * 0.01;

        Ok(MemoryBlock {
            block_id: row.get(0)?,
            title,
            content,
            conclusion,
            keywords,
            score,
        })
    })?;

    let mut blocks = Vec::new();
    for row in rows {
        let block = row?;
        if block.score >= 1.0 {
            blocks.push(block);
        }
    }
    blocks.sort_by(|a, b| b.score.total_cmp(&a.score));
    blocks.truncate(config.recall_top_k.max(1));

    let mut out = String::new();
    let mut used = 0usize;
    for block in blocks {
        let excerpt = truncate_chars(&block.content, 900);
        let chunk = format!(
            "[{}] {}\nConclusion: {}\nKeywords: {}\nExcerpt: {}\n\n",
            block.block_id, block.title, block.conclusion, block.keywords, excerpt
        );
        let chunk_len = chunk.chars().count();
        if used + chunk_len > config.max_memory_chars && !out.is_empty() {
            break;
        }
        used += chunk_len;
        out.push_str(&chunk);
    }

    Ok(out.trim().to_string())
}

fn inject_memory(body: &mut Value, memory: &str, session_id: &str, app_type: &str) {
    let text = format!(
        "<VCTX_MEMORY session=\"{}\">\n{}\n</VCTX_MEMORY>",
        escape_attr(session_id),
        memory
    );

    if matches!(app_type, "claude" | "claude-desktop") {
        inject_top_level_system(body, text);
        return;
    }

    if let Some(system) = body.get_mut("system") {
        match system {
            Value::String(existing) => {
                existing.push_str("\n\n");
                existing.push_str(&text);
            }
            Value::Array(items) => {
                items.push(json!({"type": "text", "text": text}));
            }
            _ => {
                *system = Value::String(text);
            }
        }
        log::info!("[VCTX] injected memory into top-level system");
        return;
    }

    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        if let Some(system_msg) = messages
            .iter_mut()
            .find(|msg| msg.get("role").and_then(Value::as_str) == Some("system"))
        {
            append_message_content(system_msg, &text);
        } else {
            messages.insert(0, json!({"role": "system", "content": text}));
        }
        log::info!("[VCTX] injected memory into messages");
    }
}

fn inject_top_level_system(body: &mut Value, text: String) {
    match body.get_mut("system") {
        Some(Value::String(existing)) => {
            existing.push_str("\n\n");
            existing.push_str(&text);
        }
        Some(Value::Array(items)) => {
            items.push(json!({"type": "text", "text": text}));
        }
        Some(other) => {
            *other = Value::String(text);
        }
        None => {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("system".to_string(), Value::String(text));
            }
        }
    }
    log::info!("[VCTX] injected memory into top-level system");
}

fn archive_checkpoint(
    db_path: &Path,
    text: &str,
    session_id: &str,
    app_type: &str,
    model: &str,
) -> Result<(), rusqlite::Error> {
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS blocks (
            block_id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT NOT NULL,
            token_count INTEGER,
            keywords TEXT,
            conclusion TEXT,
            session_id TEXT,
            project_id TEXT,
            user_id TEXT,
            source TEXT DEFAULT 'manual',
            is_recalled INTEGER DEFAULT 0,
            recall_from TEXT,
            fingerprint TEXT,
            embedding TEXT,
            created_at TEXT,
            last_access TEXT,
            importance REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_blocks_fingerprint ON blocks(fingerprint);",
    )?;

    let fingerprint = sha256_hex(text);
    let existing: Option<String> = conn
        .query_row(
            "SELECT block_id FROM blocks WHERE fingerprint=? LIMIT 1",
            params![&fingerprint],
            |row| row.get(0),
        )
        .optional()?;
    if existing.is_some() {
        return Ok(());
    }

    let now = Utc::now().to_rfc3339();
    let block_id = format!("{}-{}", Utc::now().format("%y%m%d"), &fingerprint[..6]);
    let title = format!("{} response checkpoint", app_type);
    let conclusion = truncate_chars(text, 220);
    let keywords = serde_json::to_string(&extract_keywords(text)).unwrap_or_else(|_| "[]".into());
    let token_count = approx_tokens(text);
    let content = format!("Model: {model}\n\n{text}");

    conn.execute(
        "INSERT INTO blocks
         (block_id, title, content, token_count, keywords, conclusion,
          session_id, project_id, user_id, source, is_recalled, recall_from,
          fingerprint, embedding, created_at, last_access, importance, access_count)
         VALUES (?, ?, ?, ?, ?, ?, ?, '', '', 'cc-switch-proxy', 0, NULL, ?, NULL, ?, ?, 1.0, 0)",
        params![
            block_id,
            title,
            content,
            token_count,
            keywords,
            conclusion,
            session_id,
            fingerprint,
            now,
            now,
        ],
    )?;
    log::info!("[VCTX] archived response checkpoint");
    Ok(())
}

fn append_message_content(message: &mut Value, extra: &str) {
    match message.get_mut("content") {
        Some(Value::String(existing)) => {
            existing.push_str("\n\n");
            existing.push_str(extra);
        }
        Some(Value::Array(items)) => items.push(json!({"type": "text", "text": extra})),
        Some(other) => *other = Value::String(extra.to_string()),
        None => {
            if let Some(obj) = message.as_object_mut() {
                obj.insert("content".to_string(), Value::String(extra.to_string()));
            }
        }
    }
}

fn body_already_has_vctx(body: &Value) -> bool {
    body.to_string().contains("<VCTX_MEMORY")
}

fn extract_query_text(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    messages.iter().rev().find_map(|msg| {
        (msg.get("role").and_then(Value::as_str) == Some("user"))
            .then(|| extract_content_text(msg.get("content").unwrap_or(&Value::Null)))
            .filter(|text| !text.trim().is_empty())
    })
}

fn extract_response_text(value: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(content) = value.get("content") {
        collect_text(content, &mut parts);
    }
    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(message) = choice.get("message") {
                collect_text(message.get("content").unwrap_or(message), &mut parts);
            }
            if let Some(delta) = choice.get("delta") {
                collect_text(delta.get("content").unwrap_or(delta), &mut parts);
            }
        }
    }
    if let Some(output) = value.get("output") {
        collect_text(output, &mut parts);
    }
    if let Some(output_text) = value.get("output_text").and_then(Value::as_str) {
        parts.push(output_text.to_string());
    }
    parts.join("\n")
}

fn extract_content_text(value: &Value) -> String {
    let mut parts = Vec::new();
    collect_text(value, &mut parts);
    parts.join("\n")
}

fn collect_text(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => parts.push(text.clone()),
        Value::Array(items) => {
            for item in items {
                collect_text(item, parts);
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                parts.push(text.to_string());
            } else if let Some(content) = map.get("content") {
                collect_text(content, parts);
            }
        }
        _ => {}
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for ch in text.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            current.push(ch);
        } else {
            if current.chars().count() >= 2 {
                terms.push(current.clone());
            }
            current.clear();
            if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
                terms.push(ch.to_string());
            }
        }
    }
    if current.chars().count() >= 2 {
        terms.push(current);
    }
    terms.sort();
    terms.dedup();
    terms
}

fn extract_keywords(text: &str) -> Vec<String> {
    let mut terms = tokenize(text);
    terms.retain(|term| term.chars().count() >= 2);
    terms.truncate(12);
    terms
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn approx_tokens(text: &str) -> usize {
    let cjk = text
        .chars()
        .filter(|ch| ('\u{4e00}'..='\u{9fff}').contains(ch))
        .count();
    let total = text.chars().count();
    cjk + total.saturating_sub(cjk) / 4
}

fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn escape_attr(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn config_for(path: &Path) -> VctxMemoryConfig {
        VctxMemoryConfig {
            enabled: true,
            db_path: path.to_string_lossy().to_string(),
            recall_top_k: 3,
            max_memory_chars: 2400,
            checkpoint_min_chars: 10,
        }
    }

    #[test]
    fn injects_claude_memory_into_top_level_system() {
        let mut body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "remember alpha"}]
        });

        inject_memory(&mut body, "alpha memory", "s1", "claude");

        assert!(body.get("system").is_some());
        assert!(body["system"].as_str().unwrap().contains("<VCTX_MEMORY"));
        assert!(!body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|msg| msg.get("role").and_then(Value::as_str) == Some("system")));
    }

    #[test]
    fn injects_openai_memory_into_system_message() {
        let mut body = json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "remember beta"}]
        });

        inject_memory(&mut body, "beta memory", "s1", "codex");

        let first = &body["messages"].as_array().unwrap()[0];
        assert_eq!(first["role"], "system");
        assert!(first["content"].as_str().unwrap().contains("<VCTX_MEMORY"));
    }

    #[test]
    fn checkpoint_writes_vctx_block() {
        let file = NamedTempFile::new().unwrap();
        let config = config_for(file.path());
        let body = json!({
            "content": [{"type": "text", "text": "This is a long enough checkpoint body for VCTX."}]
        });
        let bytes = serde_json::to_vec(&body).unwrap();

        maybe_checkpoint_response(&bytes, &config, "session-a", "claude", "mimo-v2.5-pro");

        let conn = Connection::open(file.path()).unwrap();
        let source: String = conn
            .query_row("SELECT source FROM blocks LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(source, "cc-switch-proxy");
    }
}
