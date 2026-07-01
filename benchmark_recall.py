"""
VCTX Recall Benchmark — Hypothesis 1
Measures recall@k accuracy for keyword-only vs hybrid (keyword+embedding) search.
"""
import json
import sys
import time
import os
from pathlib import Path

# ── Force fresh DB ──────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "vctx_benchmark.db"
if DB_PATH.exists():
    DB_PATH.unlink()

# Patch DB_PATH before importing server
import server
server.DB_PATH = DB_PATH
server.DB_DIR = DB_PATH.parent

# ── Test corpus: 20 knowledge blocks with known topics ──────────────
CORPUS = [
    {
        "title": "Git Rebase Workflow",
        "content": "When working on a feature branch, use git rebase main to replay commits on top of the latest main branch. This creates a linear history. Use git rebase -i for interactive rebase to squash, reorder, or edit commits. Always force-push with --force-with-lease after rebase.",
        "conclusion": "Rebase rewrites history to create clean linear commits on top of main.",
        "keywords": ["git", "rebase", "workflow", "commits", "history"],
        "query": "how to rebase a branch",
        "expected_title": "Git Rebase Workflow",
    },
    {
        "title": "Python Virtual Environments",
        "content": "Use python -m venv .venv to create a virtual environment. Activate with source .venv/bin/activate on Linux or .venv\\Scripts\\activate on Windows. Install packages with pip install. Keep dependencies in requirements.txt generated via pip freeze.",
        "conclusion": "Virtual environments isolate Python project dependencies from the system.",
        "keywords": ["python", "venv", "virtual", "environment", "pip", "dependencies"],
        "query": "python dependency isolation",
        "expected_title": "Python Virtual Environments",
    },
    {
        "title": "Docker Container Networking",
        "content": "Docker containers communicate through bridge networks by default. Use docker network create to define custom networks. Containers on the same network can reach each other by service name. Expose ports with -p flag. Use host network mode for maximum performance.",
        "conclusion": "Docker networking uses bridge networks; containers resolve each other by name on shared networks.",
        "keywords": ["docker", "network", "container", "bridge", "ports"],
        "query": "how do containers talk to each other",
        "expected_title": "Docker Container Networking",
    },
    {
        "title": "React useState Hook",
        "content": "The useState hook adds state to functional components. Call useState(initialValue) to get [state, setState]. State updates trigger re-renders. Use functional updates setState(prev => prev + 1) when new state depends on previous. Multiple useState calls are independent.",
        "conclusion": "useState is the fundamental React hook for managing component state in functional components.",
        "keywords": ["react", "useState", "hook", "state", "component", "re-render"],
        "query": "react state management in components",
        "expected_title": "React useState Hook",
    },
    {
        "title": "SQL Index Optimization",
        "content": "Database indexes speed up SELECT queries at the cost of slower writes. Create indexes on columns used in WHERE, JOIN, and ORDER BY clauses. Composite indexes follow leftmost prefix rule. Use EXPLAIN to verify index usage. Avoid over-indexing; each index consumes storage and slows INSERT/UPDATE.",
        "conclusion": "Indexes trade write speed for read speed; use EXPLAIN to verify they are being used.",
        "keywords": ["sql", "index", "database", "query", "optimization", "explain"],
        "query": "speed up slow database queries",
        "expected_title": "SQL Index Optimization",
    },
    {
        "title": "OAuth 2.0 Authorization Code Flow",
        "content": "The Authorization Code flow is the most secure OAuth 2.0 grant for web apps. User is redirected to the authorization server, approves access, receives an authorization code, which the backend exchanges for tokens. Always use PKCE for public clients. Store refresh tokens securely.",
        "conclusion": "Authorization Code + PKCE is the recommended OAuth flow for web and mobile applications.",
        "keywords": ["oauth", "authorization", "code", "flow", "pkce", "token", "security"],
        "query": "secure authentication for web app",
        "expected_title": "OAuth 2.0 Authorization Code Flow",
    },
    {
        "title": "CSS Flexbox Layout",
        "content": "Flexbox is a one-dimensional layout method for arranging items in rows or columns. Set display: flex on the parent. Use justify-content for main axis alignment and align-items for cross axis. flex-wrap allows items to wrap. flex-shrink and flex-grow control how items resize.",
        "conclusion": "Flexbox arranges items in one dimension with powerful alignment and sizing controls.",
        "keywords": ["css", "flexbox", "layout", "flex", "alignment", "responsive"],
        "query": "center a div vertically",
        "expected_title": "CSS Flexbox Layout",
    },
    {
        "title": "Kubernetes Pod Lifecycle",
        "content": "A Pod is the smallest deployable unit in Kubernetes. Pods go through Pending, Running, Succeeded/Failed phases. Use liveness probes to detect unhealthy containers and readiness probes to control traffic routing. Pods are ephemeral; use Deployments for self-healing and scaling.",
        "conclusion": "Pods are ephemeral compute units; use Deployments for reliability and probes for health checking.",
        "keywords": ["kubernetes", "pod", "lifecycle", "deployment", "probe", "container"],
        "query": "kubernetes health checking",
        "expected_title": "Kubernetes Pod Lifecycle",
    },
    {
        "title": "TypeScript Generics",
        "content": "Generics provide type-safe reusable components. Define with <T> syntax. Constrain with extends keyword: <T extends HasLength>. Generic functions, interfaces, and classes allow the same code to work with multiple types while preserving type information. Default types: <T = string>.",
        "conclusion": "Generics enable type-safe code reuse across different types in TypeScript.",
        "keywords": ["typescript", "generics", "type", "generic", "constraint", "reusable"],
        "query": "write reusable type-safe functions",
        "expected_title": "TypeScript Generics",
    },
    {
        "title": "WebSocket Real-Time Communication",
        "content": "WebSocket provides full-duplex communication over a single TCP connection. Client initiates with HTTP Upgrade handshake. Use for real-time features: chat, live dashboards, gaming. Implement heartbeat ping/pong to detect disconnections. Scale with Redis Pub/Sub across multiple servers.",
        "conclusion": "WebSocket enables real-time bidirectional communication; use heartbeat for connection health.",
        "keywords": ["websocket", "real-time", "communication", "tcp", "bidirectional", "heartbeat"],
        "query": "real-time data streaming to browser",
        "expected_title": "WebSocket Real-Time Communication",
    },
    {
        "title": "Linux Process Management",
        "content": "Use ps aux to list all processes. top/htop for interactive monitoring. kill -9 PID to force terminate. Use nohup or disown to keep processes running after logout. systemd manages services: systemctl start/stop/status/restart. Background tasks with & and manage with jobs.",
        "conclusion": "Linux process management uses ps/kill/systemctl; systemd is the standard service manager.",
        "keywords": ["linux", "process", "kill", "systemd", "systemctl", "ps", "top"],
        "query": "how to kill a process by name",
        "expected_title": "Linux Process Management",
    },
    {
        "title": "Redis Caching Patterns",
        "content": "Use Redis as a caching layer with TTL-based expiration. Cache-aside pattern: check cache first, on miss query DB and populate cache. Write-through: write to cache and DB simultaneously. Use Redis sorted sets for leaderboards and rate limiting. Monitor hit rate with INFO stats.",
        "conclusion": "Cache-aside is the most common Redis pattern; monitor hit rate to tune TTL.",
        "keywords": ["redis", "cache", "caching", "ttl", "cache-aside", "pattern"],
        "query": "how to implement caching layer",
        "expected_title": "Redis Caching Patterns",
    },
    {
        "title": "Nginx Reverse Proxy Configuration",
        "content": "Configure Nginx as a reverse proxy with proxy_pass directive. Set proxy_set_header to forward client information. Use upstream blocks for load balancing across multiple backends. Enable gzip compression. Configure SSL termination with listen 443 ssl and certificate paths.",
        "conclusion": "Nginx reverse proxy forwards requests to backends; commonly used for SSL termination and load balancing.",
        "keywords": ["nginx", "reverse", "proxy", "load", "balancing", "ssl", "upstream"],
        "query": "set up load balancer for my app",
        "expected_title": "Nginx Reverse Proxy Configuration",
    },
    {
        "title": "pytest Testing Best Practices",
        "content": "Use fixtures for test setup and teardown. Parametrize tests with @pytest.mark.parametrize for data-driven testing. Use tmp_path fixture for filesystem tests. Mock external dependencies with monkeypatch or unittest.mock. Group tests in classes for shared fixtures. Run with -x to stop on first failure.",
        "conclusion": "pytest fixtures and parametrize enable clean, data-driven, isolated tests.",
        "keywords": ["pytest", "testing", "fixture", "parametrize", "mock", "python"],
        "query": "how to write good unit tests in python",
        "expected_title": "pytest Testing Best Practices",
    },
    {
        "title": "JWT Token Structure",
        "content": "JWT consists of three Base64URL-encoded parts: header (algorithm, type), payload (claims), signature. Store access tokens in memory (not localStorage). Use short expiration (15min) for access tokens, longer for refresh tokens. Validate signature, expiration, and issuer on every request.",
        "conclusion": "JWTs carry signed claims; keep access tokens short-lived and store them in memory only.",
        "keywords": ["jwt", "token", "authentication", "claims", "signature", "access", "refresh"],
        "query": "how JWT authentication works",
        "expected_title": "JWT Token Structure",
    },
    {
        "title": "PostgreSQL Query Plans",
        "content": "Use EXPLAIN ANALYZE to see actual execution plans. Look for sequential scans on large tables (often missing indexes). Nested loop joins are fast for small result sets; hash joins for larger ones. Check for high cost sorts that could use indexes. Use pg_stat_statements to find slow queries.",
        "conclusion": "EXPLAIN ANALYZE reveals actual execution; look for seq scans and high-cost sorts.",
        "keywords": ["postgresql", "explain", "query", "plan", "execution", "index", "scan"],
        "query": "analyze slow postgres query",
        "expected_title": "PostgreSQL Query Plans",
    },
    {
        "title": "CORS Configuration",
        "content": "Cross-Origin Resource Sharing controls which domains can access your API. Set Access-Control-Allow-Origin header (use specific domain, not * in production). Preflight OPTIONS requests check allowed methods and headers. Credentials require explicit Allow-Credentials: true and specific origin.",
        "conclusion": "CORS headers control cross-origin access; always specify exact origins in production.",
        "keywords": ["cors", "cross-origin", "access-control", "preflight", "security", "api"],
        "query": "fix CORS error in browser",
        "expected_title": "CORS Configuration",
    },
    {
        "title": "Terraform Infrastructure as Code",
        "content": "Terraform uses HCL to declare infrastructure. terraform plan shows changes before applying. Use modules for reusable components. State tracks resource mapping; store remotely with S3 backend. Use variables and outputs for parameterization. terraform import brings existing resources under management.",
        "conclusion": "Terraform declaratively manages infrastructure; always review plan before apply.",
        "keywords": ["terraform", "infrastructure", "iac", "hcl", "state", "module", "plan"],
        "query": "manage cloud infrastructure with code",
        "expected_title": "Terraform Infrastructure as Code",
    },
    {
        "title": "GraphQL Schema Design",
        "content": "Design GraphQL schemas around business domains, not database tables. Use connections (Relay spec) for paginated lists. Avoid N+1 queries with DataLoader. Define input types for mutations. Use fragments for reusable field sets. Keep resolvers thin; delegate to service layer.",
        "conclusion": "GraphQL schemas should model business domains; use DataLoader to prevent N+1 queries.",
        "keywords": ["graphql", "schema", "query", "mutation", "dataloader", "relay", "pagination"],
        "query": "graphql api design patterns",
        "expected_title": "GraphQL Schema Design",
    },
    {
        "title": "Prometheus Monitoring Setup",
        "content": "Prometheus scrapes metrics from /metrics endpoints. Define alert rules in .rules files. Use Grafana for dashboards. Key metrics: rate(http_requests_total[5m]) for request rate, histogram_quantile for latency percentiles. Use labels carefully; high cardinality causes memory issues.",
        "conclusion": "Prometheus scrapes metrics and alerts on rules; watch label cardinality to avoid memory bloat.",
        "keywords": ["prometheus", "monitoring", "metrics", "grafana", "alert", "scrape"],
        "query": "set up application monitoring",
        "expected_title": "Prometheus Monitoring Setup",
    },
]


# ── Benchmark ───────────────────────────────────────────────────────
def run_benchmark():
    print("=" * 60)
    print("VCTX Recall Benchmark — Hypothesis 1")
    print("=" * 60)

    # Archive all blocks
    t0 = time.time()
    for block in CORPUS:
        result = json.loads(server.vctx_archive(
            title=block["title"],
            content=block["content"],
            conclusion=block["conclusion"],
            keywords=block["keywords"],
            session_id="benchmark",
        ))
        assert result.get("status") == "archived", f"Archive failed: {result}"
    archive_time = time.time() - t0
    print(f"\nArchived {len(CORPUS)} blocks in {archive_time:.2f}s")

    # Test recall with embedding enabled (hybrid mode)
    print(f"\nEmbedding available: {server._HAS_EMBEDDING}")
    print("-" * 60)

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    results_detail = []

    for block in CORPUS:
        query = block["query"]
        expected = block["expected_title"]

        t1 = time.time()
        raw = server.vctx_search(query=query, top_k=5)
        search_time = time.time() - t1

        data = json.loads(raw)
        result_titles = [r["title"] for r in data["results"]]

        hit_at_1 = result_titles[0] == expected if result_titles else False
        hit_at_3 = expected in result_titles[:3]
        hit_at_5 = expected in result_titles[:5]

        recall_at_1 += int(hit_at_1)
        recall_at_3 += int(hit_at_3)
        recall_at_5 += int(hit_at_5)

        status = "HIT@1" if hit_at_1 else ("HIT@3" if hit_at_3 else ("HIT@5" if hit_at_5 else "MISS"))
        results_detail.append({
            "query": query,
            "expected": expected,
            "top1": result_titles[0] if result_titles else "",
            "status": status,
            "time_ms": round(search_time * 1000),
            "score": data["results"][0]["score"] if data["results"] else 0,
        })

    n = len(CORPUS)
    print(f"\n{'Query':<40} {'Expected':<30} {'Top-1 Hit':<30} {'Status':<8}")
    print("-" * 108)
    for r in results_detail:
        mark = "Y" if r["status"] != "MISS" else "N"
        print(f"{r['query']:<40} {r['expected']:<30} {r['top1']:<30} {r['status']:<8}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Recall@1: {recall_at_1}/{n} = {recall_at_1/n:.1%}")
    print(f"Recall@3: {recall_at_3}/{n} = {recall_at_3/n:.1%}")
    print(f"Recall@5: {recall_at_5}/{n} = {recall_at_5/n:.1%}")
    print(f"Avg search time: {sum(r['time_ms'] for r in results_detail)/n:.0f}ms")
    print(f"Search mode: {data.get('search_mode', 'unknown')}")

    # Now disable embedding and test keyword-only
    print("\n" + "=" * 60)
    print("KEYWORD-ONLY COMPARISON (embedding disabled)")
    print("=" * 60)

    old_has = server._HAS_EMBEDDING
    server._HAS_EMBEDDING = False

    kw_recall_1 = 0
    kw_recall_3 = 0
    kw_recall_5 = 0
    kw_results = []

    for block in CORPUS:
        query = block["query"]
        expected = block["expected_title"]

        raw = server.vctx_search(query=query, top_k=5)
        data = json.loads(raw)
        result_titles = [r["title"] for r in data["results"]]

        hit_at_1 = result_titles[0] == expected if result_titles else False
        hit_at_3 = expected in result_titles[:3]
        hit_at_5 = expected in result_titles[:5]

        kw_recall_1 += int(hit_at_1)
        kw_recall_3 += int(hit_at_3)
        kw_recall_5 += int(hit_at_5)

        status = "HIT@1" if hit_at_1 else ("HIT@3" if hit_at_3 else ("HIT@5" if hit_at_5 else "MISS"))
        kw_results.append({"query": query, "expected": expected, "status": status})

    server._HAS_EMBEDDING = old_has

    for r in kw_results:
        print(f"{r['query']:<40} {r['expected']:<30} {r['status']:<8}")

    print(f"\nKeyword-only Recall@1: {kw_recall_1}/{n} = {kw_recall_1/n:.1%}")
    print(f"Keyword-only Recall@3: {kw_recall_3}/{n} = {kw_recall_3/n:.1%}")
    print(f"Keyword-only Recall@5: {kw_recall_5}/{n} = {kw_recall_5/n:.1%}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<15} {'Hybrid (emb)':<18} {'Keyword-only':<18} {'Delta':<10}")
    print("-" * 55)
    print(f"{'Recall@1':<15} {recall_at_1/n:<18.1%} {kw_recall_1/n:<18.1%} {'+' if recall_at_1>=kw_recall_1 else ''}{(recall_at_1-kw_recall_1)/n:.1%}")
    print(f"{'Recall@3':<15} {recall_at_3/n:<18.1%} {kw_recall_3/n:<18.1%} {'+' if recall_at_3>=kw_recall_3 else ''}{(recall_at_3-kw_recall_3)/n:.1%}")
    print(f"{'Recall@5':<15} {recall_at_5/n:<18.1%} {kw_recall_5/n:<18.1%} {'+' if recall_at_5>=kw_recall_5 else ''}{(recall_at_5-kw_recall_5)/n:.1%}")

    # Cleanup
    if DB_PATH.exists():
        DB_PATH.unlink()

    return {
        "hybrid": {"r1": recall_at_1/n, "r3": recall_at_3/n, "r5": recall_at_5/n},
        "keyword_only": {"r1": kw_recall_1/n, "r3": kw_recall_3/n, "r5": kw_recall_5/n},
    }


if __name__ == "__main__":
    run_benchmark()
