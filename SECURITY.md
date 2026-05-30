# Security Audit — clinical-survival-analysis

## Version: 1.1.0 — Security Hardened
**Audit Date:** 2026-05-30
**Auditor:** Internal Security Review

## Summary

A full security review was performed across all Python source files in `src/`, `app.py`, and `tests/`, covering the OWASP Top 10, secrets detection, PHI/PII logging, input validation, pickle/deserialization safety, CORS, rate limiting, and dependency versions. Three HIGH and three MEDIUM issues were identified and resolved.

---

## Findings & Fixes

### HIGH

**H-1 — Internal exception details leaked to API callers**
- File: `src/api.py`, line 229 (pre-fix)
- Pattern: `raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")`
- Risk: Exception messages from scikit-survival, numpy, or PyTorch can expose model architecture, feature column names, tensor shapes, and internal stack paths to any caller — a meaningful model-inversion attack surface.
- Fix: The raw exception is now logged server-side via `logger.error(..., exc_info=True)`. The HTTP response returns only the generic string `"Prediction failed. Please contact support."`.

**H-2 — CORS wildcard (missing entirely — defaults to allow-all in browser contexts)**
- File: `src/api.py`
- Risk: No `CORSMiddleware` was configured, allowing any origin to call the API from a browser page, enabling cross-site request forgery from third-party sites.
- Fix: `CORSMiddleware` added with `allow_origins` restricted to `http://localhost:8501` and `http://127.0.0.1:8501` (the Streamlit dashboard) by default. Override via `ALLOWED_ORIGINS` environment variable in production.

**H-3 — No rate limiting on `/predict` endpoint**
- File: `src/api.py`
- Risk: The `/predict` endpoint performs model inference (CPU/memory intensive). Without rate limiting, a single client can exhaust server resources or abuse the endpoint for oracle-style model-extraction attacks.
- Fix: A per-IP sliding-window rate limiter is implemented as an ASGI middleware, defaulting to 60 requests per 60-second window. Configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS` environment variables.

### MEDIUM

**M-1 — Streamlit UI exposes raw exception text to browser**
- File: `app.py`, lines 92 and 471 (pre-fix)
- Pattern: `st.error(f"Error loading data: {exc}")` and `st.error(f"Deep learning models failed to load: {dl_exc}")`
- Risk: Full Python exception strings (including file paths, class names, internal state) rendered in the browser UI. In a deployed context this reveals server internals to end users.
- Fix: Exceptions are now logged server-side via `logger.error(..., exc_info=True)`. The `st.error()` calls display a static, user-safe message.

**M-2 — Startup model-training failure silently swallowed**
- File: `src/api.py` `_startup()` handler
- Risk: If the fallback RSF model failed to train at startup (e.g., due to a corrupted dependency), the failure was silently discarded, making the root cause invisible in production logs.
- Fix: `logger.error("Startup model training failed: %s", exc, exc_info=True)` added before the empty fallback path.

**M-3 — Incomplete `.gitignore` — PyTorch and broad model artifact patterns missing**
- File: `.gitignore`
- Risk: PyTorch checkpoint files (`*.pt`, `*.pth`) were not ignored. If a developer saves a model with `torch.save()`, it could be accidentally committed. The `data/` and `models/` directories were partially covered (specific extensions only) but not as whole directories.
- Fix: Added top-level patterns `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, and directory-level exclusions for `data/` and `models/`. Added `.env.*` and `*.env` to catch variant env-file names.

### LOW

**L-1 — `competing_risks.plot_competing_risks()` accepts raw `output_path: str`**
- File: `src/competing_risks.py`
- Risk: This is an internal (non-HTTP) function called only from pipeline code. The path is not user-supplied at runtime in the current codebase. No path traversal is exploitable through the API. Flagged as low-severity for awareness.
- Recommendation: If this function is ever exposed to user-supplied paths (e.g., via a future API endpoint), validate that the resolved path lies within a designated output directory using `Path(output_path).resolve().is_relative_to(ALLOWED_OUTPUT_DIR)`.

**L-2 — No authentication on `/predict` and `/models` endpoints**
- File: `src/api.py`
- Risk: The prediction endpoint is unauthenticated. In the current deployment context (local/research dashboard), this is acceptable. In any production or Internet-facing deployment, authenticated access control is required.
- Recommendation: Add API key authentication via `fastapi.Security` with `APIKeyHeader` before any Internet-facing deployment.

---

## Items Confirmed Clean

| Check | Result |
|---|---|
| Hardcoded secrets/API keys/passwords | None found |
| `pickle.load()` / `joblib.load()` from user-supplied paths | Not present — models trained in-process only |
| `eval()` / `exec()` in Python source | Not present |
| Shell command injection (`subprocess`, `os.system`) | Not present |
| PHI/PII field logging | Not present — no patient identifiers in the data model |
| SQL injection | Not applicable — no database layer |
| PyTorch `torch.load()` from user-supplied paths | Not present |
| `dangerouslySetInnerHTML` / `innerHTML` equivalent | Not applicable |
| Committed `.env` or data files | None found in git history |
| Dependency versions (fastapi 0.109.2, pydantic 2.13.4, numpy 1.26.4) | No known critical CVEs at audit date |

---

## Status

All CRITICAL and HIGH issues resolved. All MEDIUM issues resolved. LOW-severity items documented with remediation guidance for future reference.
