# Task 38: MCP Security & Compliance Documentation

## Overview

Task 38 introduces a unified **Security & Compliance layer** for every Model Context Protocol (MCP) in the Alita-KGoT Enhanced system.  The implementation focuses on four pillars:

1. **Static & Dependency Security Analysis** – automated scans in CI/CD
2. **Secure, Isolated Execution** – Docker/Sarus-style sandboxes with resource limits
3. **Secrets & Permission Management** – HashiCorp Vault integration and sandbox path validation
4. **Comprehensive Audit Logging** – immutable, Winston-ingestable JSON logs

The core logic lives in `security/mcp_security_compliance.py` and a new GitHub Actions workflow `/.github/workflows/security_scans.yml`.

---

## 1 ️⃣  Static & Dependency Security Analysis

| Scan Type | Tool | Workflow File | Trigger |
|-----------|------|---------------|---------|
| SAST      | **Bandit** | `.github/workflows/security_scans.yml` | `push`, `pull_request` (\*.py) |
| SCA       | **Safety** | `.github/workflows/security_scans.yml` | same |

### Key Points
* **Bandit** scans *all* Python files in `alita-kgot-enhanced` & sub-modules, returning SARIF to GitHub's "Security" tab.
* **Safety** checks `requirements.txt` for CVEs and uploads a JSON report as a build artifact.
* Workflow is self-contained (installs tools via `pip`) and requires no additional secrets.

---

## 2 ️⃣  Secure, Isolated Execution

`MCPSecurityCompliance.execute_in_container()` now:

* Spins an **ephemeral** container using the official `python:3.9-slim` image.
* Disables networking (`network_disabled=True`).
* Enforces 256 MiB RAM (`mem_limit`) and 50 % CPU (`cpu_quota`).
* Auto-removes the container on exit (`remove=True`).
* Gracefully degrades if the Docker SDK is unavailable.

```python
result = security.execute_in_container("print('hello world')", timeout=30)
```

Returned structure:
```jsonc
{
  "exit_code": 0,
  "stdout": "hello world",
  "stderr": ""
}
```

> ⚠️  For high-performance or GPU workloads swap the image or limits in code.

---

## 3 ️⃣  Secrets & Permission Management

### HashiCorp Vault Integration
* **VaultSecretsManager** auto-discovers `VAULT_ADDR` & `VAULT_TOKEN` env vars.
* Falls back to an in-memory **MockSecretsManager** during development.
* Usage:

```python
api_key = security.get_api_key('my_service')  # Transparently pulls from Vault
```

### Sandbox Path Validation
`security.validate_path(path, sandbox_dir="/sandbox")` enforces file MCPs operate **only** inside the designated sandbox.

```python
if not security.validate_path(user_supplied_path):
    raise PermissionError('Path outside sandbox.')
```

---

## 4 ️⃣  Immutable Audit Logging

Every MCP should call `log_mcp_operation()` after execution:

```python
security.log_mcp_operation(
    mcp_name='file_operations_mcp',
    user='alice',
    params={'target': '/sandbox/report.txt'},  # 🚫 redact sensitive data first!
    result={'status': 'success'}
)
```

Log entries are written to `logs/system/audit.log` in **single-line JSON** for easy ingestion by Winston-based collectors.

Example log line:
```json
{"timestamp":"2025-07-05T14:22:17.345Z","mcp":"file_operations_mcp","user":"alice","params":{"target":"/sandbox/report.txt"},"result":{"status":"success"}}
```

> The central logging service can ship this file to ELK / Grafana Loki for long-term retention.

---

## Usage Checklist for MCP Developers

1. **Import the helper**:
   ```python
   from security.mcp_security_compliance import MCPSecurityCompliance
   security = MCPSecurityCompliance()
   ```
2. **Static analysis** (optional during local dev):
   ```bash
   bandit -r your_mcp.py -f txt | cat
   safety check -r requirements.txt | cat
   ```
3. **Run code in sandbox** if your MCP executes user code.
4. **Validate external paths** with `validate_path()`.
5. **Retrieve secrets** via Vault integration – never hard-code keys.
6. **Emit audit log** for each MCP run.

---

## Future Enhancements
* **OPA / Rego Policy Enforcement** – fine-grained runtime policies.
* **Automated Secret Rotation** – scheduled Vault rotations with notification.
* **Real Sarus Runtime** – swap Docker for Sarus where HPC isolation is required.
* **SBOM Generation** – produce Software Bill of Materials after each build.

---

© 2025 Alita-KGoT Enhanced Team – Security & Compliance Module v1.0 