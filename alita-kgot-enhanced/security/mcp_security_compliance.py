# alita-kgot-enhanced/security/mcp_security_compliance.py

# Standard Library
import subprocess
import os
import json
import datetime
import logging
from typing import Dict, Any, Optional

# Third-Party (optional – handled gracefully if missing)
try:
    import docker  # type: ignore
except ImportError:  # pragma: no cover – Docker might be absent in local env
    docker = None

try:
    import hvac  # type: ignore  # HashiCorp Vault client
except ImportError:  # pragma: no cover – Vault client optional
    hvac = None

# ---------------------------------------------------------------------------
# Secrets Management Implementations
# ---------------------------------------------------------------------------

class MockSecretsManager:
    """A lightweight in-memory secrets store used as a fallback when no real
    secrets backend is configured. *Never* use this in production – it exists
    solely to avoid hard-coding credentials during development and tests.
    """

    def __init__(self):
        self._secrets = {
            "api_keys": {"some_service": "fake_api_key_12345"}
        }

    def get_secret(self, secret_name: str):
        return self._secrets.get(secret_name)

class VaultSecretsManager:
    """Secrets manager backed by HashiCorp Vault.

    The client is instantiated only if the *hvac* package is available and the
    necessary environment variables (``VAULT_ADDR`` & ``VAULT_TOKEN``) or
    explicit parameters are provided. Otherwise, callers should fall back to
    :class:`MockSecretsManager`.
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        if hvac is None:  # hvac not installed – cannot leverage Vault
            self._client = None
            return

        self._client = hvac.Client(
            url=url or os.getenv("VAULT_ADDR"),
            token=token or os.getenv("VAULT_TOKEN"),
        )

    def get_secret(self, secret_name: str):
        """Fetches *latest* version of ``secret_name`` from Vault KV v2."""
        if not self._client or not self._client.is_authenticated():
            return None

        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=secret_name
            )
            return response["data"]["data"]  # Vault KV v2 payload location
        except Exception:  # noqa: BLE001 – any Vault exception results in None
            return None

class MCPSecurityCompliance:
    """
    Provides security and compliance functionalities for MCPs.
    """

    def __init__(self, secrets_manager: 'Optional[Any]' = None):
        """Initializes the compliance helper.

        Preference order for secrets backends:
        1. **Custom manager** provided explicitly via *secrets_manager*.
        2. **HashiCorp Vault** via :class:`VaultSecretsManager` if available.
        3. **In-memory mock** via :class:`MockSecretsManager`.
        """

        if secrets_manager is not None:
            self.secrets_manager = secrets_manager
        else:
            # Attempt Vault first; fall back to mock if unavailable
            vault_mgr = VaultSecretsManager()
            self.secrets_manager = (
                vault_mgr if vault_mgr.get_secret("api_keys") else MockSecretsManager()
            )

        # ------------------------------------------------------------------
        # Audit logger – emits JSON lines that can be ingested by Winston
        # running in a side-car / centralized logging service.
        # ------------------------------------------------------------------
        self._logger = logging.getLogger("mcp-audit")
        if not self._logger.handlers:  # Prevent duplicate handlers in notebooks
            self._logger.setLevel(logging.INFO)
            os.makedirs("logs/system", exist_ok=True)
            handler = logging.FileHandler("logs/system/audit.log")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def scan_code(self, file_path: str) -> Dict[str, Any]:
        """
        Performs static analysis security testing (SAST) on a Python file.

        Args:
            file_path: The absolute path to the Python file to scan.

        Returns:
            A dictionary containing the scan results.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        try:
            # Using Bandit for SAST
            result = subprocess.run(
                ["bandit", "-r", file_path, "-f", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            return {"results": result.stdout}
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"error": f"Bandit scan failed: {e}"}

    def check_dependencies(self, requirements_path: str) -> Dict[str, Any]:
        """
        Checks for vulnerabilities in third-party libraries using safety.

        Args:
            requirements_path: The absolute path to the requirements.txt file.

        Returns:
            A dictionary containing the vulnerability scan results.
        """
        if not os.path.exists(requirements_path):
            return {"error": f"File not found: {requirements_path}"}

        try:
            # Using safety for SCA
            result = subprocess.run(
                ["safety", "check", "-r", requirements_path, "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            return {"results": result.stdout}
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"error": f"Safety scan failed: {e}"}

    def execute_in_container(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Runs *arbitrary* Python code inside an **ephemeral** Docker container
        with **strict resource limits**. Networking is disabled to eliminate
        outbound calls unless explicitly allowed.

        ``docker-py`` is leveraged when present; otherwise we gracefully alert
        callers that secure execution cannot proceed.
        """

        if docker is None:
            return {"error": "Docker SDK is unavailable – cannot isolate code."}

        client = docker.from_env()
        try:
            container = client.containers.run(
                image="python:3.9-slim",  # Small, trusted base image
                command=["python", "-c", code],
                network_disabled=True,
                mem_limit="256m",  # Prevents memory exhaustion
                cpu_period=100_000,  # 100 ms scheduling window
                cpu_quota=50_000,   # 50% of a single vCPU
                detach=True,
                stderr=True,
                stdout=True,
                remove=True,  # Auto-clean after exit
            )

            # Wait for completion respecting *timeout*
            try:
                exit_status = container.wait(timeout=timeout)
            except Exception:  # broad catch to ensure container cleanup
                container.kill()
                raise

            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()

            return {
                "exit_code": exit_status.get("StatusCode", -1),
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
        except docker.errors.ContainerError as err:
            return {"error": f"Container execution failed: {err}"}
        except docker.errors.ImageNotFound:
            return {"error": "Required Docker image missing."}
        except docker.errors.DockerException as err:
            return {"error": f"Docker runtime error: {err}"}

    def get_api_key(self, service_name: str) -> str:
        """
        Retrieves an API key from the secrets manager.

        Args:
            service_name: The name of the service requiring the API key.

        Returns:
            The API key.
        """
        api_keys = self.secrets_manager.get_secret("api_keys")
        return api_keys.get(service_name) if api_keys else None

    def validate_path(self, path: str, sandbox_dir: str = "/sandbox") -> bool:
        """Ensures *path* resides inside the allowed *sandbox_dir*.

        Returns ``True`` when compliant, ``False`` otherwise.
        """
        abs_path = os.path.abspath(path)
        return abs_path.startswith(os.path.abspath(sandbox_dir))

    # ------------------------------------------------------------------
    # Audit Logging – JSON Lines compatible with Winston ingestion.
    # ------------------------------------------------------------------
    def log_mcp_operation(
        self,
        mcp_name: str,
        user: str,
        params: Dict[str, Any],
        result: 'Any',
    ) -> None:
        """Persists an **immutable** audit record for each MCP invocation. The
        record is emitted as a single JSON line which can be tailed by a
        Winston-based forwarder to ship logs to the centralized store.
        """

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "mcp": mcp_name,
            "user": user,
            # Sensitive parameters must be redacted by the caller *before* this
            # function is invoked – we do *not* attempt best-effort sanitization
            # here to avoid accidental data leaks.
            "params": params,
            "result": result,
        }

        self._logger.info(json.dumps(log_entry, ensure_ascii=False))

if __name__ == '__main__':
    # Example Usage
    security_compliance = MCPSecurityCompliance()

    # 1. Scan a file for security vulnerabilities
    # Create a dummy file to scan
    dummy_mcp_code = """
import os

def insecure_mcp(path):
    # This is a security risk
    os.system(f"ls {path}")

def secure_mcp(data):
    print(f"Processing data: {data}")
"""
    dummy_file_path = "/tmp/dummy_mcp.py"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_mcp_code)

    scan_results = security_compliance.scan_code(dummy_file_path)
    print("SAST Scan Results:")
    print(scan_results)

    # 2. Check dependencies for vulnerabilities
    # Create a dummy requirements file
    dummy_requirements_path = "/tmp/requirements.txt"
    with open(dummy_requirements_path, "w") as f:
        f.write("requests==2.25.1") # A version with known vulnerabilities

    dependency_scan_results = security_compliance.check_dependencies(dummy_requirements_path)
    print("\nSCA Scan Results:")
    print(dependency_scan_results)

    # 3. Execute code in a secure environment
    secure_code = "print('This code is running in a container.')"
    execution_result = security_compliance.execute_in_container(secure_code)
    print("\nSecure Execution Results:")
    print(execution_result)

    # 4. Retrieve a secret
    api_key = security_compliance.get_api_key("some_service")
    print(f"\nRetrieved API Key: {api_key}")

    # 5. Log an MCP operation
    security_compliance.log_mcp_operation(
        mcp_name="example_mcp",
        user="test_user",
        params={"input": "some_data"},
        result={"output": "some_result"}
    )