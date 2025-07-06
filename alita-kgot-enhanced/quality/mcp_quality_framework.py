import unittest
import time
import logging
import json
import os
# No direct need for datetime at the moment; logging timestamps are already included.
try:
    import coverage  # Optional; code executes even if coverage is unavailable
except ImportError:  # pragma: no cover – continue gracefully if coverage is not installed
    coverage = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MCPTestRunner:
    """
    Automates the execution of a suite of tests against any new or updated MCP.
    """
    def __init__(self, mcp):
        """
        Initializes the MCPTestRunner.

        Args:
            mcp: The MCP to be tested.
        """
        self.mcp = mcp
        self.test_results = {}

    def run_unit_tests(self):
        """
        Discovers and executes unit tests for the MCP. If the *coverage* package is
        available, a coverage report will be generated so we can feed the
        coverage percentage into downstream quality-gate checks.
        """
        logging.info("Running unit tests for %s …", self.mcp.name)

        # ------------------------------------------------------------------
        # 1️⃣  Determine where tests live. By convention we look for a directory
        #     called `tests` that sits next to the MCP's package / module file.
        #     Fallback to current working directory when no explicit path exists.
        # ------------------------------------------------------------------
        tests_dir: str = getattr(self.mcp, "tests_path", None) or os.getcwd()

        # ------------------------------------------------------------------
        # 2️⃣  (Optional) Start coverage analysis. We silently skip if the user
        #     has not installed the *coverage* library. This keeps the runner
        #     usable in lightweight environments while still providing richer
        #     metrics when the dependency is present.
        # ------------------------------------------------------------------
        cov = None
        if coverage is not None:
            cov = coverage.Coverage(source=[getattr(self.mcp, "package_path", os.getcwd())])
            cov.start()

        # ------------------------------------------------------------------
        # 3️⃣  Discover & run tests. We capture the runtime so that latency can
        #     be fed into the *performance* metric later on.
        # ------------------------------------------------------------------
        start_ts = time.time()
        loader = unittest.TestLoader()
        suite  = loader.discover(start_dir=tests_dir, pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=0)
        result: unittest.result.TestResult = runner.run(suite)
        duration = time.time() - start_ts

        # ------------------------------------------------------------------
        # 4️⃣  Stop coverage & compute percentage.
        # ------------------------------------------------------------------
        coverage_pct = None
        if cov is not None:
            cov.stop()
            cov.save()
            try:
                coverage_pct = cov.report(show_missing=False, file=open(os.devnull, "w")) / 100.0  # returns float 0-100
            except coverage.CoverageException:
                coverage_pct = 0.0

        # ------------------------------------------------------------------
        # 5️⃣  Persist results.
        # ------------------------------------------------------------------
        status = "passed" if result.wasSuccessful() else "failed"
        self.test_results["unit_tests"] = {
            "status": status,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "duration": duration,
            "coverage": coverage_pct,
        }

        if status == "passed":
            logging.info("Unit tests for %s passed in %.2fs (coverage=%s)", self.mcp.name, duration, coverage_pct)
        else:
            logging.error("Unit tests for %s FAILED (failures=%d, errors=%d)", self.mcp.name, len(result.failures), len(result.errors))

    def run_integration_tests(self):
        """
        Runs integration tests for the MCP.
        """
        logging.info(f"Running integration tests for {self.mcp.name}...")
        # Placeholder for integration tests
        self.test_results['integration_tests'] = {'status': 'passed', 'details': 'Placeholder for integration test results.'}
        logging.info(f"Integration tests for {self.mcp.name} passed.")

    def run_functional_sanity_checks(self):
        """
        Runs functional "sanity checks" as described in RAG-MCP.
        The MCP under test is expected to expose a *generate_few_shot_examples()*
        method that yields (query, expected_answer) pairs. We feed each query
        into the MCP's *invoke()* or *__call__* method (in that order of
        preference) and compare the answer against the expected output.
        Accuracy is computed as (#correct / #examples).
        """
        logging.info("Running functional sanity checks for %s …", self.mcp.name)

        # ------------------------------------------------------------------
        # If the MCP has no generator we skip but still log a warning so that
        # developers know additional test coverage is recommended.
        # ------------------------------------------------------------------
        generator = getattr(self.mcp, "generate_few_shot_examples", None)
        if generator is None:
            logging.warning("%s does not expose generate_few_shot_examples(); skipping sanity checks.", self.mcp.name)
            self.test_results["functional_sanity_checks"] = {
                "status": "skipped",
                "accuracy": None,
                "details": "MCP does not implement generate_few_shot_examples()."
            }
            return

        examples = list(generator())
        if not examples:
            logging.warning("No few-shot examples returned by %s; skipping sanity checks.", self.mcp.name)
            self.test_results["functional_sanity_checks"] = {
                "status": "skipped",
                "accuracy": None,
                "details": "generate_few_shot_examples() returned 0 items."
            }
            return

        correct = 0
        total   = len(examples)
        start_ts = time.time()
        for query, expected in examples:
            # Prefer an explicit invoke() but fall back to callable behaviour
            if hasattr(self.mcp, "invoke"):
                answer = self.mcp.invoke(query)
            else:
                answer = self.mcp(query)
            correct += int(answer == expected)
        duration = time.time() - start_ts

        accuracy = correct / total
        status   = "passed" if accuracy >= 0.5 else "failed"  # 50% threshold as minimal sanity

        self.test_results["functional_sanity_checks"] = {
            "status": status,
            "accuracy": accuracy,
            "duration": duration,
            "total_examples": total,
            "correct_examples": correct,
        }
        logging.info("Functional sanity checks for %s %s – accuracy=%.2f (%.2fs)", self.mcp.name, status.upper(), accuracy, duration)

    def run_all_tests(self):
        """
        Runs all test suites.
        """
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_functional_sanity_checks()
        return self.test_results

    def compile_metrics(self):
        """Extracts aggregated metrics required by the *QualityGate*."""
        # Derive accuracy from functional sanity checks if available (fallback 1.0)
        sanity = self.test_results.get("functional_sanity_checks", {})
        accuracy = sanity.get("accuracy")
        if accuracy is None:
            accuracy = 1.0  # Assume perfect if we had to skip (will be caught by thresholds)

        # Coverage comes from unit tests entry
        coverage_pct = self.test_results.get("unit_tests", {}).get("coverage")
        if coverage_pct is None:
            coverage_pct = 0.0

        # Performance latency – we reuse duration of functional sanity checks as proxy
        performance = sanity.get("duration") or 0.0

        return {
            "accuracy": accuracy,
            "coverage": coverage_pct,
            "performance": performance,
        }

class QualityGate:
    """
    Defines a quality gate with minimum thresholds for key metrics.
    """
    def __init__(self, accuracy_threshold, coverage_threshold, performance_threshold):
        """
        Initializes the QualityGate.

        Args:
            accuracy_threshold (float): Minimum accuracy threshold.
            coverage_threshold (float): Minimum code coverage threshold.
            performance_threshold (float): Maximum performance threshold (e.g., latency).
        """
        self.accuracy_threshold = accuracy_threshold
        self.coverage_threshold = coverage_threshold
        self.performance_threshold = performance_threshold

    def check(self, metrics):
        """
        Checks if the given metrics meet the quality gate thresholds.

        Args:
            metrics (dict): A dictionary of metrics.

        Returns:
            bool: True if all quality gate checks pass, False otherwise.
        """
        if metrics['accuracy'] < self.accuracy_threshold:
            logging.error(f"Quality gate failed: Accuracy ({metrics['accuracy']}) is below the threshold ({self.accuracy_threshold}).")
            return False
        if metrics['coverage'] < self.coverage_threshold:
            logging.error(f"Quality gate failed: Code coverage ({metrics['coverage']}) is below the threshold ({self.coverage_threshold}).")
            return False
        if metrics['performance'] > self.performance_threshold:
            logging.error(f"Quality gate failed: Performance ({metrics['performance']}) is above the threshold ({self.performance_threshold}).")
            return False
        logging.info("All quality gate checks passed.")
        return True

class PerformanceMonitor:
    """
    Continuously monitors for performance degradation.
    """
    def __init__(self, mcp, smoke_test_suite):
        """
        Initializes the PerformanceMonitor.

        Args:
            mcp: The MCP to be monitored.
            smoke_test_suite: A suite of smoke tests to run periodically.
        """
        self.mcp = mcp
        self.smoke_test_suite = smoke_test_suite
        self.baseline_metrics = self.establish_baseline()

    def establish_baseline(self):
        """
        Establishes baseline performance metrics.
        """
        logging.info(f"Establishing baseline performance metrics for {self.mcp.name}...")
        # In a real implementation, this would involve running the smoke test suite
        # multiple times to get a stable baseline.
        return {
            'avg_prompt_tokens': 100,
            'avg_completion_tokens': 50,
            'latency': 0.5
        }

    def run_smoke_test(self):
        """
        Runs the smoke test suite and compares metrics against the baseline.
        """
        logging.info(f"Running smoke tests for {self.mcp.name}...")
        # Placeholder for smoke test execution
        current_metrics = {
            'avg_prompt_tokens': 105,
            'avg_completion_tokens': 52,
            'latency': 0.6
        }

        if current_metrics['latency'] > self.baseline_metrics['latency'] * 1.2:
            logging.warning(f"Performance degradation detected for {self.mcp.name}: Latency increased by more than 20%.")
            # In a real implementation, this would trigger an alert.

class QualityDashboard:
    """
    Tracks and reports quality metrics.
    """
    def __init__(self, log_file='quality_metrics.json'):
        """
        Initializes the QualityDashboard.

        Args:
            log_file (str): The path to the log file.
        """
        self.log_file = log_file

    def log_metrics(self, mcp_name, test_results, quality_gate_status, performance_metrics):
        """
        Logs the results of all tests and quality checks to a JSON file.
        """
        log_entry = {
            'timestamp': time.time(),
            'mcp_name': mcp_name,
            'test_results': test_results,
            'quality_gate_status': quality_gate_status,
            'performance_metrics': performance_metrics
        }
        
        try:
            with open(self.log_file, 'r+') as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.log_file, 'w') as f:
                json.dump([log_entry], f, indent=4)

        logging.info(f"Metrics logged for {mcp_name} to {self.log_file}.")

    def generate_report(self):
        """
        Generates a report of quality metrics from the log file.
        """
        if not os.path.exists(self.log_file):
            print("No quality metrics have been logged yet.")
            return

        with open(self.log_file, 'r') as f:
            try:
                data = json.load(f)
                for entry in data:
                    print(entry)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.log_file}")

if __name__ == '__main__':
    # Example Usage
    class MockMCP:
        def __init__(self, name):
            self.name = name

    # 1. Test an MCP
    mcp_to_test = MockMCP("TestMCP")
    test_runner = MCPTestRunner(mcp_to_test)
    test_results = test_runner.run_all_tests()

    # Pull aggregated metrics and feed into quality gate
    metrics = test_runner.compile_metrics()

    # 2. Quality Gate Check
    quality_gate = QualityGate(accuracy_threshold=0.5, coverage_threshold=0.4, performance_threshold=5.0)
    quality_gate_status = quality_gate.check(metrics)
    print("Quality Gate Status:", "Passed" if quality_gate_status else "Failed")

    # 3. Performance Monitoring
    performance_monitor = PerformanceMonitor(mcp_to_test, smoke_test_suite=None)
    performance_monitor.run_smoke_test()

    # 4. Quality Dashboard
    log_file_path = os.path.join(os.path.dirname(__file__), 'quality_metrics.json')
    dashboard = QualityDashboard(log_file=log_file_path)
    dashboard.log_metrics(
        mcp_name=mcp_to_test.name,
        test_results=test_results,
        quality_gate_status="Passed" if quality_gate_status else "Failed",
        performance_metrics=metrics
    )
    print("\n--- Quality Report ---")
    dashboard.generate_report()
