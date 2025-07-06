#!/usr/bin/env python3
"""
Security Gate for Production Deployment
=======================================

This module implements comprehensive security scanning and compliance
verification as a quality gate in the production deployment pipeline.

Features:
- Container vulnerability scanning
- Secret detection and validation
- Compliance framework verification
- Security policy enforcement
- Risk assessment and reporting
- Integration with security tools

Author: Alita KGoT Enhanced Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import re
import hashlib
import tempfile

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.logging.winston_config import setup_winston_logging
from security.mcp_security_compliance import MCPSecurityCompliance

@dataclass
class VulnerabilityReport:
    """Container vulnerability scan report"""
    image: str
    scan_time: datetime
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    unknown_count: int
    vulnerabilities: List[Dict[str, Any]]
    passed: bool
    risk_score: float

@dataclass
class SecretScanReport:
    """Secret scanning report"""
    path: str
    scan_time: datetime
    secrets_found: List[Dict[str, Any]]
    passed: bool
    risk_level: str

@dataclass
class ComplianceReport:
    """Security compliance report"""
    framework: str
    scan_time: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    compliance_score: float
    details: List[Dict[str, Any]]
    passed: bool

@dataclass
class SecurityAssessment:
    """Overall security assessment"""
    timestamp: datetime
    version: str
    environment: str
    vulnerability_reports: List[VulnerabilityReport]
    secret_reports: List[SecretScanReport]
    compliance_reports: List[ComplianceReport]
    overall_score: float
    risk_level: str
    passed: bool
    recommendations: List[str]

class SecurityGate:
    """
    Security gate that enforces security policies and compliance
    requirements before allowing deployment to proceed.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the security gate
        
        Args:
            config_path: Path to security configuration
        """
        self.logger = setup_winston_logging(
            'security_gate',
            log_file='logs/deployment/security_gate.log'
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize security compliance framework
        self.compliance = MCPSecurityCompliance()
        
        # Security tools configuration
        self.tools = {
            'trivy': self._check_trivy_available(),
            'semgrep': self._check_semgrep_available(),
            'cosign': self._check_cosign_available()
        }
        
        # Vulnerability database update
        self._update_vulnerability_db()
        
        self.logger.info("Security gate initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load security gate configuration"""
        default_config = {
            'vulnerability_scanning': {
                'enabled': True,
                'severity_threshold': 'MEDIUM',  # UNKNOWN, LOW, MEDIUM, HIGH, CRITICAL
                'max_critical': 0,
                'max_high': 5,
                'max_medium': 20,
                'fail_on_timeout': True,
                'timeout': 600,  # 10 minutes
                'scanners': ['trivy']
            },
            'secret_scanning': {
                'enabled': True,
                'patterns': [
                    r'(?i)api[_-]?key[_-]?[:=]\s*["\']?([a-z0-9]{32,})["\']?',
                    r'(?i)secret[_-]?key[_-]?[:=]\s*["\']?([a-z0-9]{32,})["\']?',
                    r'(?i)password[_-]?[:=]\s*["\']?([a-z0-9]{8,})["\']?',
                    r'(?i)token[_-]?[:=]\s*["\']?([a-z0-9]{20,})["\']?',
                    r'-----BEGIN\s+[A-Z\s]+PRIVATE\s+KEY-----',
                    r'(?i)aws[_-]?access[_-]?key[_-]?id[_-]?[:=]\s*["\']?([A-Z0-9]{20})["\']?',
                    r'(?i)aws[_-]?secret[_-]?access[_-]?key[_-]?[:=]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'
                ],
                'exclude_paths': [
                    'tests/',
                    '*.test.js',
                    '*.spec.py',
                    'node_modules/',
                    '.git/',
                    '*.md'
                ],
                'fail_on_secrets': True
            },
            'compliance': {
                'enabled': True,
                'frameworks': ['CIS', 'NIST', 'SOC2'],
                'minimum_score': 0.8,  # 80%
                'required_checks': [
                    'container_security',
                    'network_policies',
                    'rbac_policies',
                    'resource_limits',
                    'image_signing'
                ]
            },
            'image_signing': {
                'enabled': True,
                'require_signature': False,  # Set to true for production
                'trusted_signers': [],
                'cosign_public_key': None
            },
            'policy_enforcement': {
                'deny_root_containers': True,
                'require_non_root_user': True,
                'deny_privileged_containers': True,
                'require_read_only_filesystem': False,
                'deny_host_network': True,
                'deny_host_pid': True,
                'deny_host_ipc': True,
                'require_security_context': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge configurations
            self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _check_trivy_available(self) -> bool:
        """Check if Trivy is available"""
        try:
            subprocess.run(['trivy', '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning("Trivy not available")
            return False
    
    def _check_semgrep_available(self) -> bool:
        """Check if Semgrep is available"""
        try:
            subprocess.run(['semgrep', '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning("Semgrep not available")
            return False
    
    def _check_cosign_available(self) -> bool:
        """Check if Cosign is available"""
        try:
            subprocess.run(['cosign', 'version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning("Cosign not available")
            return False
    
    def _update_vulnerability_db(self) -> None:
        """Update vulnerability database"""
        if self.tools['trivy']:
            try:
                self.logger.info("Updating Trivy vulnerability database")
                subprocess.run(['trivy', 'image', '--download-db-only'], 
                             capture_output=True, check=True, timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.logger.warning(f"Failed to update vulnerability database: {e}")
    
    async def run_security_assessment(
        self,
        images: List[str],
        source_paths: List[str],
        version: str,
        environment: str
    ) -> SecurityAssessment:
        """
        Run comprehensive security assessment
        
        Args:
            images: List of container images to scan
            source_paths: List of source code paths to scan
            version: Version being assessed
            environment: Target environment
            
        Returns:
            Security assessment results
        """
        self.logger.info(f"Starting security assessment for version {version}")
        
        assessment_start = time.time()
        
        # Initialize assessment
        assessment = SecurityAssessment(
            timestamp=datetime.now(),
            version=version,
            environment=environment,
            vulnerability_reports=[],
            secret_reports=[],
            compliance_reports=[],
            overall_score=0.0,
            risk_level="UNKNOWN",
            passed=False,
            recommendations=[]
        )
        
        try:
            # 1. Container vulnerability scanning
            if self.config.get('vulnerability_scanning', {}).get('enabled', True):
                self.logger.info("Running container vulnerability scans")
                vulnerability_reports = await self._scan_container_vulnerabilities(images)
                assessment.vulnerability_reports = vulnerability_reports
            
            # 2. Secret scanning
            if self.config.get('secret_scanning', {}).get('enabled', True):
                self.logger.info("Running secret detection scans")
                secret_reports = await self._scan_secrets(source_paths)
                assessment.secret_reports = secret_reports
            
            # 3. Compliance verification
            if self.config.get('compliance', {}).get('enabled', True):
                self.logger.info("Running compliance verification")
                compliance_reports = await self._verify_compliance(environment)
                assessment.compliance_reports = compliance_reports
            
            # 4. Image signature verification
            if self.config.get('image_signing', {}).get('enabled', True):
                self.logger.info("Verifying image signatures")
                await self._verify_image_signatures(images)
            
            # 5. Calculate overall assessment
            self._calculate_assessment_score(assessment)
            
            duration = time.time() - assessment_start
            self.logger.info(f"Security assessment completed in {duration:.2f} seconds")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Security assessment failed: {str(e)}")
            assessment.passed = False
            assessment.risk_level = "CRITICAL"
            assessment.recommendations.append(f"Security assessment failed: {str(e)}")
            return assessment
    
    async def _scan_container_vulnerabilities(
        self,
        images: List[str]
    ) -> List[VulnerabilityReport]:
        """
        Scan container images for vulnerabilities
        
        Args:
            images: List of container images to scan
            
        Returns:
            List of vulnerability reports
        """
        reports = []
        config = self.config.get('vulnerability_scanning', {})
        
        for image in images:
            self.logger.info(f"Scanning vulnerabilities in image: {image}")
            
            try:
                # Run Trivy scan
                report = await self._run_trivy_scan(image, config)
                reports.append(report)
                
            except Exception as e:
                self.logger.error(f"Vulnerability scan failed for {image}: {str(e)}")
                
                # Create failed report
                failed_report = VulnerabilityReport(
                    image=image,
                    scan_time=datetime.now(),
                    total_vulnerabilities=0,
                    critical_count=0,
                    high_count=0,
                    medium_count=0,
                    low_count=0,
                    unknown_count=0,
                    vulnerabilities=[],
                    passed=False,
                    risk_score=1.0  # Maximum risk for failed scans
                )
                reports.append(failed_report)
        
        return reports
    
    async def _run_trivy_scan(
        self,
        image: str,
        config: Dict[str, Any]
    ) -> VulnerabilityReport:
        """Run Trivy vulnerability scan"""
        
        # Create temporary file for JSON output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            # Build Trivy command
            cmd = [
                'trivy', 'image',
                '--format', 'json',
                '--output', output_file,
                '--timeout', str(config.get('timeout', 600)) + 's',
                image
            ]
            
            # Add severity filter if specified
            severity_threshold = config.get('severity_threshold', 'MEDIUM')
            if severity_threshold != 'UNKNOWN':
                cmd.extend(['--severity', severity_threshold + ',HIGH,CRITICAL'])
            
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Execute scan
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=config.get('timeout', 600)
            )
            
            if process.returncode != 0 and config.get('fail_on_timeout', True):
                raise Exception(f"Trivy scan failed: {stderr.decode()}")
            
            # Parse results
            with open(output_file, 'r') as f:
                scan_results = json.load(f)
            
            return self._parse_trivy_results(image, scan_results, config)
            
        finally:
            # Cleanup temporary file
            try:
                os.unlink(output_file)
            except OSError:
                pass
    
    def _parse_trivy_results(
        self,
        image: str,
        results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> VulnerabilityReport:
        """Parse Trivy scan results"""
        
        vulnerabilities = []
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'UNKNOWN': 0
        }
        
        # Extract vulnerabilities from results
        for result in results.get('Results', []):
            for vuln in result.get('Vulnerabilities', []):
                severity = vuln.get('Severity', 'UNKNOWN')
                severity_counts[severity] += 1
                
                vulnerabilities.append({
                    'id': vuln.get('VulnerabilityID', ''),
                    'severity': severity,
                    'title': vuln.get('Title', ''),
                    'description': vuln.get('Description', ''),
                    'package': vuln.get('PkgName', ''),
                    'installed_version': vuln.get('InstalledVersion', ''),
                    'fixed_version': vuln.get('FixedVersion', ''),
                    'cvss_score': vuln.get('CVSS', {}).get('nvd', {}).get('V3Score', 0.0)
                })
        
        # Calculate risk score
        risk_score = self._calculate_vulnerability_risk_score(severity_counts)
        
        # Determine if scan passed
        max_critical = config.get('max_critical', 0)
        max_high = config.get('max_high', 5)
        max_medium = config.get('max_medium', 20)
        
        passed = (
            severity_counts['CRITICAL'] <= max_critical and
            severity_counts['HIGH'] <= max_high and
            severity_counts['MEDIUM'] <= max_medium
        )
        
        return VulnerabilityReport(
            image=image,
            scan_time=datetime.now(),
            total_vulnerabilities=sum(severity_counts.values()),
            critical_count=severity_counts['CRITICAL'],
            high_count=severity_counts['HIGH'],
            medium_count=severity_counts['MEDIUM'],
            low_count=severity_counts['LOW'],
            unknown_count=severity_counts['UNKNOWN'],
            vulnerabilities=vulnerabilities,
            passed=passed,
            risk_score=risk_score
        )
    
    def _calculate_vulnerability_risk_score(
        self,
        severity_counts: Dict[str, int]
    ) -> float:
        """Calculate risk score based on vulnerability severities"""
        
        # Weighted scoring system
        weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.7,
            'MEDIUM': 0.4,
            'LOW': 0.1,
            'UNKNOWN': 0.2
        }
        
        total_score = sum(
            count * weights[severity] 
            for severity, count in severity_counts.items()
        )
        
        # Normalize to 0-1 scale (assuming max of 100 vulnerabilities)
        max_possible_score = 100 * weights['CRITICAL']
        normalized_score = min(total_score / max_possible_score, 1.0)
        
        return normalized_score
    
    async def _scan_secrets(
        self,
        source_paths: List[str]
    ) -> List[SecretScanReport]:
        """
        Scan source code for hardcoded secrets
        
        Args:
            source_paths: List of source code paths to scan
            
        Returns:
            List of secret scan reports
        """
        reports = []
        config = self.config.get('secret_scanning', {})
        patterns = config.get('patterns', [])
        exclude_paths = config.get('exclude_paths', [])
        
        for source_path in source_paths:
            if not os.path.exists(source_path):
                continue
                
            self.logger.info(f"Scanning for secrets in: {source_path}")
            
            try:
                secrets_found = []
                
                # Scan files recursively
                for root, dirs, files in os.walk(source_path):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not self._is_excluded_path(
                        os.path.join(root, d), exclude_paths
                    )]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip excluded files
                        if self._is_excluded_path(file_path, exclude_paths):
                            continue
                        
                        # Scan file for secrets
                        file_secrets = await self._scan_file_for_secrets(
                            file_path, patterns
                        )
                        secrets_found.extend(file_secrets)
                
                # Determine risk level and pass status
                risk_level = "LOW"
                if secrets_found:
                    # Assess risk based on secret types
                    high_risk_patterns = ['private_key', 'api_key', 'aws_']
                    if any(any(pattern in secret['type'].lower() for pattern in high_risk_patterns)
                           for secret in secrets_found):
                        risk_level = "HIGH"
                    else:
                        risk_level = "MEDIUM"
                
                passed = len(secrets_found) == 0 or not config.get('fail_on_secrets', True)
                
                report = SecretScanReport(
                    path=source_path,
                    scan_time=datetime.now(),
                    secrets_found=secrets_found,
                    passed=passed,
                    risk_level=risk_level
                )
                
                reports.append(report)
                
            except Exception as e:
                self.logger.error(f"Secret scanning failed for {source_path}: {str(e)}")
                
                # Create failed report
                failed_report = SecretScanReport(
                    path=source_path,
                    scan_time=datetime.now(),
                    secrets_found=[],
                    passed=False,
                    risk_level="HIGH"  # Fail-safe to high risk
                )
                reports.append(failed_report)
        
        return reports
    
    def _is_excluded_path(self, path: str, exclude_patterns: List[str]) -> bool:
        """Check if path should be excluded from scanning"""
        for pattern in exclude_patterns:
            if pattern in path:
                return True
        return False
    
    async def _scan_file_for_secrets(
        self,
        file_path: str,
        patterns: List[str]
    ) -> List[Dict[str, Any]]:
        """Scan individual file for secrets"""
        secrets = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for i, pattern in enumerate(patterns):
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        secret = {
                            'file': file_path,
                            'line': line_num,
                            'type': f"pattern_{i}",
                            'pattern': pattern,
                            'match': match.group(0)[:50] + '...' if len(match.group(0)) > 50 else match.group(0),
                            'severity': self._assess_secret_severity(pattern, match.group(0))
                        }
                        secrets.append(secret)
                        
        except Exception as e:
            self.logger.warning(f"Failed to scan file {file_path}: {str(e)}")
        
        return secrets
    
    def _assess_secret_severity(self, pattern: str, match: str) -> str:
        """Assess severity of detected secret"""
        high_risk_indicators = ['private_key', 'aws_secret', 'api_key']
        medium_risk_indicators = ['password', 'token']
        
        pattern_lower = pattern.lower()
        match_lower = match.lower()
        
        for indicator in high_risk_indicators:
            if indicator in pattern_lower or indicator in match_lower:
                return "HIGH"
        
        for indicator in medium_risk_indicators:
            if indicator in pattern_lower or indicator in match_lower:
                return "MEDIUM"
        
        return "LOW"
    
    async def _verify_compliance(self, environment: str) -> List[ComplianceReport]:
        """
        Verify compliance with security frameworks
        
        Args:
            environment: Target environment
            
        Returns:
            List of compliance reports
        """
        reports = []
        config = self.config.get('compliance', {})
        frameworks = config.get('frameworks', ['CIS'])
        
        for framework in frameworks:
            self.logger.info(f"Verifying {framework} compliance")
            
            try:
                # Run compliance assessment using security framework
                assessment_results = await self.compliance.run_compliance_assessment()
                
                # Parse results for specific framework
                framework_results = assessment_results.get(framework.lower(), {})
                
                total_checks = framework_results.get('total_checks', 0)
                passed_checks = framework_results.get('passed_checks', 0)
                failed_checks = framework_results.get('failed_checks', 0)
                warnings = framework_results.get('warnings', 0)
                
                compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0
                minimum_score = config.get('minimum_score', 0.8)
                passed = compliance_score >= minimum_score
                
                report = ComplianceReport(
                    framework=framework,
                    scan_time=datetime.now(),
                    total_checks=total_checks,
                    passed_checks=passed_checks,
                    failed_checks=failed_checks,
                    warnings=warnings,
                    compliance_score=compliance_score,
                    details=framework_results.get('details', []),
                    passed=passed
                )
                
                reports.append(report)
                
            except Exception as e:
                self.logger.error(f"Compliance verification failed for {framework}: {str(e)}")
                
                # Create failed report
                failed_report = ComplianceReport(
                    framework=framework,
                    scan_time=datetime.now(),
                    total_checks=0,
                    passed_checks=0,
                    failed_checks=1,
                    warnings=0,
                    compliance_score=0.0,
                    details=[{"check": "framework_assessment", "status": "failed", "error": str(e)}],
                    passed=False
                )
                reports.append(failed_report)
        
        return reports
    
    async def _verify_image_signatures(self, images: List[str]) -> bool:
        """
        Verify container image signatures
        
        Args:
            images: List of images to verify
            
        Returns:
            True if all signatures are valid
        """
        config = self.config.get('image_signing', {})
        
        if not config.get('require_signature', False):
            self.logger.info("Image signature verification not required")
            return True
        
        if not self.tools['cosign']:
            self.logger.warning("Cosign not available for signature verification")
            return False
        
        for image in images:
            self.logger.info(f"Verifying signature for image: {image}")
            
            try:
                # Verify signature using Cosign
                cmd = ['cosign', 'verify', image]
                
                # Add public key if specified
                public_key = config.get('cosign_public_key')
                if public_key:
                    cmd.extend(['--key', public_key])
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    self.logger.error(f"Signature verification failed for {image}: {stderr.decode()}")
                    return False
                
                self.logger.info(f"Signature verified for {image}")
                
            except Exception as e:
                self.logger.error(f"Signature verification error for {image}: {str(e)}")
                return False
        
        return True
    
    def _calculate_assessment_score(self, assessment: SecurityAssessment) -> None:
        """Calculate overall security assessment score"""
        
        scores = []
        
        # Vulnerability scanning score
        if assessment.vulnerability_reports:
            vuln_scores = [1.0 - report.risk_score for report in assessment.vulnerability_reports 
                          if report.passed]
            if vuln_scores:
                scores.append(sum(vuln_scores) / len(vuln_scores))
            else:
                scores.append(0.0)  # All vulnerability scans failed
        
        # Secret scanning score
        if assessment.secret_reports:
            secret_scores = [1.0 if report.passed else 0.0 for report in assessment.secret_reports]
            scores.append(sum(secret_scores) / len(secret_scores))
        
        # Compliance score
        if assessment.compliance_reports:
            compliance_scores = [report.compliance_score for report in assessment.compliance_reports]
            scores.append(sum(compliance_scores) / len(compliance_scores))
        
        # Calculate overall score
        if scores:
            assessment.overall_score = sum(scores) / len(scores)
        else:
            assessment.overall_score = 0.0
        
        # Determine risk level and pass status
        if assessment.overall_score >= 0.9:
            assessment.risk_level = "LOW"
        elif assessment.overall_score >= 0.7:
            assessment.risk_level = "MEDIUM"
        elif assessment.overall_score >= 0.5:
            assessment.risk_level = "HIGH"
        else:
            assessment.risk_level = "CRITICAL"
        
        # Determine overall pass status
        vulnerability_passed = all(report.passed for report in assessment.vulnerability_reports)
        secrets_passed = all(report.passed for report in assessment.secret_reports)
        compliance_passed = all(report.passed for report in assessment.compliance_reports)
        
        assessment.passed = vulnerability_passed and secrets_passed and compliance_passed
        
        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(assessment)
    
    def _generate_recommendations(self, assessment: SecurityAssessment) -> List[str]:
        """Generate security recommendations based on assessment"""
        recommendations = []
        
        # Vulnerability recommendations
        for report in assessment.vulnerability_reports:
            if not report.passed:
                if report.critical_count > 0:
                    recommendations.append(
                        f"🚨 CRITICAL: Fix {report.critical_count} critical vulnerabilities in {report.image}"
                    )
                if report.high_count > 0:
                    recommendations.append(
                        f"⚠️ HIGH: Address {report.high_count} high-severity vulnerabilities in {report.image}"
                    )
        
        # Secret recommendations
        for report in assessment.secret_reports:
            if not report.passed and report.secrets_found:
                recommendations.append(
                    f"🔐 Remove {len(report.secrets_found)} hardcoded secrets from {report.path}"
                )
        
        # Compliance recommendations
        for report in assessment.compliance_reports:
            if not report.passed:
                recommendations.append(
                    f"📋 Improve {report.framework} compliance score from "
                    f"{report.compliance_score:.1%} to required minimum"
                )
        
        # General recommendations
        if assessment.risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append(
                "🛡️ Consider additional security hardening before deployment"
            )
        
        return recommendations

def main():
    """Main entry point for security gate"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Gate for Production Deployment')
    parser.add_argument('--images', nargs='+', required=True, help='Container images to scan')
    parser.add_argument('--source-paths', nargs='+', default=['.'], help='Source code paths to scan')
    parser.add_argument('--version', required=True, help='Version being assessed')
    parser.add_argument('--environment', required=True, help='Target environment')
    parser.add_argument('--config', help='Security configuration file')
    parser.add_argument('--output', help='Output file for assessment results')
    
    args = parser.parse_args()
    
    async def run_assessment():
        # Initialize security gate
        gate = SecurityGate(args.config)
        
        # Run security assessment
        assessment = await gate.run_security_assessment(
            args.images,
            args.source_paths,
            args.version,
            args.environment
        )
        
        # Output results
        results = asdict(assessment)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            print(json.dumps(results, indent=2, default=str))
        
        # Exit with appropriate code
        sys.exit(0 if assessment.passed else 1)
    
    asyncio.run(run_assessment())

if __name__ == "__main__":
    main() 