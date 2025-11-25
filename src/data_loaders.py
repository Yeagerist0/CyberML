"""
Data Loaders for OSINT Sources
Handles ingestion from honeypot telemetry, Shodan, CVE feeds, GitHub PoC, and threat intel.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class HoneypotLoader:
    """Load and parse honeypot telemetry data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def load(self) -> pd.DataFrame:
        """Load honeypot data from file or generate sample data."""
        if self.data_path.exists() and self.data_path.stat().st_size > 0:
            return self._load_from_file()
        return self._generate_sample_data()
    
    def _load_from_file(self) -> pd.DataFrame:
        """Load from actual honeypot logs (JSON lines or CSV)."""
        if str(self.data_path).endswith('.json'):
            return pd.read_json(self.data_path, lines=True)
        return pd.read_csv(self.data_path)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample honeypot telemetry."""
        np.random.seed(42)
        n_records = 10000
        
        # Time range: last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        dates = pd.date_range(start_date, end_date, periods=n_records)
        
        attack_types = [
            "ssh_bruteforce", "sql_injection", "xss", "rce_attempt",
            "directory_traversal", "file_inclusion", "command_injection",
            "credential_stuffing", "port_scan", "vulnerability_scan"
        ]
        
        # Weighted by realistic frequency
        attack_weights = [0.25, 0.15, 0.12, 0.08, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]
        
        sectors = ["education", "government", "healthcare", "finance", "critical_infrastructure"]
        sector_weights = [0.25, 0.30, 0.15, 0.20, 0.10]
        
        countries = ["CN", "RU", "US", "KP", "IR", "BR", "IN", "NG", "RO", "UA"]
        country_weights = [0.25, 0.20, 0.10, 0.05, 0.08, 0.07, 0.08, 0.05, 0.06, 0.06]
        
        data = {
            "timestamp": dates,
            "source_ip": [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" 
                          for _ in range(n_records)],
            "dest_port": np.random.choice([22, 80, 443, 3389, 8080, 21, 23, 3306, 5432, 27017], n_records),
            "attack_type": np.random.choice(attack_types, n_records, p=attack_weights),
            "target_sector": np.random.choice(sectors, n_records, p=sector_weights),
            "source_country": np.random.choice(countries, n_records, p=country_weights),
            "payload_size": np.random.exponential(500, n_records).astype(int),
            "success": np.random.choice([True, False], n_records, p=[0.05, 0.95]),
            "severity_score": np.random.uniform(1, 10, n_records).round(2)
        }
        
        return pd.DataFrame(data)


class ShodanLoader:
    """Load and parse Shodan snapshot data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def load(self) -> pd.DataFrame:
        """Load Shodan data."""
        if self.data_path.exists() and self.data_path.stat().st_size > 0:
            return self._load_from_file()
        return self._generate_sample_data()
    
    def _load_from_file(self) -> pd.DataFrame:
        if str(self.data_path).endswith('.json'):
            return pd.read_json(self.data_path, lines=True)
        return pd.read_csv(self.data_path)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample Shodan-like exposure data."""
        np.random.seed(43)
        n_records = 5000
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start_date, end_date, periods=n_records)
        
        services = [
            "apache", "nginx", "iis", "openssh", "mysql", "postgresql",
            "mongodb", "redis", "elasticsearch", "jenkins", "kubernetes"
        ]
        
        vulnerabilities = [
            "CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003", 
            "CVE-2023-9999", "CVE-2023-8888", "CVE-2024-1234",
            None, None, None  # Many services have no known vulns
        ]
        
        sectors = ["education", "government", "healthcare", "finance", "critical_infrastructure"]
        
        data = {
            "scan_date": dates,
            "ip": [f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}" 
                   for _ in range(n_records)],
            "port": np.random.choice([22, 80, 443, 8080, 3306, 5432, 6379, 9200, 27017], n_records),
            "service": np.random.choice(services, n_records),
            "version": [f"{np.random.randint(1,10)}.{np.random.randint(0,20)}.{np.random.randint(0,50)}" 
                        for _ in range(n_records)],
            "sector": np.random.choice(sectors, n_records),
            "vulnerability": np.random.choice(vulnerabilities, n_records),
            "exposed_days": np.random.exponential(30, n_records).astype(int),
            "has_authentication": np.random.choice([True, False], n_records, p=[0.7, 0.3]),
            "ssl_enabled": np.random.choice([True, False], n_records, p=[0.6, 0.4])
        }
        
        return pd.DataFrame(data)


class CVELoader:
    """Load and parse CVE feed data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def load(self) -> pd.DataFrame:
        """Load CVE data."""
        if self.data_path.exists() and self.data_path.stat().st_size > 0:
            return self._load_from_file()
        return self._generate_sample_data()
    
    def _load_from_file(self) -> pd.DataFrame:
        if str(self.data_path).endswith('.json'):
            return pd.read_json(self.data_path, lines=True)
        return pd.read_csv(self.data_path)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample CVE data."""
        np.random.seed(44)
        n_records = 2000
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, periods=n_records)
        
        vendors = ["microsoft", "apache", "linux", "oracle", "cisco", "adobe", "vmware", "fortinet"]
        products = ["windows", "httpd", "kernel", "java", "ios", "reader", "esxi", "fortigate"]
        
        attack_vectors = ["network", "local", "adjacent", "physical"]
        attack_vector_weights = [0.6, 0.25, 0.10, 0.05]
        
        data = {
            "published_date": dates,
            "cve_id": [f"CVE-{np.random.choice([2023, 2024, 2025])}-{np.random.randint(1000, 99999)}" 
                       for _ in range(n_records)],
            "vendor": np.random.choice(vendors, n_records),
            "product": np.random.choice(products, n_records),
            "cvss_score": np.clip(np.random.normal(6.5, 2, n_records), 0, 10).round(1),
            "attack_vector": np.random.choice(attack_vectors, n_records, p=attack_vector_weights),
            "exploit_available": np.random.choice([True, False], n_records, p=[0.15, 0.85]),
            "patch_available": np.random.choice([True, False], n_records, p=[0.6, 0.4]),
            "cisa_kev": np.random.choice([True, False], n_records, p=[0.05, 0.95]),  # Known Exploited Vuln
            "description": [f"Vulnerability in {np.random.choice(products)} allowing {np.random.choice(['RCE', 'privilege escalation', 'data disclosure', 'DoS'])}" 
                           for _ in range(n_records)]
        }
        
        return pd.DataFrame(data)


class GitHubPoCLoader:
    """Load GitHub Proof-of-Concept emergence data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def load(self) -> pd.DataFrame:
        """Load GitHub PoC data."""
        if self.data_path.exists() and self.data_path.stat().st_size > 0:
            return self._load_from_file()
        return self._generate_sample_data()
    
    def _load_from_file(self) -> pd.DataFrame:
        if str(self.data_path).endswith('.json'):
            return pd.read_json(self.data_path, lines=True)
        return pd.read_csv(self.data_path)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample GitHub PoC data."""
        np.random.seed(45)
        n_records = 800
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, periods=n_records)
        
        languages = ["python", "go", "ruby", "javascript", "bash", "c", "rust"]
        language_weights = [0.35, 0.20, 0.10, 0.15, 0.10, 0.05, 0.05]
        
        exploit_types = ["rce", "sqli", "xss", "lfi", "auth_bypass", "ssrf", "deserialization"]
        
        data = {
            "discovered_date": dates,
            "cve_id": [f"CVE-{np.random.choice([2023, 2024, 2025])}-{np.random.randint(1000, 99999)}" 
                       for _ in range(n_records)],
            "repo_url": [f"https://github.com/user{np.random.randint(1,1000)}/exploit-{np.random.randint(1,10000)}" 
                        for _ in range(n_records)],
            "stars": np.random.exponential(50, n_records).astype(int),
            "forks": np.random.exponential(10, n_records).astype(int),
            "language": np.random.choice(languages, n_records, p=language_weights),
            "exploit_type": np.random.choice(exploit_types, n_records),
            "weaponized": np.random.choice([True, False], n_records, p=[0.3, 0.7]),
            "days_since_cve": np.random.exponential(30, n_records).astype(int),
            "description": [f"PoC for {np.random.choice(exploit_types)} vulnerability" 
                           for _ in range(n_records)]
        }
        
        return pd.DataFrame(data)


class ThreatIntelLoader:
    """Load threat intelligence blog/report data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def load(self) -> pd.DataFrame:
        """Load threat intel data."""
        if self.data_path.exists() and self.data_path.stat().st_size > 0:
            return self._load_from_file()
        return self._generate_sample_data()
    
    def _load_from_file(self) -> pd.DataFrame:
        if str(self.data_path).endswith('.json'):
            return pd.read_json(self.data_path, lines=True)
        return pd.read_csv(self.data_path)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample threat intel data."""
        np.random.seed(46)
        n_records = 500
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, periods=n_records)
        
        threat_actors = ["APT28", "APT29", "Lazarus", "Sandworm", "FIN7", "Conti", "REvil", "Unknown"]
        actor_weights = [0.15, 0.12, 0.15, 0.10, 0.12, 0.08, 0.08, 0.20]
        
        target_sectors = ["education", "government", "healthcare", "finance", "critical_infrastructure"]
        
        ttps = ["phishing", "supply_chain", "zero_day", "credential_theft", "ransomware", "wiper"]
        
        sources = ["mandiant", "crowdstrike", "microsoft", "google_tag", "kaspersky", "cisa"]
        
        data = {
            "published_date": dates,
            "source": np.random.choice(sources, n_records),
            "threat_actor": np.random.choice(threat_actors, n_records, p=actor_weights),
            "target_sector": np.random.choice(target_sectors, n_records),
            "ttp": np.random.choice(ttps, n_records),
            "severity": np.random.choice(["low", "medium", "high", "critical"], n_records, 
                                         p=[0.1, 0.3, 0.4, 0.2]),
            "confidence": np.random.uniform(0.5, 1.0, n_records).round(2),
            "iocs_count": np.random.exponential(20, n_records).astype(int),
            "summary": [f"Report on {np.random.choice(threat_actors)} targeting {np.random.choice(target_sectors)} sector using {np.random.choice(ttps)}" 
                       for _ in range(n_records)]
        }
        
        return pd.DataFrame(data)


class DataAggregator:
    """Aggregate all data sources into a unified dataset."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.loaders = {
            "honeypot": HoneypotLoader(data_dir / "honeypot"),
            "shodan": ShodanLoader(data_dir / "shodan"),
            "cve": CVELoader(data_dir / "cve"),
            "github_poc": GitHubPoCLoader(data_dir / "github_poc"),
            "threat_intel": ThreatIntelLoader(data_dir / "threat_intel"),
        }
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources."""
        data = {}
        for name, loader in self.loaders.items():
            print(f"Loading {name} data...")
            data[name] = loader.load()
            print(f"  Loaded {len(data[name])} records")
        return data
    
    def create_weekly_aggregation(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate all data sources into weekly feature vectors."""
        # Get date range
        all_dates = []
        for name, df in data.items():
            date_col = self._get_date_column(df)
            if date_col:
                all_dates.extend(pd.to_datetime(df[date_col]).tolist())
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create weekly periods
        weeks = pd.date_range(min_date, max_date, freq='W-MON')
        
        weekly_data = []
        for week_start in weeks:
            week_end = week_start + timedelta(days=7)
            week_features = {"week_start": week_start}
            
            # Aggregate each source
            week_features.update(self._aggregate_honeypot(data["honeypot"], week_start, week_end))
            week_features.update(self._aggregate_shodan(data["shodan"], week_start, week_end))
            week_features.update(self._aggregate_cve(data["cve"], week_start, week_end))
            week_features.update(self._aggregate_github_poc(data["github_poc"], week_start, week_end))
            week_features.update(self._aggregate_threat_intel(data["threat_intel"], week_start, week_end))
            
            weekly_data.append(week_features)
        
        return pd.DataFrame(weekly_data)
    
    def _get_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in a dataframe."""
        date_cols = ["timestamp", "scan_date", "published_date", "discovered_date"]
        for col in date_cols:
            if col in df.columns:
                return col
        return None
    
    def _aggregate_honeypot(self, df: pd.DataFrame, start: datetime, end: datetime) -> Dict:
        """Aggregate honeypot data for a week."""
        mask = (pd.to_datetime(df["timestamp"]) >= start) & (pd.to_datetime(df["timestamp"]) < end)
        week_df = df[mask]
        
        features = {
            "honeypot_total_attacks": len(week_df),
            "honeypot_unique_ips": week_df["source_ip"].nunique() if len(week_df) > 0 else 0,
            "honeypot_success_rate": week_df["success"].mean() if len(week_df) > 0 else 0,
            "honeypot_avg_severity": week_df["severity_score"].mean() if len(week_df) > 0 else 0,
        }
        
        # Per-sector attack counts
        for sector in ["education", "government", "healthcare", "finance", "critical_infrastructure"]:
            features[f"honeypot_{sector}_attacks"] = len(week_df[week_df["target_sector"] == sector])
        
        # Attack type distribution
        for attack in ["ssh_bruteforce", "sql_injection", "rce_attempt", "vulnerability_scan"]:
            features[f"honeypot_{attack}_count"] = len(week_df[week_df["attack_type"] == attack])
        
        return features
    
    def _aggregate_shodan(self, df: pd.DataFrame, start: datetime, end: datetime) -> Dict:
        """Aggregate Shodan exposure data for a week."""
        mask = (pd.to_datetime(df["scan_date"]) >= start) & (pd.to_datetime(df["scan_date"]) < end)
        week_df = df[mask]
        
        features = {
            "shodan_exposed_services": len(week_df),
            "shodan_vulnerable_services": week_df["vulnerability"].notna().sum() if len(week_df) > 0 else 0,
            "shodan_no_auth_count": (~week_df["has_authentication"]).sum() if len(week_df) > 0 else 0,
            "shodan_no_ssl_count": (~week_df["ssl_enabled"]).sum() if len(week_df) > 0 else 0,
            "shodan_avg_exposed_days": week_df["exposed_days"].mean() if len(week_df) > 0 else 0,
        }
        
        # Per-sector exposure
        for sector in ["education", "government", "healthcare", "finance", "critical_infrastructure"]:
            features[f"shodan_{sector}_exposed"] = len(week_df[week_df["sector"] == sector])
        
        return features
    
    def _aggregate_cve(self, df: pd.DataFrame, start: datetime, end: datetime) -> Dict:
        """Aggregate CVE data for a week."""
        mask = (pd.to_datetime(df["published_date"]) >= start) & (pd.to_datetime(df["published_date"]) < end)
        week_df = df[mask]
        
        features = {
            "cve_new_count": len(week_df),
            "cve_critical_count": len(week_df[week_df["cvss_score"] >= 9.0]) if len(week_df) > 0 else 0,
            "cve_high_count": len(week_df[(week_df["cvss_score"] >= 7.0) & (week_df["cvss_score"] < 9.0)]) if len(week_df) > 0 else 0,
            "cve_avg_cvss": week_df["cvss_score"].mean() if len(week_df) > 0 else 0,
            "cve_network_vector_count": len(week_df[week_df["attack_vector"] == "network"]) if len(week_df) > 0 else 0,
            "cve_exploit_available": week_df["exploit_available"].sum() if len(week_df) > 0 else 0,
            "cve_no_patch": (~week_df["patch_available"]).sum() if len(week_df) > 0 else 0,
            "cve_cisa_kev": week_df["cisa_kev"].sum() if len(week_df) > 0 else 0,
        }
        
        return features
    
    def _aggregate_github_poc(self, df: pd.DataFrame, start: datetime, end: datetime) -> Dict:
        """Aggregate GitHub PoC data for a week."""
        mask = (pd.to_datetime(df["discovered_date"]) >= start) & (pd.to_datetime(df["discovered_date"]) < end)
        week_df = df[mask]
        
        features = {
            "poc_new_count": len(week_df),
            "poc_total_stars": week_df["stars"].sum() if len(week_df) > 0 else 0,
            "poc_weaponized_count": week_df["weaponized"].sum() if len(week_df) > 0 else 0,
            "poc_avg_days_since_cve": week_df["days_since_cve"].mean() if len(week_df) > 0 else 0,
            "poc_rce_count": len(week_df[week_df["exploit_type"] == "rce"]) if len(week_df) > 0 else 0,
        }
        
        return features
    
    def _aggregate_threat_intel(self, df: pd.DataFrame, start: datetime, end: datetime) -> Dict:
        """Aggregate threat intel data for a week."""
        mask = (pd.to_datetime(df["published_date"]) >= start) & (pd.to_datetime(df["published_date"]) < end)
        week_df = df[mask]
        
        features = {
            "intel_report_count": len(week_df),
            "intel_critical_count": len(week_df[week_df["severity"] == "critical"]) if len(week_df) > 0 else 0,
            "intel_high_count": len(week_df[week_df["severity"] == "high"]) if len(week_df) > 0 else 0,
            "intel_avg_confidence": week_df["confidence"].mean() if len(week_df) > 0 else 0,
            "intel_total_iocs": week_df["iocs_count"].sum() if len(week_df) > 0 else 0,
        }
        
        # Per-sector threat activity
        for sector in ["education", "government", "healthcare", "finance", "critical_infrastructure"]:
            features[f"intel_{sector}_reports"] = len(week_df[week_df["target_sector"] == sector])
        
        # Threat actor activity
        for actor in ["APT28", "APT29", "Lazarus", "Sandworm"]:
            features[f"intel_{actor.lower()}_activity"] = len(week_df[week_df["threat_actor"] == actor])
        
        return features


if __name__ == "__main__":
    from config import DATA_DIR
    
    aggregator = DataAggregator(DATA_DIR)
    raw_data = aggregator.load_all()
    weekly_data = aggregator.create_weekly_aggregation(raw_data)
    print(f"\nCreated weekly aggregation with {len(weekly_data)} weeks and {len(weekly_data.columns)} features")
    print(weekly_data.head())
