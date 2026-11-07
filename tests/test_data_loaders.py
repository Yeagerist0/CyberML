import json

from src.data_loaders import DataAggregator, HoneypotLoader


def test_honeypot_jsonl_is_read_as_json_lines(tmp_path):
    path = tmp_path / "honeypot.jsonl"
    path.write_text(
        "\n".join(json.dumps({"timestamp": "2026-11-01", "source_ip": ip}) for ip in ("192.0.2.1", "192.0.2.2")),
        encoding="utf-8",
    )

    records = HoneypotLoader(path).load()

    assert records["source_ip"].tolist() == ["192.0.2.1", "192.0.2.2"]


def test_aggregator_prefers_real_jsonl_over_empty_placeholder(tmp_path):
    (tmp_path / "honeypot").touch()
    (tmp_path / "honeypot.jsonl").write_text('{"source_ip":"192.0.2.3"}\n', encoding="utf-8")

    records = DataAggregator(tmp_path).loaders["honeypot"].load()

    assert records["source_ip"].tolist() == ["192.0.2.3"]


def test_directory_placeholder_uses_sample_data(tmp_path):
    (tmp_path / "github_poc").mkdir()

    records = DataAggregator(tmp_path).loaders["github_poc"].load()

    assert not records.empty
