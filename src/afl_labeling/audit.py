"""Audit trail: record what information left each site."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


_AUDIT_VERSION = "1.0"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_site_audit(
    site_id: str,
    cases_processed: int,
    labels_exported: int,
    qc_flags_seen: List[str],
    data_leaving_site: List[str],
    data_staying_local: List[str],
) -> Dict[str, Any]:
    return {
        "audit_version": _AUDIT_VERSION,
        "site_id": site_id,
        "timestamp": _timestamp(),
        "cases_processed": cases_processed,
        "labels_exported_count": labels_exported,
        "qc_flags_seen": qc_flags_seen,
        "information_that_LEFT_site": data_leaving_site,
        "information_that_STAYED_LOCAL": data_staying_local,
        "privacy_statement": (
            "Raw image volumes, patient metadata, and pixel-level labels "
            "NEVER left this site. Only model weight deltas (as scalar norms) "
            "and aggregated QC statistics were transmitted to the coordinator."
        ),
    }


def write_site_audit(audit_dir: Path, site_id: str, record: Dict[str, Any]) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{site_id}_audit.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def build_fl_audit(
    fl_rounds: int,
    sites: List[str],
    round_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "audit_version": _AUDIT_VERSION,
        "timestamp": _timestamp(),
        "fl_rounds_completed": fl_rounds,
        "participating_sites": sites,
        "aggregation_method": "FedAvg (simulated — gradient norm summaries only)",
        "data_shared_with_coordinator": [
            "model_delta_norm (scalar)",
            "cosine_similarity_to_global (scalar)",
            "label_quality_summary (aggregated per-site stats)",
            "cases_processed_count (integer)",
            "cases_needing_review_count (integer)",
        ],
        "data_NOT_shared": [
            "raw MRI volumes",
            "individual case metadata",
            "per-voxel labels",
            "patient identifiers",
        ],
        "round_summaries": round_summaries,
        "privacy_guarantee": (
            "This simulation demonstrates the FL communication protocol. "
            "In production, differential privacy noise (e.g., Gaussian mechanism) "
            "should be applied to gradient updates before transmission."
        ),
    }
