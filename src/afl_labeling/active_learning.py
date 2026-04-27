"""ActiveLearningAgent — rank cases by review priority."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .schemas import ActiveLearningRecord, QCRecord

# Weights for priority scoring
_W_BOUNDARY_UNC = 0.30
_W_SITE_SHIFT = 0.25
_W_CONFIDENCE_INV = 0.25  # (1 - confidence)
_W_STATUS = 0.20

_STATUS_WEIGHT = {
    "reject_or_redraw": 1.0,
    "review_required": 0.6,
    "auto_qc_pass": 0.0,
}

_FLAG_REASONS = {
    "high_boundary_uncertainty": "Boundary uncertainty above threshold — likely needs boundary correction.",
    "small_prostate_volume": "Prostate volume implausibly small — possible under-segmentation.",
    "fragmented_mask": "Multiple disconnected components — likely spurious segmentation.",
    "lesion_outside_prostate": "Lesion candidate detected outside gland — anatomically implausible.",
    "high_site_shift_risk": "Case from a high-shift site — model accuracy may be degraded.",
}


class ActiveLearningAgent:
    """Prioritise cases for human review using QC outputs and confidence."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def rank_cases(
        self,
        qc_records: List[QCRecord],
        confidence_map: dict[str, float],
    ) -> List[ActiveLearningRecord]:
        """
        *confidence_map* maps case_id → prostate confidence score.
        Returns records sorted by descending priority.
        """
        scored: List[tuple[float, QCRecord]] = []
        for rec in qc_records:
            conf = confidence_map.get(rec.case_id, 0.5)
            status_w = _STATUS_WEIGHT.get(rec.status, 0.0)

            score = (
                _W_BOUNDARY_UNC * rec.boundary_uncertainty
                + _W_SITE_SHIFT * rec.site_shift_risk
                + _W_CONFIDENCE_INV * (1.0 - conf)
                + _W_STATUS * status_w
            )
            # Extra bump for multiple flags
            score += 0.05 * max(0, len(rec.risk_flags) - 1)
            scored.append((float(score), rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[ActiveLearningRecord] = []
        for rank, (score, rec) in enumerate(scored, start=1):
            reasons = [_FLAG_REASONS[f] for f in rec.risk_flags if f in _FLAG_REASONS]
            if rec.status != "auto_qc_pass" and not reasons:
                reasons = [f"QC status: {rec.status}"]
            results.append(
                ActiveLearningRecord(
                    case_id=rec.case_id,
                    site_id=rec.site_id,
                    priority_score=round(score, 4),
                    rank=rank,
                    reasons=reasons,
                )
            )

        self._save(results)
        return results

    def _save(self, records: List[ActiveLearningRecord]) -> None:
        data = [r.model_dump() for r in records]
        (self.out_dir / "active_learning_queue.json").write_text(
            json.dumps(data, indent=2)
        )
