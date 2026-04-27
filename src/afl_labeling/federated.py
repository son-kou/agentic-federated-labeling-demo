"""FederatedCoordinatorAgent — simulate FL rounds without moving image data.

Each site contributes only:
  - a model_delta_summary (string describing weight diff)
  - a label_quality_summary (aggregated QC stats)
  - counts of processed/review-needed cases

Image data NEVER leave the site. The coordinator sees only these summaries.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from .schemas import FLRoundSummary, SiteContribution


def _load_qc_records(qc_dir: Path, site_id: str) -> List[Dict[str, Any]]:
    site_qc_dir = qc_dir / site_id
    if not site_qc_dir.exists():
        return []
    records = []
    for f in sorted(site_qc_dir.glob("*_qc.json")):
        records.append(json.loads(f.read_text()))
    return records


def _simulate_model_delta(site_id: str, round_number: int, rng: random.Random) -> str:
    """Describe a simulated gradient update (no actual weights in the demo)."""
    norm = round(rng.uniform(0.01, 0.08) * (1.0 / round_number), 5)
    return f"∥Δw∥={norm:.5f}  cosine_sim_to_global=+{rng.uniform(0.88, 0.98):.3f}"


def _site_contribution(
    site_id: str,
    qc_records: List[Dict[str, Any]],
    round_number: int,
    rng: random.Random,
) -> SiteContribution:
    total = len(qc_records)
    if total == 0:
        return SiteContribution(
            site_id=site_id,
            model_delta_summary="no data",
            label_quality_summary="no data",
            cases_processed=0,
            cases_needing_review=0,
            mean_confidence=0.0,
        )

    review_needed = sum(1 for r in qc_records if r["status"] != "auto_qc_pass")
    # Confidence isn't stored in QC, derive from Dice proxy
    dice_vals = [r["dice_vs_gt"] for r in qc_records if r.get("dice_vs_gt") is not None]
    mean_conf = float(sum(dice_vals) / len(dice_vals)) if dice_vals else round(rng.uniform(0.65, 0.88), 3)

    # Simulate quality improving across rounds
    round_boost = (round_number - 1) * 0.02
    mean_conf = min(1.0, mean_conf + round_boost)

    qc_pass_rate = sum(1 for r in qc_records if r["status"] == "auto_qc_pass") / total
    quality_summary = (
        f"pass_rate={qc_pass_rate:.2f}  review_rate={(review_needed/total):.2f}"
        f"  mean_dice={mean_conf:.3f}  flags_seen={_collect_flags(qc_records)}"
    )

    return SiteContribution(
        site_id=site_id,
        model_delta_summary=_simulate_model_delta(site_id, round_number, rng),
        label_quality_summary=quality_summary,
        cases_processed=total,
        cases_needing_review=review_needed,
        mean_confidence=round(mean_conf, 4),
    )


def _collect_flags(records: List[Dict[str, Any]]) -> str:
    from collections import Counter
    cnt: Counter = Counter()
    for r in records:
        for flag in r.get("risk_flags", []):
            cnt[flag] += 1
    if not cnt:
        return "none"
    return ",".join(f"{k}×{v}" for k, v in cnt.most_common(3))


class FederatedCoordinator:
    """Orchestrate simulated FL rounds across all sites."""

    def __init__(
        self,
        data_root: Path,
        outputs_root: Path,
        config: Dict[str, Any],
    ) -> None:
        self.data_root = data_root
        self.outputs_root = outputs_root
        self.config = config
        self.sites: List[str] = config.get("sites", ["site_A", "site_B", "site_C"])
        self.qc_dir = outputs_root / "qc_reports"
        self.fl_dir = outputs_root / "fl_rounds"
        self.fl_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(config.get("random_seed", 42))

    def run_rounds(self, num_rounds: int = 3) -> List[FLRoundSummary]:
        summaries: List[FLRoundSummary] = []
        for rnd in range(1, num_rounds + 1):
            summary = self._run_one_round(rnd)
            summaries.append(summary)
            self._save_round(summary)
            print(
                f"  [FL round {rnd}] "
                f"global_conf={summary.global_mean_confidence:.3f}  "
                f"review_rate={summary.global_review_rate:.2f}"
            )
        return summaries

    def _run_one_round(self, rnd: int) -> FLRoundSummary:
        contributions: List[SiteContribution] = []
        for site_id in self.sites:
            qc_records = _load_qc_records(self.qc_dir, site_id)
            contrib = _site_contribution(site_id, qc_records, rnd, self._rng)
            contributions.append(contrib)

        confs = [c.mean_confidence for c in contributions if c.cases_processed > 0]
        global_conf = float(sum(confs) / len(confs)) if confs else 0.0

        total_cases = sum(c.cases_processed for c in contributions)
        total_review = sum(c.cases_needing_review for c in contributions)
        review_rate = total_review / total_cases if total_cases > 0 else 0.0

        return FLRoundSummary(
            round_number=rnd,
            site_contributions=contributions,
            global_mean_confidence=round(global_conf, 4),
            global_review_rate=round(review_rate, 4),
            notes=(
                f"Round {rnd}: global model updated from {len(contributions)} site(s). "
                "No image data left any site. Only gradient summaries and QC stats shared."
            ),
        )

    def _save_round(self, summary: FLRoundSummary) -> None:
        path = self.fl_dir / f"round_{summary.round_number:03d}_summary.json"
        path.write_text(summary.model_dump_json(indent=2))
