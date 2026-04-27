"""Pydantic data models shared across the pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class CaseMetadata(BaseModel):
    case_id: str
    site_id: str
    spacing: List[float]
    modalities: List[str]
    has_lesion: bool
    synthetic_shift_type: str


class LabelProvenance(BaseModel):
    model_name: str
    model_version: str
    generated_at: datetime
    input_case_id: str
    label_type: str  # "prostate_gland" | "lesion_candidate"
    confidence_score: float = Field(ge=0.0, le=1.0)


class LabelRecord(BaseModel):
    case_id: str
    site_id: str
    provenance: LabelProvenance
    label_path: str  # relative path to .npz mask


class QCRecord(BaseModel):
    case_id: str
    site_id: str
    prostate_volume_ml: float
    num_connected_components: int
    boundary_uncertainty: float
    lesion_inside_prostate: Optional[bool]
    volume_plausible: bool
    site_shift_risk: float
    status: str  # "auto_qc_pass" | "review_required" | "reject_or_redraw"
    risk_flags: List[str]
    dice_vs_gt: Optional[float]


class ActiveLearningRecord(BaseModel):
    case_id: str
    site_id: str
    priority_score: float
    rank: int
    reasons: List[str]


class SiteQCSummary(BaseModel):
    site_id: str
    total_cases: int
    qc_pass: int
    review_required: int
    rejected: int
    mean_confidence: float
    mean_dice: Optional[float]

    @property
    def qc_pass_rate(self) -> float:
        return self.qc_pass / self.total_cases if self.total_cases else 0.0

    @property
    def review_rate(self) -> float:
        return self.review_required / self.total_cases if self.total_cases else 0.0


class FLRoundSummary(BaseModel):
    round_number: int
    site_contributions: List[SiteContribution]
    global_mean_confidence: float
    global_review_rate: float
    notes: str


class SiteContribution(BaseModel):
    site_id: str
    model_delta_summary: str
    label_quality_summary: str
    cases_processed: int
    cases_needing_review: int
    mean_confidence: float
