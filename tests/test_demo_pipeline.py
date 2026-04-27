"""Integration tests for the demo pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("demo_root")


@pytest.fixture(scope="module")
def demo_config():
    return {
        "sites": ["site_A", "site_B"],
        "cases_per_site": 3,
        "volume_shape": [32, 32, 24],
        "random_seed": 7,
        "fl_rounds": 2,
    }


@pytest.fixture(scope="module")
def data_dir(tmp_root, demo_config):
    from src.afl_labeling.synthetic_data import generate_sites
    d = tmp_root / "data" / "sites"
    generate_sites(d, demo_config)
    return d


@pytest.fixture(scope="module")
def outputs_dir(tmp_root, demo_config, data_dir):
    from src.afl_labeling.agents import run_site_pipeline
    out = tmp_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for site in demo_config["sites"]:
        run_site_pipeline(data_dir / site, out)
    return out


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_site_dirs_created(self, data_dir, demo_config):
        for site in demo_config["sites"]:
            assert (data_dir / site).exists(), f"Site dir missing: {site}"

    def test_correct_case_count(self, data_dir, demo_config):
        for site in demo_config["sites"]:
            cases = list((data_dir / site).iterdir())
            assert len(cases) == demo_config["cases_per_site"], (
                f"{site}: expected {demo_config['cases_per_site']} cases, got {len(cases)}"
            )

    def test_volume_shape(self, data_dir, demo_config):
        site = demo_config["sites"][0]
        case_dir = sorted((data_dir / site).iterdir())[0]
        data = np.load(case_dir / "volume.npz")
        expected = tuple(demo_config["volume_shape"])
        assert data["volume"].shape == expected, (
            f"Expected {expected}, got {data['volume'].shape}"
        )

    def test_metadata_fields(self, data_dir, demo_config):
        site = demo_config["sites"][0]
        case_dir = sorted((data_dir / site).iterdir())[0]
        meta = json.loads((case_dir / "metadata.json").read_text())
        for field in ["case_id", "site_id", "spacing", "modalities", "has_lesion", "synthetic_shift_type"]:
            assert field in meta, f"Missing metadata field: {field}"

    def test_masks_binary(self, data_dir, demo_config):
        site = demo_config["sites"][0]
        case_dir = sorted((data_dir / site).iterdir())[0]
        data = np.load(case_dir / "volume.npz")
        assert set(np.unique(data["gt_prostate"])).issubset({0, 1})
        assert set(np.unique(data["gt_lesion"])).issubset({0, 1})


class TestAutoLabels:
    def test_prostate_label_exists(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        label_dir = outputs_dir / "labels" / site
        assert label_dir.exists(), "Labels dir not created"
        case_dirs = list(label_dir.iterdir())
        assert len(case_dirs) > 0, "No case label dirs found"
        first = sorted(case_dirs)[0]
        assert (first / "prostate_pred.npz").exists(), "prostate_pred.npz missing"

    def test_provenance_fields(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        label_dir = outputs_dir / "labels" / site
        first = sorted(label_dir.iterdir())[0]
        prov = json.loads((first / "prostate_provenance.json").read_text())
        for field in ["model_name", "model_version", "generated_at", "input_case_id", "label_type", "confidence_score"]:
            assert field in prov, f"Provenance missing: {field}"

    def test_confidence_in_range(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        label_dir = outputs_dir / "labels" / site
        first = sorted(label_dir.iterdir())[0]
        prov = json.loads((first / "prostate_provenance.json").read_text())
        assert 0.0 <= prov["confidence_score"] <= 1.0


class TestQCReports:
    def test_qc_files_created(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        qc_dir = outputs_dir / "qc_reports" / site
        assert qc_dir.exists(), "QC dir not created"
        assert len(list(qc_dir.glob("*_qc.json"))) == demo_config["cases_per_site"]

    def test_qc_status_valid(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        qc_dir = outputs_dir / "qc_reports" / site
        valid_statuses = {"auto_qc_pass", "review_required", "reject_or_redraw"}
        for f in qc_dir.glob("*_qc.json"):
            rec = json.loads(f.read_text())
            assert rec["status"] in valid_statuses, f"Invalid status: {rec['status']}"

    def test_qc_fields_present(self, outputs_dir, demo_config):
        site = demo_config["sites"][0]
        qc_dir = outputs_dir / "qc_reports" / site
        required = ["case_id", "site_id", "prostate_volume_ml", "boundary_uncertainty",
                    "status", "risk_flags", "dice_vs_gt"]
        for f in qc_dir.glob("*_qc.json"):
            rec = json.loads(f.read_text())
            for field in required:
                assert field in rec, f"QC missing field: {field}"


class TestFederatedLearning:
    @pytest.fixture(scope="class")
    def fl_outputs(self, tmp_root, demo_config, data_dir, outputs_dir):
        from src.afl_labeling.federated import FederatedCoordinator
        coordinator = FederatedCoordinator(
            data_root=data_dir, outputs_root=outputs_dir, config=demo_config
        )
        coordinator.run_rounds(demo_config["fl_rounds"])
        return outputs_dir / "fl_rounds"

    def test_round_files_created(self, fl_outputs, demo_config):
        files = sorted(fl_outputs.glob("round_*_summary.json"))
        assert len(files) == demo_config["fl_rounds"], (
            f"Expected {demo_config['fl_rounds']} round files, got {len(files)}"
        )

    def test_round_structure(self, fl_outputs):
        for f in sorted(fl_outputs.glob("round_*_summary.json")):
            rnd = json.loads(f.read_text())
            assert "round_number" in rnd
            assert "site_contributions" in rnd
            assert "global_mean_confidence" in rnd
            assert "global_review_rate" in rnd
            assert "notes" in rnd

    def test_no_image_data_in_summary(self, fl_outputs):
        """Verify round summaries contain no raw image arrays or pixel data."""
        for f in sorted(fl_outputs.glob("round_*_summary.json")):
            content = f.read_text()
            data = json.loads(content)
            # Must not contain raw arrays or pixel-level data
            assert "pixel_data" not in content
            assert "voxel_array" not in content
            assert "npz" not in content
            # site_contributions must only carry scalar/string fields
            for contrib in data.get("site_contributions", []):
                assert isinstance(contrib["cases_processed"], int)
                assert isinstance(contrib["cases_needing_review"], int)
                assert isinstance(contrib["mean_confidence"], float)
                assert isinstance(contrib["model_delta_summary"], str)
                assert isinstance(contrib["label_quality_summary"], str)


class TestMeetingReport:
    @pytest.fixture(scope="class")
    def report_path(self, tmp_root, demo_config, data_dir, outputs_dir):
        from src.afl_labeling.report import generate_meeting_report
        return generate_meeting_report(
            data_root=data_dir, outputs_root=outputs_dir, config=demo_config
        )

    def test_report_created(self, report_path):
        assert report_path.exists(), "meeting_report.md not created"

    def test_report_not_empty(self, report_path):
        content = report_path.read_text()
        assert len(content) > 500, "Report suspiciously short"

    def test_report_contains_key_sections(self, report_path):
        content = report_path.read_text()
        for section in ["Research Question", "Per-Site", "Auto-Label", "Federated", "Privacy"]:
            assert section in content, f"Missing section: {section}"


class TestDocFiles:
    """Verify all collaborator-facing documentation files are present and non-empty."""

    DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"

    EXPECTED = [
        "demo_script.md",
        "collaborator_one_pager.md",
        "real_deployment_plan.md",
        "model_replacement_plan.md",
        "privacy_boundary.md",
    ]

    def test_docs_directory_exists(self):
        assert self.DOCS_ROOT.exists(), "docs/ directory not found"

    @pytest.mark.parametrize("filename", EXPECTED)
    def test_doc_file_exists(self, filename):
        path = self.DOCS_ROOT / filename
        assert path.exists(), f"docs/{filename} is missing"

    @pytest.mark.parametrize("filename", EXPECTED)
    def test_doc_file_not_empty(self, filename):
        path = self.DOCS_ROOT / filename
        content = path.read_text()
        assert len(content) > 200, f"docs/{filename} is suspiciously short ({len(content)} chars)"

    def test_demo_script_has_sections(self):
        content = (self.DOCS_ROOT / "demo_script.md").read_text()
        for section in ["Opening", "problem", "privacy", "next steps"]:
            assert section.lower() in content.lower(), f"demo_script.md missing section: {section}"

    def test_privacy_boundary_has_tables(self):
        content = (self.DOCS_ROOT / "privacy_boundary.md").read_text()
        assert "Never" in content, "privacy_boundary.md missing 'Never' entries"
        assert "shareable" in content.lower(), "privacy_boundary.md missing shareable entries"

    def test_deployment_plan_has_phases(self):
        content = (self.DOCS_ROOT / "real_deployment_plan.md").read_text()
        for phase in ["Phase 1", "Phase 2", "Phase 6", "Phase 7"]:
            assert phase in content, f"real_deployment_plan.md missing {phase}"
