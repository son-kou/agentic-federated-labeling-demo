import argparse
from pathlib import Path
from src.afl_labeling.synthetic_data import generate_sites
from src.afl_labeling.agents import run_site_pipeline
from src.afl_labeling.federated import FederatedCoordinator
from src.afl_labeling.report import generate_meeting_report
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--run-pipeline", action="store_true")
    p.add_argument("--run-fl", action="store_true")
    p.add_argument("--config", default="configs/demo_config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "sites"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_data:
        print("Generating synthetic data...")
        generate_sites(data_dir, cfg)

    if args.run_pipeline:
        print("Running per-site pipeline (autolabel + qc + active learning)...")
        for site in cfg["sites"]:
            site_dir = data_dir / site
            run_site_pipeline(site_dir, out_dir)

    if args.run_fl:
        print("Simulating federated rounds...")
        coordinator = FederatedCoordinator(
            data_root=data_dir, outputs_root=out_dir, config=cfg
        )
        coordinator.run_rounds(cfg.get("fl_rounds", 3))

    print("Generating meeting report...")
    generate_meeting_report(data_root=data_dir, outputs_root=out_dir, config=cfg)


if __name__ == "__main__":
    main()
