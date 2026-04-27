# Model Replacement Plan

**How to replace the mock AutoLabelingAgent with a real segmentation model**

The current `AutoLabelingAgent` in `src/afl_labeling/autolabel.py` uses a heuristic
that degrades the synthetic ground-truth mask to simulate imperfect auto-labels. This
is sufficient to demonstrate the infrastructure workflow. When connecting to real data,
replace the internals of `AutoLabelingAgent.label_case()` with one of the options below.

---

## Expected Interface (unchanged regardless of model)

```python
def label_case(
    case_id: str,
    site_id: str,
    volume: np.ndarray,          # float32, shape (D, H, W), normalised to [0, 1]
    gt_prostate: np.ndarray,     # uint8 binary mask — for demo validation only
    gt_lesion: np.ndarray,       # uint8 binary mask — for demo validation only
) -> tuple[LabelRecord, LabelRecord | None]:
    ...
```

The `LabelRecord` always carries a `LabelProvenance`:

```python
LabelProvenance(
    model_name    = "nnUNet_prostate_v2",
    model_version = "2.0.0",
    generated_at  = datetime.now(timezone.utc),
    input_case_id = case_id,
    label_type    = "prostate_gland",
    confidence_score = 0.87,   # see uncertainty section below
)
```

In a real model, the provenance is populated from the model registry, not hardcoded.
The `gt_prostate` and `gt_lesion` arguments are dropped (not available in production).

---

## Option 1 — MONAI Label

**Best for:** interactive labeling sessions; radiologist corrections in OHIF / 3D Slicer.

```python
from monailabel.client import MONAILabelClient

client = MONAILabelClient(server_url="http://localhost:8000")
result = client.infer(model="prostate_segmentation", image_id=case_id)
pred_mask = result["label"]           # numpy array
confidence = result["params"]["score"]
```

- Run MONAI Label server locally on the site machine.
- Model weights stay local; only the client call crosses the process boundary.
- Supports active learning: `client.next_sample()` returns the highest-priority unlabeled case.
- Uncertainty proxy: use softmax entropy from the segmentation head.

**Swap-in location:** `autolabel.py`, replace `_degrade_gt_mask()` with `client.infer()`.

---

## Option 2 — nnU-Net v2

**Best for:** batch auto-labeling of a full site archive; no interactive component needed.

```bash
nnUNetv2_predict \
  -i INPUT_FOLDER -o OUTPUT_FOLDER \
  -d DATASET_ID -c 3d_fullres \
  --save_probabilities
```

In Python:
```python
import subprocess, numpy as np, nibabel as nib

def run_nnunet(nifti_path: Path, output_dir: Path) -> tuple[np.ndarray, float]:
    subprocess.run(["nnUNetv2_predict", "-i", str(nifti_path.parent),
                    "-o", str(output_dir), "-d", "Dataset101_Prostate",
                    "-c", "3d_fullres", "--save_probabilities"], check=True)
    seg = nib.load(output_dir / nifti_path.name).get_fdata().astype(np.uint8)
    prob = nib.load(output_dir / nifti_path.stem + "_prob.nii.gz").get_fdata()
    confidence = float(prob[seg > 0].mean()) if seg.sum() > 0 else 0.5
    return seg, confidence
```

- Pretrained on PI-CAI / Prostate158; fine-tune on local data after Phase 5.
- `--save_probabilities` gives per-voxel softmax output → use entropy as uncertainty proxy.
- GPU recommended; CPU inference is ~5–10 min per case on a 96×96×64 volume.

**Swap-in location:** wrap `run_nnunet()` inside `AutoLabelingAgent.label_case()`.

---

## Option 3 — MedSAM or MedSAM2

**Best for:** interactive or semi-automatic labeling from bounding-box prompts; fast inference.

```python
from segment_anything import SamPredictor

predictor = SamPredictor(medsam_model)
predictor.set_image(slice_rgb)           # 2D slice for slice-by-slice mode
masks, scores, _ = predictor.predict(
    box=np.array([x0, y0, x1, y1]),      # bounding box prompt
    multimask_output=False,
)
```

For 3D, run slice-by-slice and stack:
```python
pred_3d = np.stack([predict_slice(volume[z]) for z in range(volume.shape[0])])
confidence = float(scores.mean())
```

- MedSAM2 supports video (slice sequence) prompting — more coherent 3D output.
- Confidence = mean IoU score returned by SAM head.
- No fine-tuning required for gland; lesion detection benefits from few-shot examples.

**Swap-in location:** replace `_degrade_gt_mask()` with the SAM inference loop.

---

## Option 4 — Prostate MRI Lesion Detector

For lesion-candidate detection specifically, use a detection model rather than a segmentation model:

```python
class LesionDetector:
    def detect(self, volume: np.ndarray, prostate_mask: np.ndarray
               ) -> list[dict]:
        # returns list of {"bbox": ..., "score": ..., "mask": ...}
        ...
```

Candidates: PI-CAI baseline detection models, nnDetection, or a nnU-Net trained on csPCa labels.
Confidence = detection score (0–1). Uncertainty proxy = 1 − confidence.

---

## Option 5 — Site-Specific Fine-Tuned Model

After Phase 5 (human correction), fine-tune a global model on local corrected labels:

```python
# Fine-tune nnU-Net on corrected labels from this site
subprocess.run([
    "nnUNetv2_train", "Dataset101_Prostate", "3d_fullres", "0",
    "--pretrained_weights", str(global_model_path),
    "--c",                        # continue from checkpoint
], check=True)
```

- Store model checkpoint locally; only the gradient summary leaves the site.
- Version and register the checkpoint in the provenance JSON.
- Compare per-site Dice before and after fine-tuning to confirm benefit.

---

## Uncertainty Proxy — Summary

| Model | Uncertainty proxy |
|-------|------------------|
| MONAI Label | Softmax entropy on segmentation head |
| nnU-Net | Mean softmax probability on predicted voxels; or test-time augmentation variance |
| MedSAM | 1 − mean IoU score from SAM head |
| Lesion detector | 1 − detection score |
| Mock (current demo) | 1 − Dice vs synthetic GT (demo only) |

The `QualityControlAgent` uses `boundary_uncertainty` as a shape-based proxy independent
of the model; this remains valid regardless of which model is used.

---

## Checklist for Swapping the Model

- [ ] Model weights are stored locally (not downloaded at inference time)
- [ ] Model version is recorded in `LabelProvenance.model_version`
- [ ] Confidence score is populated from a real probability estimate
- [ ] `gt_prostate` / `gt_lesion` arguments are removed from the call signature
- [ ] NIfTI or DICOM input loader replaces the NumPy `.npz` loader in `preprocessing.py`
- [ ] Per-case inference time is logged (to estimate radiologist waiting time)
- [ ] QC thresholds are validated against a small radiologist-labelled reference set
