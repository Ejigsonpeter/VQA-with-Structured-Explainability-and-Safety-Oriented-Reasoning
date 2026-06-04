# CSMorgan — ImageCLEFmed MEDVQA-GI 2026

Code and submission pipelines for team **CSMorgan-MEDVQA** at the [ImageCLEFmed MEDVQA-GI 2026](https://www.imageclef.org/) challenge, covering both competition tracks:

- **Task 1 — Visual Question Answering:** generate concise answers to clinical questions about GI endoscopy images.
- **Task 2 — Explainability & Safety:** produce clinician-oriented textual justifications, calibrated confidence, safety notes, and visual grounding (masks + bounding boxes) for each answer.

All work is built on the [Kvasir-VQA-x1](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1) dataset and the SimulaMet baseline adapters.

> **Author:** Ojonugwa Ejiga Peter — Morgan State University, Computer Vision & AI Lab (csmorgan)
> **HF handle:** [`sageofai`](https://huggingface.co/sageofai)

---

## Repository contents

| Notebook | Track | What it does |
|---|---|---|
| `CSMORGAN_FineTune_Submit.ipynb` | Task 1 | One-shot pipeline: continue-train the SimulaMet Qwen2.5-VL adapter via QLoRA, push to HF, validate, and submit. |
| `CSMORGAN_v9_SimulaMetDecoding.ipynb` | Task 1 | Controlled experiment: take the v1 baseline unchanged and patch **only** the decoding parameters to SimulaMet's published settings, then validate. |
| `CSMORGAN_Task2.ipynb` | Task 2 | Full explainability + safety pipeline: answer generation, MedGemma textual explanations, SAM-based visual grounding, and ZIP packaging. |

Each notebook is self-contained and designed to run top-to-bottom in Google Colab (A100 recommended for Task 1 training).

---

## Task 1 — Visual Question Answering

**Goal:** maximize answer quality (primary metric: ROUGE-1) on the Kvasir-VQA-x1 test set.

### Approach
- **Base model:** `Qwen/Qwen2.5-VL-7B-Instruct`, loaded in 4-bit NF4 (bitsandbytes, double quantization, bf16 compute).
- **Starting weights:** `SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft` — the published SimulaMet LoRA adapter, loaded as trainable initial weights rather than from scratch.
- **Fine-tuning:** continued QLoRA training for 1 epoch on a random sample (default 3,000) of Kvasir-VQA-x1, with gradient checkpointing and gradient accumulation.
- **Inference:** the trained adapter is paired with the v1 repo's `normalization.py` and `answer_bank.json` for canonical-answer mapping.

### Pipeline (`CSMORGAN_FineTune_Submit.ipynb`)
1. Install dependencies (`transformers`, `peft`, `bitsandbytes`, `ms-swift`, `medvqa`, …).
2. Hugging Face login (write-scoped token).
3. Mount Google Drive — checkpoints survive Colab disconnects and auto-resume.
4. Continue-train the adapter (~45 min on A100 at 3,000 samples).
5. Push the adapter to `sageofai/Qwen25VL-MEDVQA-GI-S1-subtask1-v2`.
6. Copy companion files (`normalization.py`, `answer_bank.json`, `requirements.txt`) from v1.
7. Push `submission_task1.py` with repo references repointed to v2.
8. `medvqa validate` (dry run — free, no submission slot consumed).
9. Real submission via `medvqa validate_and_submit`, gated behind a `CONFIRM` flag.

```bash
# Validation only (does not consume a submission)
medvqa validate --competition=gi-2026 --task=1 \
  --repo_id=sageofai/Qwen25VL-MEDVQA-GI-S1-subtask1-v2
```

### The v9 decoding experiment (`CSMORGAN_v9_SimulaMetDecoding.ipynb`)
A low-risk ablation testing the hypothesis that the v1 baseline underperformed SimulaMet's reported eval accuracy because of **greedy decoding**. It takes v1 unchanged and patches only the `RequestConfig`:

```python
RequestConfig(max_tokens=512, temperature=0.3, top_k=20,
              top_p=0.7, repetition_penalty=1.05)
```

Everything else (adapter, normalization, answer bank) is identical to v1, so any score change is attributable to decoding alone. The notebook validates, runs a prediction-length distribution diagnostic, and only submits if the result clearly beats the v1 baseline of `rouge1 = 0.5423`.

---

## Task 2 — Explainability & Safety

**Goal:** for each answer, output a structured, clinically grounded explanation with calibrated confidence and optional visual localization.

### Output schema
Each prediction follows a fixed format enforced during fine-tuning:

```
ANSWER:        <concise answer>
JUSTIFICATION: <evidence-based reasoning referencing visible findings>
CONFIDENCE:    <float 0.0–0.85, capped to stay conservative>
SAFETY_NOTE:   <clinical concern, or None>
```

### Components
- **Answer generation:** Qwen2.5-VL-7B QLoRA (same family as Task 1).
- **Textual explanation + safety:** `google/medgemma-4b-it` fine-tuned via QLoRA (`SimulaMet/MedGemma-KvasirVQA-x1-ft`), trained on the structured S2 format above. Confidence is deliberately capped at 0.85, and generic/empty justifications are filtered out.
- **Visual explanation:** `Mayank022/sam-vit-base-kvasir-polyp-segmentation` (SAM fine-tuned on Kvasir-SEG) generates segmentation masks; bounding boxes are derived from the masks. Entries without reliable localization keep `visual_explanation` as an empty list rather than emitting a low-confidence guess.

### Pipeline (`CSMORGAN_Task2.ipynb`)
Run cells in order (`01 → restart → 02 … → 10`):
1. Install dependencies (HPC-safe, falls back to `--user`).
2. Mount Drive and create the output directory tree (`images/`, `jsonl/`, `checkpoints/`, `results/`, `logs/`, `plots/`).
3. Imports, seeding, model registry.
4. Load images (`SimulaMet-HOST/Kvasir-VQA`) and QA pairs (`SimulaMet/Kvasir-VQA-x1`).
5. Fine-tune MedGemma-4B (QLoRA, S2 safety format).
6. Run answer + explanation inference → `s2_results.json`, `submission_task2.jsonl`.
7. Materialize any missing images, then generate SAM masks/overlays/bboxes.
8. Package everything into the submission ZIP.

### Submission
Task 2 is submitted **by email**, not via the `medvqa` CLI:

- **To:** `steven@simula.no`
- **Subject:** `ImageCLEFmed-MEDVQA-GI-2026 Task2 Submission - CSMorgan-MEDVQA`
- **Attachment:** `CSMorgan_2_task2.zip` (JSONL + visuals + `submission_task2.py` metadata)

---

## Setup

These notebooks target **Google Colab** with a GPU runtime. For Task 1 training, switch to an **A100** runtime (Runtime → Change runtime type → A100).

Requirements:
- A Hugging Face account with a **write-scoped** token (for pushing adapters and submitting).
- A Google account (Drive is used for checkpoint persistence).
- For Task 1 submission, the `medvqa` CLI configured for the `gi-2026` competition.

Key dependencies (installed by each notebook's first cell):

```
transformers>=4.45.0   accelerate>=0.34.0   peft>=0.13.0
bitsandbytes>=0.43.0   ms-swift==3.8.0      qwen_vl_utils==0.0.11
datasets               evaluate             medvqa
nltk  rouge_score  sacrebleu  Pillow  pandas  matplotlib  huggingface_hub
```

---

## Models & artifacts

| Repo | Role |
|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | Base VLM |
| `SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft` | SimulaMet baseline adapter (starting weights) |
| `sageofai/Qwen25VL-MEDVQA-GI-S1-subtask1` | v1 baseline (ROUGE-1 = 0.5423) |
| `sageofai/Qwen25VL-MEDVQA-GI-S1-subtask1-v2` | Continued QLoRA adapter |
| `sageofai/Qwen25VL-MEDVQA-GI-S1-subtask1-v9` | Decoding-parameter ablation |
| `google/medgemma-4b-it` + `SimulaMet/MedGemma-KvasirVQA-x1-ft` | Task 2 explanation/safety |
| `Mayank022/sam-vit-base-kvasir-polyp-segmentation` | Task 2 visual grounding |

---

## Datasets

- [`SimulaMet/Kvasir-VQA-x1`](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1) — QA pairs (train/test).
- [`SimulaMet-HOST/Kvasir-VQA`](https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA) — source endoscopy images.
- `SimulaMet/Kvasir-VQA-test` — validation split used for length diagnostics.

Please follow the licensing and usage terms of these datasets and the challenge.

---

## Reproducibility notes

- A fixed seed (`42`) is used for sampling and training.
- Task 1 training auto-resumes from the last Drive checkpoint after a disconnect.
- The v9 experiment isolates a single variable (decoding) to keep results interpretable.
- Task 2 confidence is capped and visual explanations are only emitted when grounding is reliable — both deliberate conservative-by-default design choices for a clinical setting.

---

## Acknowledgements

Built on the SimulaMet Kvasir-VQA baselines and the open-source Qwen2.5-VL, MedGemma, and SAM model families. Thanks to the ImageCLEFmed MEDVQA-GI 2026 organizers.

