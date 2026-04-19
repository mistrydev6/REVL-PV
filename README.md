# REVL-PV: Reasoning-Boosted Vision-Language for Photovoltaics

A domain-aware, reasoning-based vision-language framework for photovoltaic defect diagnosis. Rather than treating defect classification as direct pattern recognition, REVL-PV integrates photovoltaic-specific domain knowledge with structured diagnostic reasoning, mirroring the Evidence→Cause→Action logic employed by human inspectors.

## Pipeline

![Pipeline](assets/model_architecture.pdf)

---

## Setup

```bash
git clone --recurse-submodules https://github.com/mistrydev6/REVL-PV.git
cd REVL-PV
pip install opencv-python numpy flask ray openrlhf vllm torch transformers
```

---

## Stage 1 — Class Balanced Multi-modal Data Curation

Set these variables at the top of the script:

```python
base_folder = "YOUR BASE FOLDER"
classes_to_augment = {"good": 50, "bad": 50}  # class name: current count
target_count = 1500
```

```bash
python Stage_1_Data_augmetation.py
```

---

## Stage 2 — Reasoning-Boosted Supervised Fine-Tuning (RSFT)

Full fine-tuning of Qwen2.5-VL using LlamaFactory with DeepSpeed ZeRO-3 and the `qwen2_vl` chat template. Teaches the model structured chain-of-thought reasoning over solar panel images.

Set these in `Stage_2_RSFT.yaml`:

```yaml
model_name_or_path: YOUR MODEL
dataset: YOUR DATASET
output_dir: YOUR OUTPUT DIR
```

Run from inside the submodule so LlamaFactory can resolve its own paths:

```bash
cd LlamaFactory
llamafactory-cli train ../Stage_2_RSFT.yaml
```

---

## Stage 3 — Two Phase Reasoning Enhancement (2PRE)

Two-phase PPO training using OpenRLHF via Ray. The solar verifier reward model runs as a Flask service and scores responses based on classification correctness, reasoning step coverage, and probability formatting.

**Start the reward server first (required for both phases):**

```bash
python solar_verifier_reward_model.py
```

### Phase 1

Set these variables in the script:

```bash
export DATASET_PATH="..."
export PRETRAIN_MODEL_PATH="..."   # your Stage 2 checkpoint
export MODEL_NAME="..."
export WANDB_API_KEY="..."
```

```bash
bash Stage_3_Phase_1.sh
```

### Phase 2

Continues from the Phase 1 checkpoint. Set your desired variables in the script:

```bash
export PRETRAIN_MODEL_PATH="..."   # your Phase 1 checkpoint
```

```bash
bash Stage_3_Phase_2.sh
```

---

## Stage 4 — Robust Inference

Runs 6-view test-time augmentation (full image + 5 crops) on a single solar panel image using the trained Qwen2.5-VL model. Final label is determined by majority voting across views.

Set at the top of the script:

```python
MODEL_PATH = "YOUR MODEL HERE"
TEST_IMAGE_PATH = "YOUR IMAGE PATH HERE"
```

```bash
python Stage_4_TTA_inference.py
```

---
