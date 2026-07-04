# Deprecated runner scripts

These are superseded, redundant implementations of the 16-phase curriculum
runner. They are kept for reference only — **do not run them**.

- `kaggle_run_phases.py` — the script that actually produced the (fake)
  `curriculum-phaseNN-*` checkpoints on the HF Hub in April 2026. Its
  `_build_sft_cmd()` never passed the required `--train_config` flag to
  `train_sft.py`, so every training subprocess failed instantly at argument
  parsing. The failure wasn't detected: the script fell back to re-uploading
  the unchanged input checkpoint under each new phase's tag and marked every
  phase "complete" anyway. All 16 phases plus the DPO pass were faked this
  way — verified by SHA-256: every `curriculum-phaseNN` checkpoint on the Hub
  is byte-identical to `patch-checkpoint-00000300`, the pre-curriculum base.

- `colab_run_phases.py` — an earlier, parallel implementation of the same
  16-phase curriculum idea for Google Colab instead of Kaggle, paired with
  `configs/training/_deprecated/milestones_v3.yaml`. Superseded by the
  Kaggle-based system before it saw real use; not audited for correctness.

**Current, active runner:** `scripts/kaggle_run_curriculum.py`, driven by
`configs/training/milestones_v2.yaml` and invoked from
`notebooks/kaggle_curriculum.ipynb`. It has been fixed to:
1. Verify each phase actually changed the model weights (SHA-256 comparison
   against the input checkpoint) before pushing to the Hub or marking the
   phase done — a failed/no-op phase now aborts the session instead of
   silently faking success.
2. Route `phase_type: dpo` (phase 16) to `train_dpo.py` instead of
   `train_sft.py` — the DPO data format has no assistant turn in `messages`,
   which the SFT instruction-loss path can't train on at all.
