# AGENTS.md

## Purpose
This file provides a **navigation map for agents and automation tools** working on this repository. It does not define coding conventions, testing frameworks, or rigid engineering practices. Instead, it points to the key entry points and detailed documentation resources.

For extended explanations, see:
- [README.md](README.md)
- [README_MORE.md](README_MORE.md)

---

## Main Entry Points
- **Specialization (self-supervised pretraining)** → [`specialization.py`](specialization.py)  
  Run: `python specialization.py [OPTIONS...]`  
  Documentation: [specialization.md](specialization.md)

- **Classification fine-tuning (cyclone detection)** → [`classification.py`](classification.py)  
  Run: `python classification.py [OPTIONS...]`  
  Documentation: [classification.md](classification.md)

- **Cyclone center tracking (regression)** → [`tracking.py`](tracking.py)  
  Run: `python tracking.py [OPTIONS...]`  
  Documentation: [tracking.md](tracking.md)

- **Inference** → [`inference_classification.py`](inference_classification.py)  
  Run: `python inference_classification.py [OPTIONS...]`  
  Documentation: [inference_classification.md](inference_classification.md)

---

## Dataset Pipeline
- **Building datasets**: [Build_dataset_videoMAE.md](Build_dataset_videoMAE.md)
- **Dataset experiments / relabeling**: [Experiment_dataset.md](Experiment_dataset.md)
- **Cyclone tracks analysis (Manos)**: [Analyze_Manos_tracks.md](Analyze_Manos_tracks.md)

---

## Training Workflows
- **Pretraining (MAE auto-supervised)**: [specialization.md](specialization.md)
- **Fine-tuning (classification)**: [classification.md](classification.md)
- **Tracking (regression)**: [tracking.md](tracking.md)

Cluster jobs (HPC/Slurm) are referenced in these docs.

---

## Inference and Post-processing
- [inference_classification.md](inference_classification.md)
- [Predict_general_data.md](Predict_general_data.md)
- [View_MED_val_preds.md](View_MED_val_preds.md)
- [View_test_tiles.md](View_test_tiles.md)

---

## Tracking Visualization
- [View_tracking_tiles.md](View_tracking_tiles.md)

---

## Evaluation and Metrics
- [metrics.md](metrics.md)
- [Model_stats.md](Model_stats.md)

---

## Notes
- No testing framework, linting rules, or formatting conventions are enforced.  
- This is a **research-driven project**: flexibility and exploration take precedence over rigid structure.  
- Agents should use this file as an **index** to locate the right documentation and scripts.

