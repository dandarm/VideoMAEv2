<div align="center">
<h1> VideoMAEv2 extension for Medicanes
<br>
<img src="misc/med_earth_icon.png" alt="Project Icon" width="100" />
</div>

# Overview

This repository adapts **VideoMAE v2** (Video Masked Autoencoder: [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)) to the study of *Mediterranean tropical-like cyclones* (Medicanes). It provides scripts for both **self-supervised pretraining** and **supervised fine-tuning** on satellite (SEVIRI MSG) imagery AirmassRGB composite datasets.

The repo is developed to:

* Pretrain VideoMAE for a **specialization phase** using unlabeled satellite video sequences.
* Fine-tune the pretrained model for **cyclone detection** and **center tracking**.

---

## Repository Structure

```
├── specialization.py            # Additional unsupervised pretraining ("specialization")
├── classification.py            # Fine-tuning entry point (classification)
├── engine_for_pretraining.py    # Training loop for pretraining
├── engine_for_finetuning.py     # Training loop for fine-tuning
├── datasets.py                  # Contains custom MedicaneDataset and TrackindDataset definitions 
├── build_dataset.py             # Tools to build dataset CSVs (tile-based)
├── make_dataset_from_rgb.py     # Entry points for dataset construction
└── arguments.py                 # Centralized training arguments
```
Check `README_MORE` for more in depth description of each module and notebook.

---

## Datasets

* **Unsupervised pretraining**: uses unlabeled video sequences (AirmassRGB), with loss for patch reconstruction
* **Supervised fine-tuning**: uses Medicanes tracks file for 
    * **classification training** for detection,
    * **Regression training** for center tracking.

---

# How to Run

## Installation

Please follow the instructions in [INSTALL.md](docs/INSTALL.md).

### Pretraining
```bash
python specialization.py [OPTIONS...]
```

### Fine-tuning: Medicane Detection 
```bash
python classification.py [OPTIONS...]
```

### Fine-tuning: Center Tracking 
```bash
python tracking.py [OPTIONS...]
```

### Inference From Folder
See [`docs/inference_from_folder.md`](docs/inference_from_folder.md).

---




## Download and Processing of AirmassRGB

To download and process **EUMETSAT satellite images** into **AirmassRGB composites**, use the script:


```bash
python medicane_utils/download_airmassRGB.py --start "2020-09-01 00:00" --end "2020-09-15 23:59"
```


## References

* [VideoMAE v1 (NeurIPS 2022)](https://arxiv.org/abs/2203.12602)
* [VideoMAE v2 (CVPR 2023)](https://arxiv.org/abs/2303.16727)


### [CVPR 2023] Official Implementation of VideoMAE V2

![flowchart](misc/VideoMAEv2_flowchart.png)

> [**VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**](https://arxiv.org/abs/2303.16727)<br>
> [Limin Wang](http://wanglimin.github.io/), [Bingkun Huang](https://github.com/congee524), [Zhiyu Zhao](https://github.com/JerryFlymi), [Zhan Tong](https://scholar.google.com/citations?user=6FsgWBMAAAAJ), [Yinan He](https://dblp.org/pid/93/7763.html), [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), [Yali Wang](https://scholar.google.com/citations?user=hD948dkAAAAJ), and [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl)<br>
> Nanjing University, Shanghai AI Lab, CAS<br>

---


