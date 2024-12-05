<div align="center">

# InstanceCap

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://github.com/NJU-PCALab/InstanceCap) [![Models](https://img.shields.io/badge/🤗HF-Models(comming_soon)-yellow)](https://github.com/NJU-PCALab/InstanceCap)

</div>

## InstanceCap Pipeline
### 1. Set Environment
- **From Global Video to Local Instances**: 
```bash
conda create -n AMC python=3.10 -y
conda activate AMC
```
create a virtual environment for Co-DETR, following this [link](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html#id2).
```bash
cd camera_motion
pip install -r requirements.txt
```

- **From Dense prompt to Structured Phrases**:
You can download the [weights](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2) of LLaVA-Video.
```bash
cd LLaVA-NeXT
conda create -n instancecap python=3.10 -y
conda activate instancecap

# llava-video
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# sam2
cd segment-anything-2
pip install -e .
```

⚠️⚠️⚠️ While you can try to merge two virtual environments, and this was possible in our attempt. But we strongly recommend splitting the environment into two for easier implementation of the environment configuration.

### 2. Quick Start
```bash
# AMC
conda activate AMC
# GPU usage:less than A6000 40GB*1
python run_co_detr_muti.py --nums 3 # By default, a maximum of three instances are detected.
python -m camera_motion.detect_muti

# MLLM
conda activate instancecap
# GPU usage: A6000 40GB*5
python instancecap.py
```

### 3. Caption Quantitative Evaluation

**3DVAE score**  and **CLIP SenbySen** [usage guide](.\eval\README.md).

<img src="..\assets\visual.png"  width="50%" align="center"/>

## InstanceCap-Captioner
**Comming soon**
