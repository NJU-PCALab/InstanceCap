<div align="center">

# Inseval

</div>

## 1. Set Environment
Inseval follows the same environment setup as InstanceCap. In order to ensure the quality of the evaluation, we **strongly recommend** using the **72B** model for the evaluation. You can download the [weights](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2) of LLaVA-Video.
```bash
cd LLaVA-NeXT
conda create -n instancecap python=3.10 -y
conda activate instancecap

# llava-video
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
```

## 2. Inference
First, use your T2V model to reason about prompts under [dimensions](.\dimensions) and organize the results into the following form:
```bash
/videos
    /multiple
        /action
            ├── 0000.mp4
            ├── 0001.mp4
            ├── 0002.mp4
            ├── 0003.mp4
            ├── 0004.mp4
            ...
        /color
        ...
    /single
        ...
```

## 3. Evaluation
After modifying the files in inseval_singlet. py and inseval_multiple.py to the correct path:
```bash
bash run.sh
```
