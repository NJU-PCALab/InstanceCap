<div align="center">

# InstanceCap

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://github.com/NJU-PCALab/InstanceCap) [![Models](https://img.shields.io/badge/🤗HF-Captioner-yellow)](https://huggingface.co/AnonMegumi/InstanceCap-Captioner)

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

**3DVAE score**  and **CLIP SenbySen** [usage guide](.\eval).

<img src="..\assets\visual.png"  width="50%" align="center"/>

## InstanceCap-Captioner
We present **InstanceCap-Captioner**, a video captioning model fine-tuned from *Qwen2.5VL-7B* on the InstanceVid dataset. It enables end-to-end generation of high-quality video descriptions and annotations.

### 1. Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "AnonMegumi/Instance-Captioner", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("AnonMegumi/Instance-Captioner")

# Define input messages
messages = [
    {
        "role": "system",
        "content": "You are a wonderful video captioner."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path-to-your-video",  # Replace with your video path
                "max_pixels": 16384,  # Optional parameter
            },
            {"type": "text", "text": "Describe the video in detail."},
        ],
    }
]

# Preprocess inputs
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
).to("cuda")

# Perform inference
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)
print(output_text)
```

### 2. Example Output
For the sample video [space_woaudio.mp4](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4), the model generates:


```
A man, middle-aged with short, graying hair and dressed in a dark blue polo shirt with a small logo on the left chest, stands at the bottom-middle of the frame. He speaks and gestures with his hands, his facial expressions changing, suggesting he is presenting or explaining something. The scene is a control room with a sleek and organized appearance. Multiple computer monitors display maps and data, and flags of various countries adorn the walls, indicating an international collaboration. The room has a clean, professional look with a polished floor and well-lit atmosphere. The camera remains static, capturing the man from the waist up, maintaining a medium distance shot.

The video transitions to a space station where two men, one in a black shirt and the other in a blue shirt, are floating and gesturing, likely engaged in a conversation or presentation. The space station’s interior is detailed and sophisticated, with various equipment and instruments visible, including a large screen displaying data and maps. The walls are adorned with flags, and the overall environment is serene and well-organized, reflecting a professional and scientific setting. The camera movement is gentle, with a slight zoom-in effect, maintaining a steady and stable shot from a medium distance, capturing the subjects from the waist up.
```
