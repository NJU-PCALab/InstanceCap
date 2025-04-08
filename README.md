<div align="center">

# InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption

> **[Tiehan Fan](https://scholar.google.com.hk/citations?user=F11LXvYAAAAJ&hl=zh-CN)**<sup>1* </sup>, **[Kepan Nan](https://scholar.google.com.hk/citations?hl=zh-CN&user=PXmlku8AAAAJ)**<sup>1*</sup>, **[Rui Xie](https://scholar.google.com.hk/citations?user=Dzr3D_EAAAAJ&hl=zh-CN&oi=sra)**<sup>1</sup>, **[Penghao Zhou](https://scholar.google.com.hk/citations?hl=zh-CN&user=yWq1Fd4AAAAJ)**<sup>2</sup>, **[Zhenheng Yang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Ds5wwRoAAAAJ)**<sup>2</sup>, **[Chaoyou Fu](https://bradyfu.github.io/)**<sup>1</sup>, **[Xiang Li](https://implus.github.io/)**<sup>3</sup>, **[Jian Yang](http://www.patternrecognition.cn/~jian/)**<sup>1</sup>, **[Ying Tai](https://tyshiwo.github.io/)**<sup>1✉</sup>  
>
> <sup>1</sup> Nanjing University   <sup>2</sup> ByteDance   <sup>3</sup> Nankai University   <sup>*</sup>Equal Contribution   <sup>✉</sup>Corresponding Author

 [![code](https://img.shields.io/badge/Github-Code-blue.svg?logo=github)](https://github.com/NJU-PCALab/InstanceCap) [![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2412.09283) [![Hugging Face Dataset](https://img.shields.io/badge/🤗HF-Dataset-yellow.svg)](https://huggingface.co/datasets/AnonMegumi/InstanceVid/tree/main) [![Models](https://img.shields.io/badge/🤗HF-Captioner-yellow)](https://huggingface.co/AnonMegumi/InstanceCap-Captioner)
<!-- [![Models](https://img.shields.io/badge/🤗HF-Models(comming_soon)-yellow)](https://github.com/NJU-PCALab/InstanceCap) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Website(comming_soon)-green)](https://github.com/NJU-PCALab/InstanceCap)  -->


</div>

## 🗣️Abstract
Text-to-video generation has evolved rapidly in recent years, delivering remarkable results. Training typically relies on video-caption paired data, which plays a crucial role in enhancing generation performance. However, current video captions often suffer from insufficient details, hallucinations and imprecise motion depiction, affecting the fidelity and consistency of generated videos. In this work, we propose a novel instance-aware structured caption framework, termed $InstanceCap$, to achieve instance-level and fine-grained video caption for the first time. Based on this scheme, we design an auxiliary models cluster to convert original video into instances to enhance instance fidelity. Video instances are further used to refine dense prompts into structured phrases, achieving concise yet precise descriptions. Furthermore, a $22K\ InstanceVid$ dataset is curated for training, and an enhancement pipeline that tailored to $InstanceCap$ structure is proposed for inference. Experimental results demonstrate that our proposed InstanceCap significantly outperform previous models, ensuring high fidelity between captions and videos while reducing hallucinations.

<img src="assets\compare_caption_v2.png"  width="100%" align="center"/>

## 🔥News
- **Comming soon**: 🎯 T2V model weights ……
- **2025.4.8**：🤖 **InstanceCap-Captioner** is released.
- **2025.2.27**: 🎉 Paper Accepted to **CVPR 2025!** 🎉
- **2024.12.13**: 🚀 Our code, dataset and arXiv paper are released.

## 🔍️InstanceCap
We provide our major contribution, the python implementation of $InstanceCap$, in this repository, and you can install and use the full version of our proposal based on [guide fo InstanceCap](InstanceCap). Alternatively, you can use the Captioner we tweaked to Qwen2.5-VL-7B based on $InstanceVid$ to get a high quality description with less difficulty.

<img src="assets\pipeline-1.png"  width="100%" align="center"/>
<img src="assets\pipeline-2.png"  width="100%" align="center"/>

## 📽️InstanceVid
### Key Features of InstanceVid

1. **Instance-aware**: The dataset contains 22K videos with corresponding captions, which are annotated with instance-level descriptions.
2. **Fine-grained Structured Caption**: The dataset is designed to be used for fine-grained structured captioning, where each instance is described by a structured caption.

### Meta Files
<img src="assets\statistics.png"  width="35%" align="right"/>

We release $InstanceVid$, containing 22K videos and captions. The meta file for this is provided in [HuggingFace Dataset](https://huggingface.co/datasets/AnonMegumi/InstanceVid/tree/main/train) with json format, JSON contains the following properties:

- **Video**: This is the name or file path of the video being referenced.
- **Global Description**: A brief summary of the video content, providing context about what is happening in the video.
- **Structured Description**:  Detailed breakdown of the video content, including information on the main instances (such as people and objects) and their actions.
  - Main Instance: Represents a specific person or object in the video.
    - No.0
      - Class: The type or category of the instance (e.g., person, car).
      - Appearance: A description of the physical appearance of the instance.
      - Actions and Motion: What the instance is doing, including its movements or posture.
      - Position: The position of the instance in the frame (e.g., bottom-left, bottom-right).
    - No.1 
      - ...
  - **Background Detail**:  A description of the environment in the video background, such as the setting, props, and any significant details about the location.
  - **Camera Movement**: Information about how the camera behaves during the video, including whether it is static or dynamic and the type of shot.

## 🏋🏽InstanceEnhancer
We share the tuning-free [InstanceEnhancer](./InstanceEnhancer) implementation process in this repository. It can easily enhance the short prompt of user input into a structured prompt,  and achieve the alignment of train-inference text data distribution. We provide prompt implementation based on GPT-4o version, you can also migrate to other models to get similar results.

## 📏Inseval
We implement a CoT reasoning framework for generating structured QA responses to ensure objective and consistent evaluation, allowing us to derive instance-level evaluation scores that align closely with human perception and preferences. This approach provides a more nuanced and reliable assessment of instance-level generation quality. 
Following [this guide](Inseval), you can use Inseval to evaluate your own generation model.

<img src="assets\detail.png"  width="32%" align="right"/>
<img src="assets\inseval_table.png"  width="65%" align="center"/>
<img src="assets\action.png"  width="65%" align="center"/>

## 👏Acknowledgment

 Our work is benefited from [HailuoAI](https://hailuoai.com/video), [OpenSora](https://github.com/hpcaitech/Open-Sora), [LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT), [CogvideoX](https://github.com/THUDM/CogVideo?tab=readme-ov-file), [Qwen](https://huggingface.co/Qwen) and [OpenVid-1M(data)](https://huggingface.co/datasets/nkp37/OpenVid-1M), without their excellent effects, we would have faced a lot of resistance in implementation.

## 📖BibTeX
```
@misc{fan2024instancecapimprovingtexttovideogeneration,
      title={InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption}, 
      author={Tiehan Fan and Kepan Nan and Rui Xie and Penghao Zhou and Zhenheng Yang and Chaoyou Fu and Xiang Li and Jian Yang and Ying Tai},
      year={2024},
      eprint={2412.09283},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09283}, 
}

@article{nan2024openvid,
  title={OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation},
  author={Nan, Kepan and Xie, Rui and Zhou, Penghao and Fan, Tiehan and Yang, Zhenheng and Chen, Zhijie and Li, Xiang and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2407.02371},
  year={2024}
}
```

## 📧Contact Information
Should you have any inquiries, please contact fantiehan@outlook.com.
