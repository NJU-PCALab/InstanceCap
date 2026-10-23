<div align="center">

# Caption Quantitative Evaluation

[![Test set](https://img.shields.io/badge/🤗HF-Test_set-yellow)](https://huggingface.co/datasets/AnonMegumi/InstanceVid/tree/main)

</div>

First [download](https://github.com/NJU-PCALab/InstanceCap) the test set with 100 videos to your desired location. Secondly, you need to prepare the weights of [3DVAE](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main/vae) and [CLIP](https://huggingface.co/openai/clip-vit-base-patch32).

```bash
python cal_3dvae_score.py --ori_videos xxx --gen_videos xxx
python senbysen.py --dir_videos xxx ---prompts xxx
```
<img src="..\..\assets\eval.png"  width="50%" align="center"/>

