<div align="center">

# Caption Quantitative Evaluation

[![Video archive](https://img.shields.io/badge/🤗HF-500_video_archive-yellow)](https://huggingface.co/datasets/AnonMegumi/500vid/tree/main)

</div>

Section 4.1 of the [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_InstanceCap_Improving_Text-to-Video_Generation_via_Instance-aware_Structured_Caption_CVPR_2025_paper.pdf) describes recaption-based reconstruction evaluation on **500 videos selected from OpenVid-1M**.

The public [500vid repository](https://huggingface.co/datasets/AnonMegumi/500vid) provides [500.zip](https://huggingface.co/datasets/AnonMegumi/500vid/resolve/main/500.zip?download=true) (about 4.82 GB). Hugging Face indexes 500 video rows and exposes paths such as `openvid500/000.mp4` inside this archive, consistent with this script's default `vid500/openvid500` path.

[InstanceVid](https://huggingface.co/datasets/AnonMegumi/InstanceVid) is the separate 22K training-data release, with training annotations under `train/` and source videos obtained by their OpenVid-1M identifiers. It also hosts `test.tar`; its relationship to `500.zip` is not established here. The previous “100 videos” instruction does not describe the 500-video protocol in the final paper.

Run the following from `InstanceCap/eval`, after installing the evaluation dependencies:

```bash
mkdir -p vid500 results
curl -L --fail \
  "https://huggingface.co/datasets/AnonMegumi/500vid/resolve/a8a0060511b2ce444f8ea079da109f7b8d42b6ea/500.zip?download=true" \
  -o vid500/500.zip
unzip vid500/500.zip -d vid500

python cal_3dvae_score.py \
  --model-path /absolute/path/to/cogvideox/vae \
  --ori_videos vid500/openvid500 \
  --gen_videos /absolute/path/to/reconstructed_videos
```

Inspect the extracted directory and verify that it contains 500 `.mp4` files before evaluation. Generate one caption per original video and use those captions to reconstruct videos with CogVideoX-5b, as described in the paper. The scoring script does not generate captions or reconstructed videos. Both directories must contain matching numbers of videos in corresponding **natural filename order**: the script pairs sorted files by position, not by video ID. Retaining the original filenames for reconstructed videos is recommended.

Pass the local VAE directory containing its configuration and weights explicitly with `--model-path`. [CogVideoX-5b VAE files](https://huggingface.co/zai-org/CogVideoX-5b/tree/main/vae) are available; the previous README linked CogVideoX1.5-5B while the script default refers to CogVideoX-5b. Record the checkpoint revision used and do not assume the versions produce interchangeable scores. Creating `results` above is necessary because the current script creates `.results` but writes logs to `results`.

For the additional sentence-by-sentence CLIP score, place the [CLIP weights and tokenizer](https://huggingface.co/openai/clip-vit-base-patch32) at `InstanceCap/checkpoints/clip-vit-base-patch32` and run:

```bash
python senbysen.py --dir_videos /absolute/path/to/original_videos --prompts /absolute/path/to/captions.txt
```

Provide one non-empty caption per line in the same natural filename order as the videos. This CLIP script is separate from the final paper's VQAScore, implemented in `vqascore.py`.

<img src="..\..\assets\eval.png" width="50%" align="center"/>
