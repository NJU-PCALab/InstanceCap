import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import numpy as np

model_id = "/root/autodl-tmp/LLaVA-NeXT/models/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_system_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

system_prompt = read_system_prompt("/root/autodl-tmp/LLaVA-NeXT/system_prompt.txt")
conv = []
conv.append({"role": "SYSTEM", "content": [{"type": "text", "text": system_prompt}]})
conv.append({"role": "USER", "content": [{"type":"video"}, {"type": "text", "text": "Describe Target Objects a man, a dog, a glass, use detailed JSON format like # Structured Output."}]})

video_path = '/root/autodl-tmp/LLaVA-NeXT/videos/example.mp4'
container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
output = processor.decode(output[0], skip_special_tokens=True)

input_ids = inputs["input_ids"]
cutoff = len(processor.decode(
    input_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
))
answer = output[cutoff+1:]
print(answer)

conv.append({"role": "ASSISTANT", "content": [{"type": "text", "text": answer}]})
conv.append({"role": "USER", "content": [{"type": "text", "text": "Based on the JSON you give, expand each section in more detail, and keep JSON format."}]})
prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
output = processor.decode(output[0], skip_special_tokens=True)

input_ids = inputs["input_ids"]
cutoff = len(processor.decode(
    input_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
))
answer = output[cutoff+1:]
print(answer)

conv.append({"role": "ASSISTANT", "content": [{"type": "text", "text": answer}]})
conv.append({"role": "USER", "content": [{"type": "text", "text": "Based on the JSON you give, expand Action section in more detail, and keep JSON format."}]})
prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
output = processor.decode(output[0], skip_special_tokens=True)

input_ids = inputs["input_ids"]
cutoff = len(processor.decode(
    input_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
))
answer = output[cutoff+1:]
print(answer)
