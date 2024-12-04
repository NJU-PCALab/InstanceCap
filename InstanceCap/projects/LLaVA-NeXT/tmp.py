queries = [
    "<image>\nHow many animated characters are there in this image?",
    "Answer with a single number in decimal format. Give no explanations."
]


def generate_response(image):
    chat = []
    for query in queries:
        chat.append({"role": "user", "content": query})
        prompt = text_processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300)
        output = processor.decode(output[0], skip_special_tokens=True)

        input_ids = inputs["input_ids"]
        cutoff = len(text_processor.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ))
        answer = output[cutoff:]
        chat.append({"role": "assistant", "content": answer})
    return answer
