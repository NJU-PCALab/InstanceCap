from openai import OpenAI
import json
from human_hit import example_message, split_example_message

def split_text_by_break(text):
    parts = [part.strip() for part in text.split('BREAK')]
    return parts

def get_response(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    results = str(completion.choices[0].message.content).replace('\n', '')
    return results
def get_answer(caption):
    system_text = "You are a good prompt enhancement assistant, you will follow the original text on the basis of following the instructions of the user to supplement and enhance the details of the original prompt, both to ensure rationality and to ensure the high quality of the text.\n"\
        "Select the appropriate ones of following words in your description: kaleidoscopic, delicate, grand, gentle, soothing, cool, mature, solitary, worn, chaotic, dramatic, cozy, shimmering,  desolate, serene, weathered, whispering, loose-fitting, vibrant, tranquil, dimly-lit, purplish, introspective,  artfully, sleek, energetic, overcast, brilliant, slender, graceful, picturesque, whimsical, contented, gentle, warm,  tender, pastel-colored, elegant.\n" \
        "Do not use any of the following negative words when describing: dull, rough, harsh, chaotic, cluttered, bleak, uninspired, garish, stiff, unrefined, artificial, heavy, disorderly,  grim, rusty, faded, cramped, jarring, obtrusive, awkward, ordinary, harsh, gloomy, cold, rigid, overcrowded,  mismatched, messy, uneven, tacky, lifeless, unbalanced, heavy-handed, overbearing, dissonant, grating, oversaturated, unpleasant, rigid, blur.\n"\
        f"Please use the \"subject\" + \"attribute\" structure more often. For example, \"An old man, his hair is white.\"\n"

    system_content = [
        {"type": "text", "text": system_text}
    ]

    # step 1: generate a story
    story_text = f"Write a reasonably detailed silent film for this text: \n{caption}\n "
    story_content = [
        {"type": "text", "text": story_text}
    ]
    story_prompt = [{"role": "system", "content": system_content}]
    story_prompt.extend(example_message)
    story_prompt.append({"role": "user", "content": story_content})
    story = get_response(story_prompt)

    # step 2:split objects from text
    split = f"Identify the subjects contained in the following passage, using BREAK to separate them in one line:  \n {story} \n" \
            f"Please distinguish between instances and scenes, and return only instances, such as: \"a man BREAK a cat BREAK a BREAK a cat BREAK a cup...\"" \
            f"Please ensure that only the main instance are output, no more than 3, and ensure that all objects in \n{caption}\n appear \n" \
            "Note that the instance you extract must be an entity that can be touched.\n" \
             "When there are multiple targets, you need to give them separately."

    split_prompt = [{"role":"system", "content": system_content}]
    split_prompt.extend(split_example_message)
    split_prompt.append({"role": "user", "content": split})
    objs = split_text_by_break(get_response(split_prompt))

    def get_app(obj):
        app = f"""
            Please describe appearance of {obj}. If the original text lacks specific descriptions, provide reasonable and vivid descriptions based on the context:
            \n{story}\n
            Please give your answer without any prefix and don't mention anything unrelated to {obj}.\n
            Your description must be within one sentence.
            """
        return app

    def get_act(obj):
        act = f"""
            Please describe action of {obj}. If the original text lacks specific descriptions, provide reasonable and vivid descriptions based on the context:
            \n{story}\n
            Please give your answer without any prefix and don't mention anything unrelated to {obj}.\n
            Your description must be within one sentence.
            """
        return act

    def get_pos(obj):
        pos = f"""
        Based on the text \n{story}\n, select a position change for {obj} that fits the context. You need to choose from the following combinations:
        - {obj} from bottom/center/top-left/middle/right to bottom/center/top-left/middle/right
        - {obj} stays at bottom/center/top-left/middle/right
        - If you think {obj} will leave the frame, add "and out of frame in the end" at the end. 
        Please ensure that the hyphen "-" is not omitted.
        For example:
        - A man from bottom-left to center.
        - A cat stays at top-right.
        - A woman from middle-left to middle-right and out of frame in the end.
        """
        return pos

    background = f"Follow the original text \n{story}\n  describe the background detail of the scene directly. If the original text lacks specific descriptions, create an imaginative and engaging one-sentence description based on the context. Do not include information beyond the scene description."
    camera = f"Based on the text provided \n{story}\n, write a suitable and varied description of camera movements and angles in one sentence."
    dict_ = {"Global Description": f"{caption}", "Story":f"{story}", "Structural Description": {"Main Instance":{}}}

    for id in range(len(objs)):
        obj = objs[id]
        dict_["Structural Description"]["Main Instance"][f"No.{id}"] = {}
        dict_["Structural Description"]["Main Instance"][f"No.{id}"]['Class'] = obj
        obj_prompt = [{"role":"system", "content": system_content}]

        # get app
        app = get_app(obj)
        obj_prompt.append({"role": "user", "content": app})

        app_response = get_response(obj_prompt)
        dict_["Structural Description"]["Main Instance"][f"No.{id}"]['Appearance'] = app_response
        obj_prompt.append({"role": "assistant", "content": app_response})

        # get actions and motion
        act = get_act(obj)
        obj_prompt.append({"role": "assistant", "content": act})
        act_response = get_response(obj_prompt)
        dict_["Structural Description"]["Main Instance"][f"No.{id}"]['Actions and Motion'] = act_response
        obj_prompt.append({"role": "assistant", "content": act_response})

        # get position
        pos = get_pos(obj)
        obj_prompt.append({"role": "assistant", "content": pos})
        pos_response = get_response(obj_prompt)
        dict_["Structural Description"]["Main Instance"][f"No.{id}"]['Position'] = pos_response

    # get background detail
    background_prompt = [{"role":"system", "content": system_content}, {"role": "user", "content": background}]
    background_response = get_response(background_prompt)
    dict_["Structural Description"]['Background Detail'] = background_response

    # get camera detail
    camera_prompt = [{"role":"system", "content": system_content}, {"role": "user", "content": camera}]
    camera_response = get_response(camera_prompt)
    dict_["Structural Description"]['Camera Movement'] = camera_response

    return dict_

if __name__ == "__main__":
    client = OpenAI(api_key="api-key")
    txt_path = ""

    ori_list = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            ori_list.append(line.strip())

    with open('enhance prompt.jsonl', 'w', encoding='utf-8') as file:
        for ori in ori_list:
            new = get_answer(ori)
            new = json.dumps(new, ensure_ascii=False)
            file.write(new + '\n')
