category_prompts = {
    'person': "Please focus primarily on the person's facial expressions, attire, age, gender, and race in the video and give description in detail. Please mention if there are any necklaces, watches, hat or other decoration; otherwise, there’s no need to bring them up.",
    'bicycle': "Please describe the bicycle in terms of color, type, size, condition, and any distinctive marks or decorations. Include details such as the presence of baskets, reflectors, or any branding.",
    'car': "Please describe the car by its color, make, model, condition, license plate (if visible), and any distinguishing features such as stickers, dents, or modifications.",
    'motorcycle': "Please provide details about the motorcycle, including its color, type, brand, engine size, condition, and any accessories such as saddlebags or custom modifications.",
    'airplane': "Please describe the airplane by its type (commercial, private, etc.), airline brand, color scheme, size, and any visible markings such as logos or tail numbers.",
    'bus': "Please describe the bus by its color, type (public, school, etc.), condition, any branding or advertising on its surface, and the route number or destination if visible.",
    'train': "Please describe the train by its type (freight, passenger, high-speed, etc.), color, length, condition, and any visible logos or car numbers.",
    'truck': "Please describe the truck by its type (pickup, semi, etc.), color, make, model, any visible logos or branding, and details such as cargo or modifications.",
    'boat': "Please describe the boat by its type (sailboat, motorboat, yacht, etc.), size, color, condition, and any identifying features like registration numbers or flags.",
    'traffic light': "Please mention the current state of the traffic light (red, yellow, green), its location, and any additional details like the presence of pedestrian signals.",
    'fire hydrant': "Please describe the fire hydrant by its color, condition, and any notable features such as signs, markings, or proximity to other objects.",
    'stop sign': "Please describe the stop sign's condition, location, and any visible obstructions or markings on it.",
    'parking meter': "Please describe the parking meter by its condition, type (modern, traditional), and any visible information like pricing or operational status.",
    'bench': "Please describe the bench by its material, color, condition, and any distinctive features such as inscriptions, decorations, or nearby objects.",
    'bird': "Please describe the bird by its species (if identifiable), color, size, behavior, and any unique markings or features.",
    'cat': "Please describe the cat by its color, breed (if identifiable), size, behavior, and any distinguishing features such as collars or patterns.",
    'dog': "Please describe the dog by its breed (if identifiable), color, size, behavior, and any accessories such as collars or leashes.",
    'horse': "Please describe the horse by its color, breed (if identifiable), size, behavior, and any accessories such as saddles or reins.",
    'sheep': "Please describe the sheep by its color, size, behavior, and any distinguishing features such as markings or tags.",
    'cow': "Please describe the cow by its color, breed (if identifiable), size, behavior, and any distinguishing features such as tags or markings.",
    'elephant': "Please describe the elephant by its size, tusk length, condition, and any unique features such as markings or behavior.",
    'bear': "Please describe the bear by its color, size, species (if identifiable), and behavior. Mention any distinctive markings.",
    'zebra': "Please describe the zebra by its size, stripe pattern, behavior, and any distinguishing features such as scars or markings.",
    'giraffe': "Please describe the giraffe by its height, color, pattern, and behavior. Mention any distinctive features.",
    'backpack': "Please describe the backpack by its color, size, condition, brand, and any distinctive features such as patches or decorations.",
    'umbrella': "Please describe the umbrella by its color, size, condition, and any visible logos or decorations.",
    'handbag': "Please describe the handbag by its color, size, material, brand, and any distinguishing features such as logos or decorations.",
    'tie': "Please describe the tie by its color, pattern, material, and any visible branding or distinctive features.",
    'suitcase': "Please describe the suitcase by its color, size, condition, brand, and any distinguishing features such as stickers or tags.",
    'frisbee': "Please describe the frisbee by its color, size, condition, and any logos or decorations.",
    'skis': "Please describe the skis by their color, size, condition, and any visible branding or decorations.",
    'snowboard': "Please describe the snowboard by its color, size, condition, brand, and any distinctive features or decorations.",
    'sports ball': "Please describe the sports ball by its type (soccer, basketball, etc.), color, condition, and any logos or patterns.",
    'kite': "Please describe the kite by its color, shape, size, and any distinctive patterns or decorations.",
    'baseball bat': "Please describe the baseball bat by its color, material, condition, and any visible branding or markings.",
    'baseball glove': "Please describe the baseball glove by its color, size, condition, and any visible branding or markings.",
    'skateboard': "Please describe the skateboard by its color, size, condition, and any distinctive patterns or stickers.",
    'surfboard': "Please describe the surfboard by its color, size, condition, brand, and any distinctive patterns or decorations.",
    'tennis racket': "Please describe the tennis racket by its color, condition, brand, and any distinctive features or patterns.",
    'bottle': "Please describe the bottle by its color, size, material, and any labels or branding.",
    'wine glass': "Please describe the wine glass by its shape, size, condition, and any visible decorations or contents.",
    'cup': "Please describe the cup by its color, size, material, and any visible decorations or contents.",
    'fork': "Please describe the fork by its material, condition, and any distinctive features or decorations.",
    'knife': "Please describe the knife by its type (butter, chef's, etc.), material, condition, and any distinctive features.",
    'spoon': "Please describe the spoon by its material, size, condition, and any distinctive features.",
    'bowl': "Please describe the bowl by its color, size, material, and any visible decorations or contents.",
    'banana': "Please describe the banana by its color, size, ripeness, and any visible spots or marks.",
    'apple': "Please describe the apple by its color, size, condition, and any visible spots or markings.",
    'sandwich': "Please describe the sandwich by its type, ingredients, size, and any visible features.",
    'orange': "Please describe the orange by its color, size, condition, and any visible spots or markings.",
    'broccoli': "Please describe the broccoli by its color, size, condition, and any distinctive features.",
    'carrot': "Please describe the carrot by its color, size, condition, and any visible spots or markings.",
    'hot dog': "Please describe the hot dog by its size, ingredients, and any visible toppings or condiments.",
    'pizza': "Please describe the pizza by its size, type, toppings, and any distinctive features.",
    'donut': "Please describe the donut by its color, size, type, and any visible decorations or toppings.",
    'cake': "Please describe the cake by its size, type, decorations, and any visible features such as layers or toppings.",
    'chair': "Please describe the chair by its color, material, condition, and any distinctive features or decorations.",
    'couch': "Please describe the couch by its color, size, material, condition, and any visible decorations or cushions.",
    'potted plant': "Please describe the potted plant by its type, size, condition, and any distinctive features such as pot design or plant health.",
    'bed': "Please describe the bed by its size, color, condition, and any visible bedding or decorations.",
    'dining table': "Please describe the dining table by its size, color, material, condition, and any distinctive features or decorations.",
    'toilet': "Please describe the toilet by its color, condition, and any distinctive features or decorations.",
    'tv': "Please describe the television by its size, condition, and any visible logos or branding.",
    'laptop': "Please describe the laptop by its color, brand, size, condition, and any visible stickers or decorations.",
    'mouse': "Please describe the mouse by its color, brand, condition, and any distinctive features or decorations.",
    'remote': "Please describe the remote by its color, size, condition, and any visible buttons or branding.",
    'keyboard': "Please describe the keyboard by its color, brand, condition, and any distinctive features or decorations.",
    'cell phone': "Please describe the cell phone by its color, brand, size, condition, and any visible stickers or decorations.",
    'microwave': "Please describe the microwave by its color, size, condition, and any visible buttons or branding.",
    'oven': "Please describe the oven by its color, size, condition, and any visible buttons or branding.",
    'toaster': "Please describe the toaster by its color, size, condition, and any distinctive features or branding.",
    'sink': "Please describe the sink by its material, condition, and any visible fixtures or decorations.",
    'refrigerator': "Please describe the refrigerator by its color, size, brand, condition, and any visible magnets or decorations.",
    'book': "Please describe the book by its color, size, title, author, and any visible wear or distinguishing features.",
    'clock': "Please describe the clock by its color, shape, size, condition, and any distinctive features or branding.",
    'vase': "Please describe the vase by its color, size, shape, material, and any distinctive decorations or patterns.",
    'scissors': "Please describe the scissors by their color, size, condition, and any distinctive features or branding.",
    'teddy bear': "Please describe the teddy bear by its color, size, condition, and any distinctive features or accessories.",
    'hair drier': "Please describe the hair drier by its color, size, condition, brand, and any distinctive features.",
    'toothbrush': "Please describe the toothbrush by its color, size, condition, and any distinctive features or branding."
}

caption = ["An astronaut is riding a horse.",
           "Two wolves were hunting a rabbit in the snow.",
           "Two men were fishing and a woman was standing by.",
           "A completely still tree."]

story = [
    "A sleek white spacesuit clings to the astronaut, each segment glinting faintly under the soft moonlight with a smooth, metallic sheen. "
    "The helmet visor, a dark, glossy curve, reflects the starry expanse, masking the rider’s face. The astronaut’s gloved hands grip leather reins, steady but light. "
    "Beneath them, a deep chestnut horse charges forward, its short, velvety coat rippling with every powerful stride. "
    "The horse’s long, black mane streams wildly in the wind, whipping against the astronaut’s suit as they gallop across the barren ground. "
    "The horse’s hooves strike the earth with rhythmic force, scattering dry dust into swirling clouds. "
    "Each movement of the astronaut mirrors the animal's fluid motion—leaning forward, rising slightly in the stirrups as the horse surges ahead, "
    "then sinking back into the saddle during brief moments of stillness. "
    "The glinting suit contrasts with the horse's dark coat, their forms moving as one against the vast, quiet night.",

    "In the vast whiteness of the snowy forest, two wolves stalk quietly. "
    "One is large and heavyset, its fur a mix of dark grays and black, blending into the shadows of the trees. "
    "The other is leaner and quicker, with a pale, almost silver coat that glimmers in the winter light. "
    "Their eyes are sharp and focused, and each step is measured—one wolf pacing powerfully, the other darting with agile precision. "
    "Ahead of them, a small white rabbit crouches in the snow, nearly invisible but for the black tips of its ears and nose. "
    "Suddenly, the rabbit bolts, kicking up a spray of powder. "
    "The dark wolf charges forward with brute force, while the silver one sprints with smooth, loping strides. "
    "The rabbit’s movements are frantic and unpredictable, weaving between tree roots and leaping over snowdrifts. "
    "The wolves flank it from both sides, their breaths clouding the air, closing in like shadows. "
    "At the last moment, the rabbit dives into a narrow burrow, disappearing into safety. "
    "The dark wolf sniffs the ground, frustrated, while the silver wolf paces briefly before the two melt back into the forest, "
    "leaving only their crisscrossing prints behind in the snow.",

    "On the bank of a quiet river, two men sit, fishing rods in hand, lost in their shared pursuit. "
    "One man is tall and lanky, his clothes slightly rumpled and sleeves rolled up as if fishing were an afterthought. "
    "His hair sticks out messily beneath a flat cap, and his exaggerated gestures reveal his impatience—he tugs too hard at his line, pulling in a clump of weeds. "
    "The other man is shorter and stockier, with neatly combed hair and a vest buttoned tight over his belly, giving him a more proper air. "
    "He watches the first man’s struggles with a smug grin, adjusting his hat to shade his eyes, clearly taking pride in his methodical approach. "
    "Standing nearby is a woman, poised and elegant, her neat dress and parasol making her seem out of place among the two fishermen. "
    "A ribbon in her hair catches the light breeze, and she stands with her arms crossed lightly, the hint of a smile playing on her lips as she observes the men’s playful bickering. "
    "The lanky man throws up his arms in frustration, while the stocky one chuckles under his breath, confident his patience will pay off.",

    "The film opens with a solitary tree standing tall in the middle of a vast field, perfectly motionless. "
    "Its trunk is thick and gnarled, twisting upward into jagged, angular branches that stretch like frozen lightning across the sky. "
    "The sparse leaves, clinging stubbornly to the crooked limbs, hang still, untouched by even the slightest breeze. "
    "As the sun rises, the world around it bustles—wind sweeps through the grass, animals scurry by, clouds drift overhead—but the tree remains utterly unmoving. "
    "Throughout the day, people pass by: a child throws a rock at the trunk, a farmer rests in its shade, and a couple carves initials into the bark. "
    "Time shifts rapidly, marked by the changing sky—sunlight fades, storms lash the field, snow blankets the branches—yet the tree stays unyielding, "
    "its jagged silhouette cutting against the changing seasons. "
    "In the final shot, the night settles in, and with only the distant howl of wind, the tree stands in absolute silence, unchanged, as if waiting for something that never comes."
]

def build_content(text):
    return  [{"type": "text", "text": text}]

example_message = [
    {
        "role": "user",
        "content": build_content(f'Write a reasonably silent film for this text: \n{caption[0]}\n'),
    },
    {
        "role": "assistant",
        "content": build_content(
            story[0]
        ),
    },
    {
        "role": "user",
        "content": build_content(f'Write a reasonably detailed silent film for this text: \n{caption[1]}\n'),
    },
    {
        "role": "assistant",
        "content": build_content(
            story[1]
        ),
    },
    {
        "role": "user",
        "content": build_content(f'Write a reasonably detailed silent film for this text: \n{caption[2]}\n'),
    },
    {
        "role": "assistant",
        "content": build_content(
            story[2]
        ),
    },
    {
        "role": "user",
        "content": build_content(f'Write a reasonably detailed silent film for this text: \n{caption[3]}\n'),
    },
    {
        "role": "assistant",
        "content": build_content(
            story[3]
        ),
    },
]


def get_split(s, c):
    split = f"Identify the subjects contained in the following passage, using BREAK to separate them in one line:  \n {s} \n" \
            f"Please distinguish between instances and scenes, and return only instances, such as: \"a man BREAK two cat BREAK a cup...\"" \
            f"Please ensure that only the main instance are output, no more than 5, and ensure that all objects in \n{c}\n appear \n" \
            "Note that the instance you extract must be an entity that can be touched.\n" \
             "When there are multiple targets, you need to give them separately."

    return split

split_example_message = [
    {
        "role": "user",
        "content": build_content(get_split(story[0], caption[0])),
    },
    {
        "role": "assistant",
        "content": build_content(
            "a astronaut BREAK helmet BREAK a horse"
        ),
    },
    {
        "role": "user",
        "content": build_content(get_split(story[1], caption[1])),
    },
    {
        "role": "assistant",
        "content": build_content(
            "a wolf BREAK a wolf BREAK a rabbit"
        ),
    },
    {
        "role": "user",
        "content": build_content(get_split(story[2], caption[2])),
    },
    {
        "role": "assistant",
        "content": build_content(
            "a man BREAK a man BREAK a woman"
        ),
    },
    {
        "role": "user",
        "content": build_content(get_split(story[3], caption[3])),
    },
    {
        "role": "assistant",
        "content": build_content(
            "a tree BREAK a boy BREAK a man BREAK a woman"
        ),
    },
]

