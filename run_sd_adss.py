import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from initno.pipelines.pipeline_sd_initno_adss import StableDiffusionInitNOPipeline
from initno.pipelines.pipeline_sd import StableDiffusionAttendAndExcitePipeline
from tqdm import tqdm
import json
import os
import random
import torch
import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import json
import time

import json, os
import numpy as np
import pandas as pd


def process_seed_dataframe_entity_only(df, top_k, root_dir, p_idx):
    df_sorted = df.sort_values("entity_mean", ascending=False)

    top_seeds = df_sorted["seed"].head(top_k).tolist()

    prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "top50_entity_seeds.json"), "w") as f:
        json.dump(top_seeds, f, indent=2)

    return top_seeds


def main(base_seed=4):
    guidance_scale_1 = 7.5
    num_infer_steps = 50
    seeds_per_prompt = 100
    top_k = 50
    root_dir = f"results/test{base_seed}"
    os.makedirs(root_dir, exist_ok=True)

    print("Loading pipe_attn...")
    pipe_attn = StableDiffusionInitNOPipeline.from_pretrained(
        "./stable_diffusion_2.1", local_files_only=True, torch_dtype=torch.float32
    ).to("mps")

    top_seeds_dict = {}

    scoring_plus_top50_times = []
    random50_times = []

    for p_idx, prompt in prompts_dict.items():
        print(f"\n=== Prompt {p_idx:03d}: {prompt}")
        token_idx_dict = prompt_token_index_dict[p_idx]
        print(token_idx_dict)

        random.seed(base_seed)
        seed_pool = random.sample(range(1_000_000), seeds_per_prompt)
        rows = []

        # === Time: scoring + top50 ===
        start_time = time.time()

        for seed in tqdm(seed_pool, desc=f"Scoring seeds p{p_idx:03d}"):
            row = {"seed": seed}
            for group, idx_list in token_idx_dict.items():
                if not idx_list:
                    row[f"{group}_mean"] = 0.0
                    continue
                gen = torch.Generator("cpu").manual_seed(seed)
                _, mean_dict = pipe_attn(
                    prompt=prompt,
                    token_indices=idx_list,
                    guidance_scale=guidance_scale_1,
                    generator=gen,
                    num_inference_steps=num_infer_steps,
                    result_root=None,
                    seed=seed,
                )
                row[f"{group}_mean"] = np.mean([mean_dict[i] for i in idx_list if i in mean_dict])
            rows.append(row)

        df = pd.DataFrame(rows)
        top_seeds = process_seed_dataframe_entity_only(df, top_k, root_dir, p_idx)
        top_seeds_dict[p_idx] = top_seeds
    
    del pipe_attn
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading pipe_sd...")
    pipe_sd = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        "./stable_diffusion_1.5", local_files_only=True
    ).to("cuda")


    for p_idx, prompt in prompts_dict.items():
        print(f"\n=== Generating images for Prompt {p_idx:03d}: {prompt}")
        token_idx_dict = prompt_token_index_dict[p_idx]
        prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
        ae_dir = os.path.join(prompt_dir, "adss")
        os.makedirs(ae_dir, exist_ok=True)
        # --- top50 ---
        #print("  ➤ Generating Attend‑Excite images (top50)…")
        print("  ➤ Generating Attend‑Excite images (top50)…")
        random.seed(base_seed)
        for seed in tqdm(top_seeds, desc="Top50"):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe_sd(
                prompt=prompt,
                token_indices=sum(token_idx_dict.values(), []),
                guidance_scale=guidance_scale_1,
                generator=gen,
                num_inference_steps=num_infer_steps,
                result_root=None,
                seed=seed,
            ).images[0]
            img.save(os.path.join(ae_dir, f"img_adss{seed}.jpg"))

        end_time = time.time()
        scoring_plus_top50_times.append(end_time - start_time)

        # === Time: random50 ===
        print("  ➤ Generating Attend‑Excite images (random50)…")
        random.seed(base_seed)
        rand_seeds = random.sample(range(1_000_000), top_k)
        start_rand = time.time()
        for seed in tqdm(rand_seeds, desc="Rand50"):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe_sd(
                prompt=prompt,
                token_indices=sum(token_idx_dict.values(), []),
                guidance_scale=guidance_scale_1,
                generator=gen,
                num_inference_steps=num_infer_steps,
                result_root=None,
                seed=seed,
            ).images[0]
            img.save(os.path.join(ae_dir, f"img_rand{seed}.jpg"))
        end_rand = time.time()
        random50_times.append(end_rand - start_rand)

        print(f"✅ Prompt {p_idx:03d} finished.")



    # === Final summary ===
    avg_score_top50_time = sum(scoring_plus_top50_times) / len(scoring_plus_top50_times)
    avg_rand50_time = sum(random50_times) / len(random50_times)
    print(f"📘 Scoring + top50 generation: {avg_score_top50_time:.2f} seconds/prompt")
    print(f"📕 Random50 generation:        {avg_rand50_time:.2f} seconds/prompt")



prompts_dict = {
    1:  "a elephant and a rabbit",
    2:  "a dog and a frog",
    3:  "a bird and a mouse",
    4:  "a monkey and a frog",
    5:  "a horse and a monkey",
    6:  "a bird and a turtle",
    7:  "a bird and a lion",
    8:  "a lion and a monkey",
    9:  "a horse and a turtle",
    10: "a bird and a monkey",
    11: "a bear and a frog",
    12: "a bear and a turtle",
    13: "a dog and a elephant",
    14: "a dog and a horse",
    15: "a turtle and a mouse",
    16: "a cat and a turtle",
    17: "a dog and a mouse",
    18: "a cat and a elephant",
    19: "a cat and a bird",
    20: "a dog and a monkey",
    21: "a lion and a mouse",
    22: "a bear and a lion",
    23: "a bird and a elephant",
    24: "a lion and a turtle",
    25: "a dog and a bird",
    26: "a bird and a rabbit",
    27: "a elephant and a turtle",
    28: "a lion and a elephant",
    29: "a cat and a rabbit",
    30: "a dog and a bear",
    31: "a dog and a rabbit",
    32: "a cat and a bear",
    33: "a bird and a horse",
    34: "a rabbit and a mouse",
    35: "a bird and a bear",
    36: "a bear and a monkey",
    37: "a horse and a frog",
    38: "a cat and a horse",
    39: "a frog and a rabbit",
    40: "a bear and a mouse",
    41: "a monkey and a rabbit",
    42: "a cat and a dog",
    43: "a lion and a frog",
    44: "a frog and a mouse",
    45: "a dog and a lion",
    46: "a lion and a rabbit",
    47: "a elephant and a frog",
    48: "a frog and a turtle",
    49: "a cat and a lion",
    50: "a horse and a rabbit",
    51: "a cat and a monkey",
    52: "a bear and a rabbit",
    53: "a turtle and a rabbit",
    54: "a elephant and a monkey",
    55: "a bird and a frog",
    56: "a lion and a horse",
    57: "a bear and a horse",
    58: "a bear and a elephant",
    59: "a horse and a mouse",
    60: "a dog and a turtle",
    61: "a monkey and a mouse",
    62: "a cat and a frog",
    63: "a monkey and a turtle",
    64: "a horse and a elephant",
    65: "a cat and a mouse",
    66: "a elephant and a mouse",
    67: "a horse with a glasses",
    68: "a bear with a glasses",
    69: "a monkey and a red car",
    70: "a elephant with a bow",
    71: "a frog and a purple balloon",
    72: "a mouse with a bow",
    73: "a bird with a crown",
    74: "a turtle and a yellow bowl",
    75: "a rabbit and a gray chair",
    76: "a dog and a black apple",
    77: "a rabbit and a white bench",
    78: "a lion and a yellow clock",
    79: "a turtle and a gray backpack",
    80: "a elephant and a green balloon",
    81: "a monkey and a orange apple",
    82: "a lion and a red car",
    83: "a lion with a crown",
    84: "a bird and a purple bench",
    85: "a rabbit and a orange backpack",
    86: "a rabbit and a orange apple",
    87: "a monkey and a green bowl",
    88: "a frog and a red suitcase",
    89: "a monkey and a green balloon",
    90: "a cat with a glasses",
    91: "a bear and a blue clock",
    92: "a cat and a gray bench",
    93: "a bear with a crown",
    94: "a lion with a bow",
    95: "a bear and a red balloon",
    96: "a bird and a black backpack",
    97: "a horse and a pink balloon",
    98: "a turtle and a yellow car",
    99: "a lion with a glasses",
    100: "a cat and a yellow balloon",
    101: "a horse and a yellow clock",
    102: "a dog with a glasses",
    103: "a horse and a blue backpack",
    104: "a frog with a bow",
    105: "a elephant with a glasses",
    106: "a mouse and a red bench",
    107: "a bird and a brown balloon",
    108: "a monkey and a yellow backpack",
    109: "a turtle and a pink balloon",
    110: "a cat and a red apple",
    111: "a monkey and a brown bench",
    112: "a rabbit with a glasses",
    113: "a bear and a gray bench",
    114: "a turtle and a blue clock",
    115: "a monkey and a blue chair",
    116: "a turtle and a blue chair",
    117: "a dog with a bow",
    118: "a elephant and a black chair",
    119: "a mouse and a purple chair",
    120: "a bear and a white car",
    121: "a lion and a black backpack",
    122: "a dog with a crown",
    123: "a horse and a green apple",
    124: "a dog and a gray clock",
    125: "a dog and a purple car",
    126: "a dog and a gray bowl",
    127: "a monkey with a bow",
    128: "a mouse and a blue clock",
    129: "a bird and a black bowl",
    130: "a horse and a white car",
    131: "a mouse and a pink apple",
    132: "a bear and a orange backpack",
    133: "a elephant and a yellow clock",
    134: "a bird and a green chair",
    135: "a mouse and a black balloon",
    136: "a turtle and a white bench",
    137: "a bird with a bow",
    138: "a turtle with a crown",
    139: "a bird and a yellow car",
    140: "a frog and a orange car",
    141: "a dog and a pink bench",
    142: "a frog with a crown",
    143: "a frog and a green bowl",
    144: "a frog and a pink bench",
    145: "a horse with a bow",
    146: "a bird and a yellow apple",
    147: "a monkey with a crown",
    148: "a cat and a blue backpack",
    149: "a turtle and a pink apple",
    150: "a dog and a orange chair",
    151: "a horse and a green suitcase",
    152: "a elephant with a crown",
    153: "a monkey and a orange suitcase",
    154: "a turtle and a orange suitcase",
    155: "a lion and a gray apple",
    156: "a mouse with a crown",
    157: "a mouse with a glasses",
    158: "a horse and a brown bowl",
    159: "a monkey and a yellow clock",
    160: "a turtle with a bow",
    161: "a dog and a brown backpack",
    162: "a cat and a purple bowl",
    163: "a lion and a white bench",
    164: "a rabbit and a blue bowl",
    165: "a lion and a brown balloon",
    166: "a horse and a pink chair",
    167: "a elephant and a green bench",
    168: "a rabbit and a white balloon",
    169: "a elephant and a pink backpack",
    170: "a lion and a orange suitcase",
    171: "a elephant and a orange apple",
    172: "a elephant and a green suitcase",
    173: "a horse with a crown",
    174: "a bear with a bow",
    175: "a rabbit and a yellow suitcase",
    176: "a horse and a blue bench",
    177: "a dog and a green suitcase",
    178: "a mouse and a red car",
    179: "a cat and a black chair",
    180: "a bear and a red suitcase",
    181: "a rabbit and a gray clock",
    182: "a bear and a pink apple",
    183: "a lion and a white chair",
    184: "a rabbit with a crown",
    185: "a mouse and a purple bowl",
    186: "a frog and a black apple",
    187: "a rabbit with a bow",
    188: "a mouse and a pink suitcase",
    189: "a lion and a pink bowl",
    190: "a frog and a black chair",
    191: "a frog and a green clock",
    192: "a bear and a white chair",
    193: "a elephant and a brown car",
    194: "a turtle with a glasses",
    195: "a cat and a black suitcase",
    196: "a cat and a yellow car",
    197: "a frog and a yellow backpack",
    198: "a bird and a black suitcase",
    199: "a cat with a crown",
    200: "a rabbit and a yellow car",
    201: "a cat with a bow",
    202: "a bird and a white clock",
    203: "a cat and a green clock",
    204: "a bear and a purple bowl",
    205: "a monkey with a glasses",
    206: "a frog with a glasses",
    207: "a elephant and a green bowl",
    208: "a bird with a glasses",
    209: "a dog and a blue balloon",
    210: "a mouse and a brown backpack",
    211: "a pink crown and a purple bow",
    212: "a blue clock and a blue apple",
    213: "a blue balloon and a orange bench",
    214: "a pink crown and a red chair",
    215: "a orange chair and a blue clock",
    216: "a purple bowl and a black bench",
    217: "a green glasses and a black crown",
    218: "a purple chair and a red bow",
    219: "a yellow glasses and a black car",
    220: "a orange backpack and a purple car",
    221: "a white balloon and a white apple",
    222: "a brown suitcase and a black clock",
    223: "a yellow backpack and a purple chair",
    224: "a gray backpack and a green clock",
    225: "a blue crown and a red balloon",
    226: "a gray suitcase and a black bowl",
    227: "a brown balloon and a pink car",
    228: "a black backpack and a green bow",
    229: "a blue balloon and a blue bow",
    230: "a white bow and a white car",
    231: "a orange bowl and a purple apple",
    232: "a brown chair and a white bench",
    233: "a purple crown and a blue suitcase",
    234: "a yellow bow and a orange bench",
    235: "a yellow glasses and a brown bow",
    236: "a red glasses and a red suitcase",
    237: "a pink bow and a gray apple",
    238: "a gray crown and a white clock",
    239: "a black car and a white clock",
    240: "a brown bowl and a green clock",
    241: "a green backpack and a yellow crown",
    242: "a orange glasses and a pink clock",
    243: "a purple chair and a orange bowl",
    244: "a orange suitcase and a brown bench",
    245: "a white glasses and a orange balloon",
    246: "a yellow backpack and a gray apple",
    247: "a green bench and a red apple",
    248: "a gray backpack and a yellow glasses",
    249: "a green glasses and a yellow chair",
    250: "a white glasses and a gray apple",
    251: "a gray suitcase and a brown bow",
    252: "a white car and a black bowl",
    253: "a purple car and a pink apple",
    254: "a gray crown and a purple apple",
    255: "a orange car and a red bench",
    256: "a red suitcase and a blue apple",
    257: "a red backpack and a yellow bowl",
    258: "a red bench and a yellow clock",
    259: "a black backpack and a pink balloon",
    260: "a blue suitcase and a gray balloon",
    261: "a yellow glasses and a gray bowl",
    262: "a white suitcase and a white chair",
    263: "a purple crown and a blue bench",
    264: "a yellow bow and a pink bowl",
    265: "a green backpack and a brown suitcase",
    266: "a green glasses and a black bench",
    267: "a white bow and a black clock",
    268: "a red crown and a black bowl",
    269: "a green chair and a purple car",
    270: "a white chair and a gray balloon",
    271: "a pink chair and a gray apple",
    272: "a yellow suitcase and a yellow car",
    273: "a green backpack and a purple bench",
    274: "a black crown and a red car",
    275: "a green balloon and a pink bowl",
    276: "a purple balloon and a white clock"
}





if __name__ == '__main__':
    # ===== Step 0: 读取 JSON =====
    with open("prompt_token_index_dict.json", "r") as f:
        prompt_token_index_dict = {int(k): v for k, v in json.load(f).items()}
    main()
