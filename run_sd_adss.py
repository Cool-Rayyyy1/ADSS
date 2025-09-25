import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from adss.pipelines.pipeline_sd_initno_adss import StableDiffusionInitNOPipeline
from adss.pipelines.pipeline_sd import StableDiffusionAttendAndExcitePipeline
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
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32
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
                gen = torch.Generator("cuda").manual_seed(seed)
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
}





if __name__ == '__main__':
    # ===== Step 0: 读取 JSON =====
    with open("prompt_token_index_dict.json", "r") as f:
        prompt_token_index_dict = {int(k): v for k, v in json.load(f).items()}
    main()
