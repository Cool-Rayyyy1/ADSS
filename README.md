# ADSS (anonymous repo)

Minimal scripts and configs to run the **ADSS** experiments.

---

## Environment

Create and activate the Conda environment (env name: `adss`) from the provided YAML:

```bash
conda env create -f environment.yml
conda activate adss
```

> Most dependencies are installed via `pip` inside the environment. If you plan to use GPU, ensure your CUDA/driver matches your PyTorch build.

---

## Data & Config

- **Dataset**: The script automatically loads the `initno` dataset (no manual path needed).
- **Body token**: Already defined in a JSON file; it is read directly by the script.

---

## Run

Single entry point:

```bash
python run_sd_adss.py
```

**Defaults:**
- `k = 50`
- Auto-load `initno` dataset
- Body token read from JSON

If the script exposes flags, you can override defaults, e.g.:

```bash
python run_sd_adss.py --k 50
```

---

## Outputs

Each prompt is split into two categories:

```
results/
  test1/
    adss/
      img_adss_.png/
        ...generated samples...
      img_rand_.png/
        ...generated samples...
```

---

## Notes

- For reproducibility, keep the same `environment.yml` and (if available) set a fixed `--seed`.
- GPU is optional; for faster runs ensure CUDA is available and compatible.
- To use custom prompts or data, follow the JSON/dir structure expected by `run_sd_adss.py`.
