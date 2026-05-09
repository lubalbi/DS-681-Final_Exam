# DS681 Final Exam — REINFORCE Character-Level LM

Final exam submission for DS681 (Deep Learning for Computer Vision, Spring 2026). Implements the REINFORCE policy-gradient algorithm from scratch and applies it to a tiny autoregressive character-level language model that has to learn to emit sequences containing the substring `"abc"`.

**Author:** Lucas Balbi

## Problem

- **Vocabulary:** `<bos>, a, b, c, <eos>` (5 tokens)
- **Policy:** memoryless — conditions only on the previous token
- **Reward:** terminal-only, `R(τ) = 1` if `"abc"` is a substring of the sampled sequence, else `0`
- **Objective:** understand temporal credit assignment — how step-1 decisions get gradient signal from a reward observed only at the end of the trajectory

## What's in here

| Task | What it does |
|------|--------------|
| Task 1 | `CharPolicy` — `nn.Embedding(5, 8)` + `nn.Linear(8, 5)` |
| Task 2 | `sample(...)` — rolls out one trajectory from `<bos>`, returns tokens and per-step log-probs |
| Task 3 | `reward(...)` — substring check for `"abc"` |
| Task 4 | Vanilla REINFORCE training loop, 1500 steps |
| Task 4b | REINFORCE with EMA baseline `b ← 0.9·b + 0.1·R` for variance reduction |
| Task 5 | Inspect learned conditional distributions, sample 10 trajectories, written discussion of credit assignment |

## Results

After 1500 training steps with the EMA baseline:

- **Final 100-step avg reward:** 1.000 (both with and without baseline; seed 0 happened to converge in both runs)
- **Learned conditionals:**
  - `π(a | <bos>) ≈ 0.995`
  - `π(b | a) ≈ 0.999`
  - `π(c | b) ≈ 0.999`
  - `π(<eos> | c) ≈ 0.03` (see note below)
- **Sampled trajectories** all start with `['a', 'b', 'c', ...]` and reach reward 1.

### Note on `π(<eos> | c)`

The expected output suggests `π(<eos> | c) ≈ 1`, but the trained model converges to `≈ 0.03`. This isn't a bug — it's a property of the reward as specified. Once the policy emits `a → b → c`, `R = 1` is already locked in regardless of what tokens come after, so every action at that step has the same advantage and no gradient differentiates between them. The discussion in Task 5c covers this.

## Running it

The notebook is designed to run inside the project's `torch.dev.gpu` (or `torch.dev.cpu`) devcontainer. The model is tiny (~50 parameters), so CPU is more than enough.

```bash
# from inside the devcontainer
make start                    # one-time environment setup
make install-notebooks        # install matplotlib etc.
```

Then open `finalexam_ds681_lucasbalbi.ipynb` and run all cells. Total runtime is under a minute.

## Files

- `finalexam_ds681_lucasbalbi.ipynb` — the full submission
- `README.md` — this file

## Assignment reference

Original spec: <https://aegean.ai/aiml-common/assignments/main/cv-spring-2026/final-exam/reinforce_charlm_assignment/reinforce_charlm_assignment>
