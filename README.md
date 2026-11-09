# PIPA — Personalized Inference for Proactive Agents

Inference-time personalization for a proactive assistant: reconstruct a bounded
user model from behavioural history, maintain a calibrated belief over how
receptive that user is, and decide whether to **act**, **ask**, or **stay
silent**. No per-user training; one frozen backbone.

This repository is the code for that pipeline. It contains no results, no
figures, and no data — the corpora the work was developed on are logs of real
people's screens and chats and are not redistributable. A synthetic corpus
generator is included so everything here runs end to end.

---

## Quick start

```bash
pip install -r requirements.txt
python pipa_dataset/scripts/make_synthetic.py      # writes a 12-user fixture
cd pipa
python run.py --task timing_annotated --split test --limit 20 \
              --conditions b0_none,b5_reconstructed,b2_retrieval
```

That runs **dry** — prompts are built and scored with a stub, no API calls, no
cost. It writes `runs/<run-id>/` with `predictions.jsonl`, `summary.json` and
`sample_prompts.txt`. **Read `sample_prompts.txt` before going live**; it is
the exact text the model will see.

To run for real, add a key and `--live`:

```bash
cp ../.env.example ../.env     # fill in one provider
python run.py --task timing_annotated --split test --limit 200 \
              --conditions b0_none,b5_reconstructed --model qwen-plus \
              --live --budget 2
```

`--live` is never the default and `--budget` is enforced: the engine raises
`BudgetExceeded` rather than quietly spending more.

---

## Layout

```
pipa/
  engine.py        LLM client: providers, pricing, on-disk response cache, budget
  tasks.py         corpora -> TaskInstance objects, splits, per-user history
  memory.py        how each condition renders history into a prompt
  profile.py       profile reconstruction (the PIPA user model)
  rules.py         population-relative cue extraction with shrinkage
  bayes.py         Beta-Binomial belief over receptivity; act/ask/silent policy
  agent.py         scoring agent: prompt -> probability
  baselines.py     RAG, KG-memory, MemGPT-style summary, generative-agents, ...
  react.py         ReAct-style tool loop
  metrics.py       AUROC, AUPRC, F1, FAR, calibration, bootstrap CIs

  run.py                    main driver: one task x many conditions
  profile_unsupervised.py   does the profile describe THIS user? (donor control)
  bayes_experiments.py      posterior calibration, convergence, PIPA vs PIPA-Mean
  bayes_table.py            uncertainty ablation on held-out decisions
  action_table.py           where decisions go: silent / ask / act
  run_simulation.py         generate behaviour from a profile alone
  score_simulation.py       does that behaviour identify its own user?
  score_suggestions.py      embedding store used by the simulation scoring

pipa_dataset/
  scripts/load.py           dataset loader (the contract the data must satisfy)
  scripts/common.py         paths and IO helpers
  scripts/make_synthetic.py generates a runnable fixture
```

Modules import each other flatly (`import tasks`), so run the entry points from
inside `pipa/`.

---

## The pipeline

```
behavioural history
   |
   |-- profile.py ----------> bounded natural-language user model
   |-- rules.py ------------> population-relative cues, shrunk by support
   |-- bayes.py ------------> Beta posterior over receptivity theta_u
   |
   v
memory.py renders the chosen condition into a prompt
   |
   v
agent.py + engine.py --> p(user wants help | context)
   |
   v
decision:  act / ask / silent     (bayes.decide, thresholds derived from costs)
   |
   v
metrics.py --> AUROC, AUPRC, F1, FAR, ECE, bootstrap CIs
```

The two halves are separable on purpose, because they behave differently: the
profile can be consumed **as prose in a prompt** (`b5_reconstructed`) or **as a
posterior in a scoring layer** (`bayes.py`). They carry the same information
and do not perform the same.

---

## Entry points

| command | question it answers |
|---|---|
| `python run.py --task timing_annotated --conditions ...` | how well does each memory condition predict the need for help? |
| `python profile_unsupervised.py --task timing_annotated --live` | does the profile describe *this* user, or any plausible user? |
| `python bayes_experiments.py` | is the posterior calibrated, does it concentrate, does the spread change decisions? (synthetic + real; **no API needed**) |
| `python bayes_table.py` | F1 / FAR / Acc / utility for PIPA vs PIPA-Mean vs κ=0 |
| `python action_table.py` | the action distribution, with an act-rate-matched control |
| `python run_simulation.py --live` then `score_simulation.py --run <id>` | can behaviour generated from a profile alone identify its own user? |

`bayes_experiments.py`, `bayes_table.py` and `action_table.py` need **no API
key** — the Bayesian layer runs on labels alone.

### Controls worth knowing about

Three are built in, because personalization claims are easy to make and hard to
support:

- **donor** (`ctrl_other_user_matched`) — the same profile object, but another
  user's. If a result survives this, it is about fluency, not identity.
- **shuffled** (`ctrl_shuffled_matched`) — history in the wrong order.
- **population-only** (`b3_statistical`) — no user-specific evidence at all.

`profile_unsupervised.py` runs the donor control directly: it scores the real
user's profile and a length-matched donor profile on the *same* trials.

---

## Conditions

Passed as `--conditions`, comma-separated:

| id | what the model sees |
|---|---|
| `b0_none` | context only, no memory |
| `b1_raw` | the full recent history (long context) |
| `b2_retrieval` | RAG over history |
| `b3_statistical` | population rate only |
| `b4_kg` | knowledge-graph memory |
| `b5_reconstructed` | **PIPA** — the reconstructed profile |
| `b9_genagents` | generative-agents style reflection |
| `b10_summary` | MemGPT-style rolling summary |
| `b13_rules` | shrunk population-relative cues (binary tasks only) |
| `ctrl_other_user_matched` | donor control |
| `ctrl_shuffled_matched` | shuffled control |

---

## Tasks

| task | label | notes |
|---|---|---|
| `timing_annotated` | human-annotated `need_help` | the clear-signal corpus |
| `timing_production` | did the user engage within 5 min | behavioural proxy, low base rate |
| `intention` | 8-way intention class | positives only |
| `suggestion` | free-text suggestion | needs paired chat turns |

`timing_production` thins to non-overlapping label windows before evaluation:
captions arrive far more often than the 5-minute lookahead, so without thinning
one chat marks a dozen captions positive and "repeat the previous label" wins.
Thinned instances stay in *history* and are dropped only from *evaluation*.

---

## Bringing your own data

Implement the contract in `pipa_dataset/scripts/load.py`. The annotated path
needs four files:

| file | key columns |
|---|---|
| `annotated/samples.parquet` | `sample_id`, `user_id`, `t0`, `app_name`, `window_title`, `context_summary`, `need_help`, `intention` |
| `annotated/history_events.parquet` | `sample_id`, `start_time`, `seconds_before_t0`, `app_name`, `window_title`, `summary` |
| `annotated/context_screens.parquet` | `sample_id`, `screen_order`, `summary` |
| `annotated/splits.json` | `{"temporal": {"train": [...], "val": [...], "test": [...]}}` |

The production path needs `production/texts_gateway.parquet` with `request_id`,
`user_id`, `timestamp`, `family` (`screen_caption` / `foreground_chat`),
`user_text`, `assistant_text`.

`make_synthetic.py` writes exactly this schema; read it as the worked example.

---

## Cost and reproducibility

- **Dry-run by default.** `--live` is required for any API call.
- **Budgets are enforced**, in dollars, per run.
- **Responses are cached** in `pipa/runs/llm_cache.sqlite`, keyed by model and
  prompt, so re-running a completed experiment is free and deterministic.
- Every run writes `summary.json` with the model id, base URL, token counts and
  spend, so a number can always be traced to the configuration that produced it.

One caveat worth stating: several providers are **not deterministic at
temperature 0**. Repeat runs of the same cell can differ by a few points. Where
it matters, report confidence intervals from repeats rather than single runs.

---

## On the synthetic corpus

It exists so the code runs, not so results reproduce. Users are drawn from a
Beta prior with no real structure: a personalised model will beat a population
baseline, because receptivity genuinely varies by user, but the magnitudes are
meaningless. Nothing in a paper should be computed from it.

`suggestion` yields no instances on the fixture — it needs paired chat turns
that the generator does not synthesise. The other three tasks all work.
