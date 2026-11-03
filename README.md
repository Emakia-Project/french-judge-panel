# french-judge-panel

A panel of five large language models from five providers (OpenAI, Google Gemini, xAI Grok, Mistral and Anthropic Claude) relabels published French harassment datasets. Every vote and every abstention is kept, not just the final label. Small on-device CoreML classifiers are then trained on three label sets (LLM0, LLM3, LLM4) built from the same texts, to measure what the relabeling changes. The repository also includes the stratified X collection and the evaluation code.

Developed by [Emakia](https://emakia.org), a US nonprofit building open-source tools against online harassment of public figures and civic participants.

Developed by [Emakia](https://emakia.org), a US nonprofit building open-source tools against online harassment of public figures and civic participants.

## What it does

1. **Collects French posts from X** in separate streams (reply-directed, topic, lexicon-seeded, neutral baseline, account-anchored), so prevalence is a design choice and not an accident of the query.
2. **Relabels a merged French corpus** with five independent judges: OpenAI, Google Gemini, xAI Grok, Mistral and Anthropic Claude. Each run records the exact model version that answered.
3. **Builds three label sets** from the same texts:
   - **LLM0**: the original published labels
   - **LLM3**: Harassment if at least 3 of 5 judges say so
   - **LLM4**: Harassment if at least 4 of 5 judges say so
4. **Trains one CoreML text classifier per label set** on a Mac, so the label set is the only variable.
5. **Evaluates** the classifiers against the panel on out-of-domain posts, per stream and per vote band, under two criteria: *contains* harassment (v1) and *targets* a person or group (v2).

## Design rules

- **An empty or refused answer is ABSTAIN, never a label.** Abstentions are excluded from the vote, with the provider's reason recorded.
- **Models are pinned**, and the returned model version is checked on every call.
- **Verdicts are read from the front of the answer.** Explanations are never searched for label words (see `reparse.py`).
- **Every run is append-only** (`runs/<run_id>/results.jsonl`) and resumable. A cost estimate and a 10-text preflight per judge run before any paid call.
- **Raw X pages are stored before any database load**, so a failure never requires a second paid call.
- **Media bytes are archived at capture time**, because X media links expire.
- **Vote counts are kept**, not just the final label. Items between the unanimous extremes are reported by their vote count.

## Folder map

| Folder | Files | Purpose |
|---|---|---|
| `collect-French-data/` | `collect.py`, `queries_loader.py`, `collection_queries.yaml`, `filters.yaml`, `download_media.py` | X API v2 collection with a budget cap, stream labels and media archiving |
| `French-filteering-buildLLM3and LLM4/` | `label_corpus_training.py`, `consensus.py`, `run_store.py` | Five-judge relabeling of the training corpus, then the LLM0/LLM3/LLM4 label sets |
| | `reparse.py`, `redo_missing.py`, `patch_judges.py`, `patch_evalllm.py`, `label_corpus.py` | Repair tools: re-parse stored answers, re-call missing pairs, record refusals |
| | `duplicate_audit.py`, `one_judge.py`, `v2_followups.py` | Analyses: human-vs-human disagreement on duplicate texts, single-judge swaps, v2 criterion follow-ups |
| `eval-llm/` | `label_corpus_eval.py`, `run_store.py` | Five-judge verdicts on the out-of-domain X posts |
| | `vision_media.py` | Gemini Vision verdicts on archived media, blind to the post text by default |
| | `merge_posts.py`, `stream_tables.py`, `combined_panel.py`, `bias_and_construct.py`, `hard_case_signals.py` | Tables and metrics per stream and vote band; construct mismatch and per-group flag rates |
| `apple-development/` | `train_classifiers.swift` | Trains the LLM0/LLM3/LLM4 CoreML classifiers (CreateML, macOS) |
| `llm-wrapper/` | provider wrappers | Earlier wrappers, kept for reference. They map failed calls to a label, so the pipeline above does not use them |

## Setup

Requires Python 3.10 or later, and macOS with Xcode for the CoreML step.

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy pyyaml httpx python-dotenv
```

Create a `.env` file in the folder you run from. Never commit it.

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...
MISTRAL_API_KEY=...
ANTHROPIC_API_KEY=...
X_BEARER_TOKEN=...          # only for collection
GOOGLE_CLOUD_PROJECT=...    # only for BigQuery
MEDIA_STORE=gs://YOUR-BUCKET/media/fr/   # only for media archiving
```

## Typical run

```
# 1. cost estimate and preflight, no paid calls
python "French-filteering-buildLLM3and LLM4/label_corpus_training.py" --corpus data/french_corpus_merged.csv --estimate

# 2. relabel with the five judges (prints a run_id)
python "French-filteering-buildLLM3and LLM4/label_corpus_training.py" --corpus data/french_corpus_merged.csv

# 3. label sets, no API calls
python "French-filteering-buildLLM3and LLM4/consensus.py" --run RUN_ID

# 4. train on the Mac, once per label set (LLM0, LLM3, LLM4)
swift apple-development/train_classifiers.swift
```

Each script's header describes its options and the reason for each setting.

## Data

This repository contains code only. Post text from X is not redistributed, in line with X's terms. The training corpus merges published French datasets. Obtain each from its authors under its own license. The released labels contain IDs, labels and vote counts.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the accompanying paper (details to be added after publication).
