# ReAct Mini-Project Experiment

This folder contains a small, reproducible experiment for demonstrating the ReAct workflow on a small sample from HotpotQA.

The goal is not to fully reproduce the original paper's benchmark pipeline. Instead, this setup gives you a scaled-down workflow demo that is easy to explain in a class report:

- `react`: iterative reasoning plus `Search[...]` and `Lookup[...]` actions over Wikipedia

## What this measures

For each question, the script records:

- predicted answer
- gold answer
- exact match after normalization
- the full reasoning trace
- tool calls and observations for the ReAct agent

It writes a JSONL log for every example and a JSON summary with aggregate accuracy.

## Files

- `run_experiment.py`: main runner
- `prompts.py`: prompt templates for CoT and ReAct
- `requirements.txt`: Python dependencies
- `.env.example`: environment variables

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set your Gemini API key.

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

Optionally copy `.env.example` into `.env` and load it with your preferred workflow.

If Wikipedia blocks search requests on your network, set a descriptive user agent:

```powershell
$env:WIKIPEDIA_USER_AGENT="react-mini-project/1.0 (academic project; contact: your_email@example.com)"
```

## Example run

```powershell
python run_experiment.py --model gemini-2.5-flash --sample-size 5 --max-react-steps 10
```

For Gemini free tier, start smaller to avoid request-per-minute limits:

```powershell
python run_experiment.py --model gemini-2.5-flash --sample-size 1 --max-react-steps 10 --sleep-seconds 12
```

## Outputs

Results are stored under `outputs/`:

- `summary_*.json`
- `traces_*.jsonl`

A curated example run is also included under `sample_results/` so the repository contains one concrete result without checking in every local run artifact.

## Notes for the report

This experiment is intentionally lightweight. It captures the main idea of ReAct from Yao et al. (2023): alternating reasoning with environment interaction can improve performance on multi-hop questions by letting the model gather evidence before committing to an answer.

This experiment now uses Google's native Gemini Python SDK directly, so authentication and error messages come from Gemini itself rather than an OpenAI-compatible wrapper.

The retrieval layer now handles blocked Wikipedia requests more gracefully. If a search or lookup fails, the run will continue and log the failure as part of the trace instead of crashing immediately.

Gemini free-tier quotas can also trigger `429` responses. The runner now retries automatically with backoff, but using a smaller sample size and a nonzero `--sleep-seconds` value is still the safest way to get a clean first run.

Good report angles:

- walk through one complete `Thought -> Action -> Observation -> ... -> Answer` trace
- show how retrieval changes the next action
- discuss cases where the agent wastes steps or searches poorly
- comment on the extra cost and latency introduced by tool use
