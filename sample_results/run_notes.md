# Run Summary: 2026-04-19

Model: `gemini-2.5-flash-lite`

Command:

```powershell
python run_experiment.py --model gemini-2.5-flash-lite --sample-size 3 --max-react-steps 10 --sleep-seconds 12
```

Aggregate result:

- examples: 3
- exact-match accuracy: 1/3 = 33.3%

Case summary:

1. `8th Military Police Brigade` question
- Outcome: semantically close but exact-match incorrect
- Gold: `Honolulu`
- Prediction: `Honolulu County`
- Takeaway: the workflow found the right supporting evidence but did not align exactly to the benchmark label.

2. `2013 America East Men's Lacrosse Tournament` question
- Outcome: incorrect
- Gold: `in 2000`
- Prediction: `2012`
- Takeaway: the agent entered an unproductive search loop and failed to ground the stadium correctly. This is a good example of ReAct's sensitivity to poor retrieval trajectories.

3. `Alexandra David-Neel and Jack Kerouac` question
- Outcome: correct
- Gold: `writer`
- Prediction: `Writer`
- Takeaway: the agent handled a simpler evidence path well and reached the right profession.

Overall interpretation:

- The run shows both the strength and the weakness of ReAct.
- Strength: when retrieval is informative, the agent can gather evidence and synthesize a reasonable answer.
- Weakness: when early search results are noisy, the agent may loop, over-search, or drift away from the actual task.
