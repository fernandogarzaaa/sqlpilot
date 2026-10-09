Model: `qwen2.5-0.5b-instruct-q4_k_m` · Spider dev sample: 150 questions (seed 42)

| Config | EX % | Correct | Failed to execute % | Mean latency (s) | p50 / p95 (s) | Prompt tok / q | Completion tok / q | LLM calls / q |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 26.7 | 40/150 | 44.0 | 3.84 | 2.61 / 10.68 | 298.8 | 56.4 | 0.99 |
| retrieval | 39.3 | 59/150 | 32.0 | 9.15 | 7.43 / 21.09 | 265.5 | 47.6 | 0.99 |
| retrieval_fix | 40.5 | 51/126 | 27.0 | 15.13 | 8.12 / 44.65 | 470.3 | 82.6 | 1.56 |
