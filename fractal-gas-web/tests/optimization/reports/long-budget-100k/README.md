# Superseded partial comparison

This zero-elite experiment was stopped when the user required five elites. Only 354 of 600 unique cases completed. Do not use it as the requested benchmark report. The complete replacement is in `../long-budget-100k-5elites/`.

Validated, deduplicated partial records are in `partial-results.jsonl`. The original `runs.jsonl` contains an interrupted concurrent-writer record and is retained only as a raw execution log. The replacement experiment uses an exclusive output lock and a fresh directory.
