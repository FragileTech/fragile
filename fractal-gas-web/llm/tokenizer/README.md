# Qwen scoring tokenizer

Source: Qwen/Qwen3.5-9B, Hugging Face revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
Tokenizer files and LICENSE are copied unchanged from that revision.
The browser build copies these into the local vendor bundle. XED alone loads them.
Together echoes prompt probabilities without token IDs; local tokenization must match
all echoed prompt token strings (including their lossy Unicode rendering) and the billed
prompt token count before its probabilities can be used.
