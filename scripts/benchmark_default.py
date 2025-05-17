import time
# import torch
# import wandb
from vllm import LLM, SamplingParams

# Prompts to test
prompts = [
    "The history of artificial intelligence began in",
    "Once upon a time in a distant galaxy,",
    "Explain the principle of reinforcement learning",
]

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=10,
    max_tokens=64,
    use_beam_search=False,
    logprobs=1,
)

# Init model (uses CUDA if available)
llm = LLM(model="gpt2-medium", draft_model="gpt2")

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

# Log output and metrics
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Generated: {output.outputs[0].text}")
    # wandb.log({f"sample_{i}/output": output.outputs[0].text})

# wandb.log({
#     "runtime_secs": end_time - start_time,
#     "num_prompts": len(prompts),
#     "tokens_per_second": sum(len(o.outputs[0].token_ids) for o in outputs) / (end_time - start_time)
# })