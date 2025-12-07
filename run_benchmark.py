import yaml
import torch
from models.load_model import load_model
from benchmarks.baseline import run_baseline
from optimizations.kv_cache import apply_kv_cache
from optimizations.speculative_decoding import speculative_decoding
from optimizations.batching import run_batching

def run_experiment(name, optimizations, model, tokenizer, config):
    prompt = config["prompt"]
    max_new_tokens = config["max_new_tokens"]
    batch_size = config["batch_size"]

    if "kv_cache" in optimizations:
        return apply_kv_cache(model, tokenizer, prompt, max_new_tokens)

    if "speculative_decoding" in optimizations:
        return speculative_decoding(model, tokenizer, prompt, max_new_tokens)

    if "batching" in optimizations:
        return run_batching(model, tokenizer, prompt, batch_size, max_new_tokens)

    return run_baseline(model, tokenizer, prompt, max_new_tokens)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model(config["model_name"], config["device"])

    for bench in config["benchmarks"]:
        name = bench["name"]
        optimizations = bench["optimizations"]

        print(f"\nRunning benchmark: {name}")
        
        results = []
        for i in range(config["num_runs"]):
            output, time_taken = run_experiment(
            name,
            optimizations,
            model,
            tokenizer,
            config
        )

        print(f"\n--- {name.upper()} Run {i+1} ---")
        print(f"Time: {time_taken:.4f} sec")
        print("Output:")
        print(output)
        print("-----------------------------------")
        
        results.append(time_taken)



        avg_time = sum(results) / len(results)
        print(f"\n>>> {name.upper()} AVG LATENCY: {avg_time:.4f} sec")

if __name__ == "__main__":
    main()
