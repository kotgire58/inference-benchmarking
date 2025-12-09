import yaml
import torch
from models.load_model import load_model
from benchmarks.baseline import run_baseline

# NEW import
from optimizations.kv_cache_master import generate_with_kv_cache


def run_experiment(name, optimizations, model, tokenizer, config):
    prompt = config["prompt"]
    max_new_tokens = config["max_new_tokens"]

    if "kv_cache_small" in optimizations:
        result = generate_with_kv_cache(
            model, tokenizer, prompt, max_new_tokens,
            cache_mode="small"
        )
        return result["output"], result["total_time"]

    if "kv_cache_optimal" in optimizations:
        result = generate_with_kv_cache(
            model, tokenizer, prompt, max_new_tokens,
            cache_mode="optimal"
        )
        return result["output"], result["total_time"]

    if "kv_cache_large" in optimizations:
        result = generate_with_kv_cache(
            model, tokenizer, prompt, max_new_tokens,
            cache_mode="large"
        )
        return result["output"], result["total_time"]

    # default baseline
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
