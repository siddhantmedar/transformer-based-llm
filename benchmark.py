"""
Benchmark script to compare inference speed with and without KV caching.
Saves snapshots of generated text and timing results.
"""
import torch
import time
import json
from dataclasses import dataclass
from tokenizers import Tokenizer
from gpt import DecoderBlock, generate, Config as GPTConfig, device, encode, decode

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

@dataclass
class Config:
    n_layers: int = 6
    n_vocab: int = vocab_size
    d_model: int = 128
    num_head: int = 4
    max_seq_length: int = 512
    block_size: int = 256

config = Config()

# Load checkpoint
print(f"Loading model checkpoint...")
checkpoint = torch.load("model_checkpoints/best_model.pt", map_location=device)

# Create model
model = DecoderBlock(
    config.n_layers, config.n_vocab, config.d_model, config.num_head, config.max_seq_length, dropout=0.1
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Test prompts
test_cases = [
    {"prompt": "Music", "tokens": 50},
    {"prompt": "Music", "tokens": 100},
    {"prompt": "Music", "tokens": 200},
    {"prompt": "To be or not to be", "tokens": 50},
    {"prompt": "To be or not to be", "tokens": 100},
    {"prompt": "To be or not to be", "tokens": 200},
]

results = {
    "device": str(device),
    "model_params": sum(p.numel() for p in model.parameters()),
    "config": {
        "n_layers": config.n_layers,
        "d_model": config.d_model,
        "num_head": config.num_head,
        "block_size": config.block_size
    },
    "benchmarks": []
}

snapshots = []

print("="*80)
print("BENCHMARK: KV CACHE vs STANDARD GENERATION")
print("="*80)

for idx, test_case in enumerate(test_cases, 1):
    prompt = test_case["prompt"]
    max_tokens = test_case["tokens"]

    print(f"\n[{idx}/{len(test_cases)}] Prompt: '{prompt}' | Generating {max_tokens} tokens")
    print("-"*80)

    # Benchmark WITHOUT KV cache
    print("  Running WITHOUT KV cache...")
    start = time.time()
    output_no_cache = generate(model, prompt, max_tokens=max_tokens, temperature=1.0, use_kv_cache=False)
    time_no_cache = time.time() - start

    # Benchmark WITH KV cache
    print("  Running WITH KV cache...")
    start = time.time()
    output_with_cache = generate(model, prompt, max_tokens=max_tokens, temperature=1.0, use_kv_cache=True)
    time_with_cache = time.time() - start

    # Calculate speedup
    speedup = time_no_cache / time_with_cache

    # Print results
    print(f"  ✓ No Cache:   {time_no_cache:.4f}s")
    print(f"  ✓ With Cache: {time_with_cache:.4f}s")
    print(f"  ✓ Speedup:    {speedup:.2f}x")
    print(f"  ✓ Time saved: {(time_no_cache - time_with_cache):.4f}s ({((1 - time_with_cache/time_no_cache)*100):.1f}%)")

    # Store benchmark data
    benchmark = {
        "test_id": idx,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "time_no_cache": round(time_no_cache, 4),
        "time_with_cache": round(time_with_cache, 4),
        "speedup": round(speedup, 2),
        "time_saved_seconds": round(time_no_cache - time_with_cache, 4),
        "time_saved_percent": round((1 - time_with_cache/time_no_cache)*100, 2)
    }
    results["benchmarks"].append(benchmark)

    # Store snapshot
    snapshot = {
        "test_id": idx,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "output_no_cache": output_no_cache,
        "output_with_cache": output_with_cache,
        "are_identical": output_no_cache == output_with_cache  # Note: outputs will differ due to randomness
    }
    snapshots.append(snapshot)

# Calculate statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

speedups = [b["speedup"] for b in results["benchmarks"]]
avg_speedup = sum(speedups) / len(speedups)
min_speedup = min(speedups)
max_speedup = max(speedups)

total_time_no_cache = sum(b["time_no_cache"] for b in results["benchmarks"])
total_time_with_cache = sum(b["time_with_cache"] for b in results["benchmarks"])
total_time_saved = total_time_no_cache - total_time_with_cache

print(f"Average Speedup:    {avg_speedup:.2f}x")
print(f"Min Speedup:        {min_speedup:.2f}x")
print(f"Max Speedup:        {max_speedup:.2f}x")
print(f"Total Time (No Cache):   {total_time_no_cache:.4f}s")
print(f"Total Time (With Cache): {total_time_with_cache:.4f}s")
print(f"Total Time Saved:        {total_time_saved:.4f}s ({(total_time_saved/total_time_no_cache*100):.1f}%)")

# Add summary to results
results["summary"] = {
    "avg_speedup": round(avg_speedup, 2),
    "min_speedup": round(min_speedup, 2),
    "max_speedup": round(max_speedup, 2),
    "total_time_no_cache": round(total_time_no_cache, 4),
    "total_time_with_cache": round(total_time_with_cache, 4),
    "total_time_saved": round(total_time_saved, 4),
    "time_saved_percent": round(total_time_saved/total_time_no_cache*100, 2)
}

# Speedup by token count
print("\n" + "="*80)
print("SPEEDUP BY TOKEN COUNT")
print("="*80)
token_counts = sorted(set(b["max_tokens"] for b in results["benchmarks"]))
for tokens in token_counts:
    token_benchmarks = [b for b in results["benchmarks"] if b["max_tokens"] == tokens]
    avg_speedup_for_tokens = sum(b["speedup"] for b in token_benchmarks) / len(token_benchmarks)
    print(f"{tokens} tokens: {avg_speedup_for_tokens:.2f}x average speedup")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✓ Saved benchmark_results.json")

with open("benchmark_snapshots.json", "w") as f:
    json.dump(snapshots, f, indent=2)
print("✓ Saved benchmark_snapshots.json")

# Create human-readable report
with open("benchmark_report.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("KV CACHE BENCHMARK REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Device: {device}\n")
    f.write(f"Model Parameters: {results['model_params']:,}\n")
    f.write(f"Layers: {config.n_layers}\n")
    f.write(f"Model Dimension: {config.d_model}\n")
    f.write(f"Attention Heads: {config.num_head}\n\n")

    f.write("="*80 + "\n")
    f.write("INDIVIDUAL BENCHMARKS\n")
    f.write("="*80 + "\n\n")

    for i, (benchmark, snapshot) in enumerate(zip(results["benchmarks"], snapshots), 1):
        f.write(f"Test {i}: '{benchmark['prompt']}' ({benchmark['max_tokens']} tokens)\n")
        f.write(f"  Time (No Cache):   {benchmark['time_no_cache']:.4f}s\n")
        f.write(f"  Time (With Cache): {benchmark['time_with_cache']:.4f}s\n")
        f.write(f"  Speedup: {benchmark['speedup']:.2f}x\n")
        f.write(f"  Time Saved: {benchmark['time_saved_seconds']:.4f}s ({benchmark['time_saved_percent']:.1f}%)\n\n")

        f.write(f"  Output (No Cache, first 200 chars):\n")
        f.write(f"    {snapshot['output_no_cache'][:200]}\n\n")

        f.write(f"  Output (With Cache, first 200 chars):\n")
        f.write(f"    {snapshot['output_with_cache'][:200]}\n\n")

        f.write("-"*80 + "\n\n")

    f.write("="*80 + "\n")
    f.write("SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Average Speedup: {results['summary']['avg_speedup']:.2f}x\n")
    f.write(f"Min Speedup: {results['summary']['min_speedup']:.2f}x\n")
    f.write(f"Max Speedup: {results['summary']['max_speedup']:.2f}x\n")
    f.write(f"Total Time (No Cache): {results['summary']['total_time_no_cache']:.4f}s\n")
    f.write(f"Total Time (With Cache): {results['summary']['total_time_with_cache']:.4f}s\n")
    f.write(f"Total Time Saved: {results['summary']['total_time_saved']:.4f}s\n")
    f.write(f"Percentage Saved: {results['summary']['time_saved_percent']:.2f}%\n")

print("✓ Saved benchmark_report.txt")

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  • benchmark_results.json    - JSON format benchmark data")
print("  • benchmark_snapshots.json  - Generated text samples")
print("  • benchmark_report.txt      - Human-readable report")
