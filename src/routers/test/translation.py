"""
Translation test suite.

Change RUN_TEST below to select which test to run:
  "latency"  — 3-round latency comparison: Realtime vs DeepL full vs DeepL bare
  "quality"  — Side-by-side quality comparison: DeepL full vs GPT Realtime
"""

RUN_TEST = "quality"

import os
import json
import time

from utils.translation_realtime import RealtimeTranslator
from utils.translation_deepl import DeepLTranslator, DEEPL_API_KEY, DEEPL_BASE_URL, CUSTOM_INSTRUCTIONS
from utils.tone import TONE_INSTRUCTIONS_KOREAN, DEFAULT_TONE

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", "cs50_confirmed.json")
TARGET_LANG = "Korean"
NUM_ROUNDS = 3


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


async def run_translator(name: str, translator, sentences: list[dict]) -> list[dict]:
    """Run all sentences through a translator and collect timing results."""
    results = []

    for entry in sentences:
        text = entry["text"]
        elapsed_ms = entry.get("elapsed_ms", 0)

        t_start = time.monotonic()
        await translator.translate_confirmed(text, elapsed_ms)
        latency_ms = (time.monotonic() - t_start) * 1000

        results.append({
            "id": entry["id"],
            "source": text,
            "latency_ms": round(latency_ms, 1),
        })

    return results


async def run_deepl_bare(translator, sentences: list[dict]) -> list[dict]:
    """Run DeepL with no context, no custom_instructions, latency_optimized only."""
    results = []

    for entry in sentences:
        text = entry["text"]

        t_start = time.monotonic()
        await translator._call_deepl(text, f"BARE ({len(text.split())}w)", context="", model_type="latency_optimized")
        latency_ms = (time.monotonic() - t_start) * 1000

        results.append({
            "id": entry["id"],
            "source": text,
            "latency_ms": round(latency_ms, 1),
        })

    return results


def compute_stats(latencies: list[float]) -> dict:
    s = sorted(latencies)
    return {
        "avg": sum(s) / len(s),
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
        "min": s[0],
        "max": s[-1],
        "total": sum(s),
    }


def print_round_results(name: str, round_num: int, results: list[dict]):
    print(f"\n  {name} — Round {round_num}")
    print(f"  {'-'*50}")

    for r in results:
        print(f"    #{r['id']:>2}  {r['latency_ms']:>7.1f}ms  {r['source'][:55]}")

    stats = compute_stats([r["latency_ms"] for r in results])
    print(f"\n    Avg: {stats['avg']:.0f}ms | P50: {stats['p50']:.0f}ms | P90: {stats['p90']:.0f}ms")


def print_aggregate(name: str, all_latencies: list[float]):
    stats = compute_stats(all_latencies)
    print(f"\n{'='*70}")
    print(f"  {name} — Aggregate over {NUM_ROUNDS} rounds ({len(all_latencies)} sentences)")
    print(f"{'='*70}")
    print(f"  Avg: {stats['avg']:.0f}ms | P50: {stats['p50']:.0f}ms | P90: {stats['p90']:.0f}ms")
    print(f"  Min: {stats['min']:.0f}ms | Max: {stats['max']:.0f}ms | Total: {stats['total']:.0f}ms")
    return stats


async def test_latency():
    """3-round latency comparison: Realtime vs DeepL full vs DeepL bare."""
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} sentences from CS50 dataset")
    print(f"Target language: {TARGET_LANG}")
    print(f"Rounds: {NUM_ROUNDS}\n")

    rt_all_latencies: list[float] = []
    dl_all_latencies: list[float] = []
    dl_bare_all_latencies: list[float] = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"  ROUND {round_num} / {NUM_ROUNDS}")
        print(f"{'='*70}")

        # --- Realtime Translator ---
        rt = RealtimeTranslator(target_lang=TARGET_LANG)
        rt_results = await run_translator("RealtimeTranslator", rt, dataset)
        print_round_results("RealtimeTranslator", round_num, rt_results)
        await rt.close()
        rt_all_latencies.extend(r["latency_ms"] for r in rt_results)

        # --- DeepL Translator (full) ---
        dl = DeepLTranslator(target_lang=TARGET_LANG)
        dl_results = await run_translator("DeepLTranslator", dl, dataset)
        print_round_results("DeepL (full)", round_num, dl_results)
        await dl.close()
        dl_all_latencies.extend(r["latency_ms"] for r in dl_results)

        # --- DeepL Translator (bare: no context, latency_optimized) ---
        dl_bare = DeepLTranslator(target_lang=TARGET_LANG)
        dl_bare_results = await run_deepl_bare(dl_bare, dataset)
        print_round_results("DeepL (bare)", round_num, dl_bare_results)
        await dl_bare.close()
        dl_bare_all_latencies.extend(r["latency_ms"] for r in dl_bare_results)

    # --- Aggregate results ---
    rt_stats = print_aggregate("RealtimeTranslator (OpenAI Realtime API)", rt_all_latencies)
    dl_stats = print_aggregate("DeepL (full: quality_optimized + context)", dl_all_latencies)
    dl_bare_stats = print_aggregate("DeepL (bare: latency_optimized, no context)", dl_bare_all_latencies)

    # --- Comparison ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (avg of {NUM_ROUNDS} rounds)")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Realtime':>12} {'DeepL full':>12} {'DeepL bare':>12}")
    print(f"  {'-'*48}")
    for metric in ["avg", "p50", "p90", "min", "max", "total"]:
        rt_val = rt_stats[metric]
        dl_val = dl_stats[metric]
        bare_val = dl_bare_stats[metric]
        print(f"  {metric.upper():<12} {rt_val:>10.0f}ms {dl_val:>10.0f}ms {bare_val:>10.0f}ms")


async def run_with_translations(name: str, translator, sentences: list[dict]) -> list[dict]:
    """Run all sentences through translate_confirmed and capture translations + timing."""
    results = []
    for entry in sentences:
        text = entry["text"]
        elapsed_ms = entry.get("elapsed_ms", 0)
        captured = {"translation": ""}

        async def capture(translated, _elapsed_ms):
            captured["translation"] = translated

        translator.on_confirmed = capture
        t_start = time.monotonic()
        await translator.translate_confirmed(text, elapsed_ms)
        latency_ms = (time.monotonic() - t_start) * 1000

        results.append({
            "id": entry["id"],
            "source": text,
            "translation": captured["translation"],
            "latency_ms": round(latency_ms, 1),
        })
    return results


async def test_quality():
    """Side-by-side quality comparison: DeepL full vs GPT Realtime."""
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} sentences from CS50 dataset")
    print(f"Target language: {TARGET_LANG}\n")

    # --- DeepL full (quality_optimized + context + custom_instructions) ---
    print("Running DeepL full (quality_optimized + context + ci)...")
    dl = DeepLTranslator(target_lang=TARGET_LANG)
    dl_results = await run_with_translations("DeepL full", dl, dataset)
    await dl.close()

    # --- GPT Realtime ---
    print("\nRunning GPT Realtime (gpt-realtime-mini)...")
    rt = RealtimeTranslator(target_lang=TARGET_LANG)
    rt_results = await run_with_translations("Realtime", rt, dataset)
    await rt.close()

    # --- Side-by-side quality ---
    print(f"\n{'='*90}")
    print(f"  SIDE-BY-SIDE QUALITY COMPARISON")
    print(f"  DeepL = quality_optimized + context + custom_instructions")
    print(f"  RT    = GPT Realtime (gpt-realtime-mini)")
    print(f"{'='*90}")
    for dl_r, rt_r in zip(dl_results, rt_results):
        print(f"\n  #{dl_r['id']:>2} Source: {dl_r['source']}")
        print(f"      DeepL ({dl_r['latency_ms']:>5.0f}ms): {dl_r['translation']}")
        print(f"         RT ({rt_r['latency_ms']:>5.0f}ms): {rt_r['translation']}")

    # --- Latency comparison ---
    dl_lats = [r["latency_ms"] for r in dl_results]
    rt_lats = [r["latency_ms"] for r in rt_results]
    dl_stats = compute_stats(dl_lats)
    rt_stats = compute_stats(rt_lats)

    print(f"\n{'='*90}")
    print(f"  LATENCY COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Metric':<12} {'DeepL':>12} {'Realtime':>12} {'Diff':>12}")
    print(f"  {'-'*48}")
    for metric in ["avg", "p50", "p90", "min", "max", "total"]:
        dl_val = dl_stats[metric]
        rt_val = rt_stats[metric]
        diff = rt_val - dl_val
        sign = "+" if diff > 0 else ""
        print(f"  {metric.upper():<12} {dl_val:>10.0f}ms {rt_val:>10.0f}ms {sign}{diff:>10.0f}ms")


async def run():
    """Entry point — dispatches to the test selected by RUN_TEST."""
    tests = {
        "latency": test_latency,
        "quality": test_quality,
    }
    test_fn = tests.get(RUN_TEST)
    if not test_fn:
        print(f"Unknown test: {RUN_TEST}. Available: {', '.join(tests.keys())}")
        return
    print(f"Running test: {RUN_TEST}\n")
    await test_fn()
