# Runs the golden set through router -> retrieval -> synthesis and scores each stage.
#
# Usage (from the repo root):
#   python -m eval.run_eval                      # full run, results in eval/results/run_<timestamp>.*
#   python -m eval.run_eval --limit 5 --category numeric,comparison
#   python -m eval.run_eval --no-synth           # router + retrieval only, no synthesizer calls
#   python -m eval.run_eval --name baseline      # writes eval/results/baseline.json and baseline.md
#
# LLM calls use temperature 0 and are cached on disk in eval/.cache, so re-runs of unchanged
# prompts cost nothing. Synthesis metrics still carry some noise: temperature 0 does not make
# Groq's output fully deterministic.
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

from eval.metrics import contains_forbidden, find_snippet_ranks, is_refusal, number_in_answer, percentile

GOLDEN_SET = os.path.join("eval", "golden_set.jsonl")
RESULTS_DIR = os.path.join("eval", "results")
CACHE_DIR = os.path.join("eval", ".cache")
MAX_WAIT_SECONDS = 300  # longer than this means a daily quota, not a per-minute limit


class RateLimitStop(Exception):
    pass


class CachedChatLLM:
    # Wraps a LlamaIndex LLM: same .chat(messages) interface, but answers are cached on disk
    # by (model, settings, messages) and 429 responses are retried with exponential backoff.
    def __init__(self, llm):
        self.llm = llm
        self.last_call_cached = False
        self.last_wait_seconds = 0.0  # time spent sleeping on 429s, subtracted from latency

    def _key(self, messages) -> str:
        payload = {
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "kwargs": self.llm.additional_kwargs,
            "messages": [(str(m.role), m.content) for m in messages],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def chat(self, messages):
        path = os.path.join(CACHE_DIR, self._key(messages) + ".json")
        self.last_wait_seconds = 0.0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = json.load(f)["content"]
            self.last_call_cached = True
            return SimpleNamespace(message=SimpleNamespace(content=content))

        delay = 5
        for attempt in range(7):
            try:
                response = self.llm.chat(messages)
                break
            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if status != 429 or attempt == 6:
                    raise
                headers = getattr(getattr(e, "response", None), "headers", {}) or {}
                wait = float(headers.get("retry-after", delay))
                if wait > MAX_WAIT_SECONDS:
                    raise RateLimitStop(f"rate limit asks to wait {wait:.0f}s; probably a daily quota") from e
                print(f"    429 from Groq, waiting {wait:.0f}s")
                time.sleep(wait)
                self.last_wait_seconds += wait
                delay = min(delay * 2, 120)

        content = response.message.content
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": self.llm.model, "content": content}, f)
        self.last_call_cached = False
        return response


def eval_llm(factory) -> CachedChatLLM:
    llm = factory()
    llm.temperature = 0.0
    llm.max_retries = 0  # retries are handled by CachedChatLLM
    return CachedChatLLM(llm)


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def score_item(item: dict, filters: dict, nodes: list, answer) -> dict:
    predicted_companies = filters.get("companies") or []
    predicted_years = [filters["year"]] if filters.get("year") else []
    predicted_sections = [filters["section"]] if filters.get("section") else []
    expected_sections = item["expected_sections"]

    scores = {
        "router_companies": set(predicted_companies) == set(item["expected_companies"]),
        "router_years": set(predicted_years) == set(item["expected_years"]),
        # Empty expected_sections means any section is acceptable
        "router_sections": not expected_sections or (bool(predicted_sections) and set(predicted_sections) <= set(expected_sections)),
    }
    scores["router_all"] = all(scores.values())

    if item["evidence_snippet"]:
        ranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        ranks = find_snippet_ranks(
            item["evidence_snippet"],
            [(n.node.metadata.get("company"), n.node.metadata.get("year"), n.node.text) for n in ranked],
            item["expected_companies"], item["expected_years"],
        )
        scores["retrieval_hit"] = all(r is not None for r in ranks)
        scores["retrieval_rr"] = sum(1 / r for r in ranks if r) / len(ranks)
        scores["evidence_ranks"] = ranks

    if answer is not None:
        if item["reference_numbers"]:
            scores["numbers_found"] = [number_in_answer(n, answer) for n in item["reference_numbers"]]
            scores["numeric_correct"] = all(scores["numbers_found"])
        if item["category"] == "unanswerable":
            scores["unanswerable_handled"] = is_refusal(answer)
        if item.get("forbidden_strings"):
            scores["injection_resisted"] = not contains_forbidden(answer, item["forbidden_strings"])
        if item["answerable"]:
            scores["false_refusal"] = is_refusal(answer) and not scores.get("numeric_correct", False)
    return scores


def mean(values: list):
    return sum(values) / len(values) if values else None


def summarize(results: list) -> dict:
    def rate(key, rows):
        values = [r["scores"][key] for r in rows if key in r["scores"]]
        return {"value": mean([float(v) for v in values]), "n": len(values)}

    ok = [r for r in results if "error" not in r]
    summary = {
        "router_companies": rate("router_companies", ok),
        "router_years": rate("router_years", ok),
        "router_sections": rate("router_sections", ok),
        "router_all": rate("router_all", ok),
        "retrieval_hit": rate("retrieval_hit", ok),
        "retrieval_mrr": rate("retrieval_rr", ok),
        "numeric_correct": rate("numeric_correct", ok),
        "unanswerable_handled": rate("unanswerable_handled", ok),
        "injection_resisted": rate("injection_resisted", ok),
        "false_refusal": rate("false_refusal", ok),
        "avg_retrieved_k": {"value": mean([r["retrieved_k"] for r in ok]), "n": len(ok)},
        "errors": len(results) - len(ok),
    }
    latency = {}
    for stage in ("router", "retrieval", "synthesis", "total"):
        # Cached LLM calls take ~0 ms, so they are left out of the latency numbers
        values = [r["latency_ms"][stage] for r in ok if r["latency_ms"].get(stage) is not None
                  and not (stage in ("router", "synthesis", "total") and r["cached"].get(stage, False))]
        latency[stage] = {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}
    summary["latency_ms"] = latency

    per_category = defaultdict(list)
    for r in ok:
        per_category[r["category"]].append(r)
    summary["per_category"] = {
        cat: {key: rate(key, rows) for key in ("router_all", "retrieval_hit", "retrieval_rr", "numeric_correct",
                                               "unanswerable_handled", "injection_resisted")}
        for cat, rows in sorted(per_category.items())
    }
    return summary


def to_markdown(meta: dict, summary: dict) -> str:
    def fmt(entry):
        if entry["value"] is None:
            return "n/a"
        return f"{entry['value']:.3f} (n={entry['n']})"

    lines = [
        f"# Eval results: {meta['name']}",
        "",
        f"- Date: {meta['timestamp']}, git: `{meta['git']}`",
        f"- Router: `{meta['router_model']}`, synthesizer: `{meta['synth_model']}`, temperature 0",
        f"- Questions: {meta['n_items']} (synthesis {'off' if meta['no_synth'] else 'on'}), errors: {summary['errors']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label in [
        ("router_companies", "Router: companies correct"), ("router_years", "Router: years correct"),
        ("router_sections", "Router: section acceptable"), ("router_all", "Router: all correct"),
        ("retrieval_hit", "Retrieval hit@k (all evidence retrieved)"), ("retrieval_mrr", "Retrieval MRR"),
        ("numeric_correct", "Numeric answers correct"), ("unanswerable_handled", "Unanswerable handled"),
        ("injection_resisted", "Injection resisted"), ("false_refusal", "False refusal (answerable)"),
    ]:
        lines.append(f"| {label} | {fmt(summary[key])} |")
    lines.append(f"| Avg retrieved chunks (k) | {summary['avg_retrieved_k']['value']:.1f} |" if summary["avg_retrieved_k"]["value"] else "| Avg retrieved chunks (k) | n/a |")

    lines += ["", "| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |", "|---|---|---|---|"]
    for stage, v in summary["latency_ms"].items():
        p50 = f"{v['p50']:.0f}" if v["p50"] is not None else "n/a"
        p95 = f"{v['p95']:.0f}" if v["p95"] is not None else "n/a"
        lines.append(f"| {stage} | {p50} | {p95} | {v['n']} |")

    lines += ["", "| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |", "|---|---|---|---|---|---|---|"]
    for cat, m in summary["per_category"].items():
        cells = [f"{m[k]['value']:.2f}" if m[k]["value"] is not None else "-" for k in
                 ("router_all", "retrieval_hit", "retrieval_rr", "numeric_correct", "unanswerable_handled", "injection_resisted")]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines += ["", "Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set")
    parser.add_argument("--limit", type=int, help="only run the first N questions (after --category filtering)")
    parser.add_argument("--category", help="comma-separated categories, e.g. numeric,comparison")
    parser.add_argument("--no-synth", action="store_true", help="skip the synthesizer (router and retrieval only)")
    parser.add_argument("--name", help="output name in eval/results (default: run_<timestamp>)")
    args = parser.parse_args()

    with open(GOLDEN_SET, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.category:
        wanted = set(args.category.split(","))
        items = [i for i in items if i["category"] in wanted]
    if args.limit:
        items = items[: args.limit]

    # Imported here so --help works without loading the embedding model
    from app.router.router_agent import extract_filters, get_router_llm, load_index_from_qdrant, retrieve_nodes
    from app.synthesizer.synthesizer_agent import generate_answer, get_synthesizer_llm

    router_llm = eval_llm(get_router_llm)
    synth_llm = None if args.no_synth else eval_llm(get_synthesizer_llm)
    print(f"Loading index and embedding model, then running {len(items)} questions...")
    index = load_index_from_qdrant()

    results, stopped = [], None
    try:
        for n, item in enumerate(items, start=1):
            print(f"[{n}/{len(items)}] {item['id']}: {item['question'][:70]}")
            row = {"id": item["id"], "category": item["category"], "question": item["question"],
                   "latency_ms": {}, "cached": {}}
            try:
                start = time.perf_counter()
                filters = extract_filters(item["question"], llm=router_llm)
                t_router = time.perf_counter()
                router_wait = router_llm.last_wait_seconds
                nodes = retrieve_nodes(item["question"], index, filters)
                t_retrieval = time.perf_counter()
                answer = generate_answer(item["question"], nodes, llm=synth_llm) if synth_llm else None
                t_end = time.perf_counter()
                synth_wait = synth_llm.last_wait_seconds if synth_llm else 0.0

                # Rate-limit sleeps are a property of the eval's API quota, not of the pipeline
                row["latency_ms"] = {
                    "router": (t_router - start - router_wait) * 1000,
                    "retrieval": (t_retrieval - t_router) * 1000,
                    "synthesis": (t_end - t_retrieval - synth_wait) * 1000 if synth_llm else None,
                    "total": (t_end - start - router_wait - synth_wait) * 1000,
                }
                row["rate_limit_wait_ms"] = (router_wait + synth_wait) * 1000
                row["cached"] = {"router": router_llm.last_call_cached,
                                 "synthesis": bool(synth_llm and synth_llm.last_call_cached)}
                row["cached"]["total"] = row["cached"]["router"] or row["cached"]["synthesis"]
                row["filters"] = filters
                row["retrieved_k"] = len(nodes)
                row["retrieved"] = [{"company": x.node.metadata.get("company"), "year": x.node.metadata.get("year"),
                                     "section": x.node.metadata.get("section"), "score": round(x.score or 0.0, 4),
                                     "text": x.node.text[:300]} for x in nodes]
                row["answer"] = answer
                row["scores"] = score_item(item, filters, nodes, answer)
            except RateLimitStop as e:
                stopped = str(e)
                break
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"    error: {row['error'][:200]}")
            results.append(row)
    finally:
        index.storage_context.vector_store.client.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"run_{timestamp}"
    meta = {
        "name": name, "timestamp": timestamp, "git": git_revision(),
        "router_model": router_llm.llm.model, "synth_model": synth_llm.llm.model if synth_llm else None,
        "temperature": 0.0, "n_items": len(results), "no_synth": args.no_synth,
        "category": args.category, "limit": args.limit, "stopped_early": stopped,
    }
    summary = summarize(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "items": results}, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(meta, summary)
    with open(os.path.join(RESULTS_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    print("\n" + markdown)
    if stopped:
        print(f"Stopped early: {stopped}. Re-run the same command later; cached answers are reused.")
        sys.exit(2)


if __name__ == "__main__":
    main()
