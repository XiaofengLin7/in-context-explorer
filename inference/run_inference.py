from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
from inference.data_loader import iter_steps_from_jsonl  # type: ignore
from inference.vllm_engine import VLLMEngine  # type: ignore

# Defaults (can be overridden by env vars or CLI args)
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 100
DEFAULT_TP_SIZE = 2
DEFAULT_N = 1
DEFAULT_SKIP_SPECIAL_TOKENS = True


def main() -> None:
    # Helper to read env var with casting
    def env_or(name: str, default: Any, cast: Any | None = None):
        val = os.environ.get(name)
        if val is None or val == "":
            return default
        if cast is None:
            return val
        if cast == list:
            # comma-separated list
            return [x.strip() for x in val.split(",") if x.strip()]
        try:
            return cast(val)
        except Exception:
            return default

    # Read defaults from environment first
    env_base_url = env_or("BASE_URL", DEFAULT_BASE_URL, str)
    env_model_path = env_or("MODEL_PATH", None, str)
    env_data_path = env_or("DATA_PATH", None, str)
    env_max_tokens = env_or("MAX_TOKENS", DEFAULT_MAX_TOKENS, int)
    env_temperature = env_or("TEMPERATURE", DEFAULT_TEMPERATURE, float)
    env_top_p = env_or("TOP_P", DEFAULT_TOP_P, float)
    env_top_k = env_or("TOP_K", DEFAULT_TOP_K, int)
    env_tp_size = env_or("TENSOR_PARALLEL_SIZE", DEFAULT_TP_SIZE, int)
    env_n = env_or("N", DEFAULT_N, int)
    env_skip_special_tokens = env_or("SKIP_SPECIAL_TOKENS", DEFAULT_SKIP_SPECIAL_TOKENS, int)
    env_output_path = env_or("OUTPUT_PATH", None, str)
    env_request_timeout = env_or("REQUEST_TIMEOUT", None, float)

    parser = argparse.ArgumentParser(description="Run vLLM inference (always starts local server)")
    parser.add_argument("--base-url", type=str, default=env_base_url)
    parser.add_argument("--model-path", type=str, default=env_model_path)
    parser.add_argument("--data-path", type=str, default=env_data_path)
    parser.add_argument("--max-tokens", type=int, default=env_max_tokens)
    parser.add_argument("--temperature", type=float, default=env_temperature)
    parser.add_argument("--top-p", type=float, default=env_top_p)
    parser.add_argument("--top-k", type=int, default=env_top_k)
    parser.add_argument("--tensor-parallel-size", type=int, default=env_tp_size)
    parser.add_argument("--n", type=int, default=env_n, help="Number of choices to generate (OpenAI compat)")
    parser.add_argument("--skip-special-tokens", action="store_true", default=bool(env_skip_special_tokens))
    parser.add_argument("--output-path", type=str, default=env_output_path)
    parser.add_argument("--request-timeout", type=float, default=env_request_timeout, help="HTTP read timeout seconds")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("model_path must be provided (via env MODEL_PATH or --model-path)")

    engine = VLLMEngine(
        model_path=args.model_path,
        base_url=args.base_url,
        request_timeout=args.request_timeout if args.request_timeout is not None else 120.0,
    )

    try:
        engine.start_server(tensor_parallel_size=args.tensor_parallel_size)

        inputs: list[list[dict]] = []
        meta: list[dict] = []

        if args.data_path and os.path.exists(args.data_path):
            for rec in iter_steps_from_jsonl(args.data_path):
                meta.append({"traj_uid": rec.get("traj_uid"), "step": rec.get("step")})
                inputs.append(rec["messages"])
        else:
            raise SystemExit("data_path must be provided (via env DATA_PATH or --data-path)")

        extra = {"skip_special_tokens": args.skip_special_tokens}
        print(f'Generating {len(inputs)} inputs...', inputs[0])
        out_list = engine.generate_batch(
            inputs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            n=args.n,
            extra=extra,
        )

        # Emit enriched per-step results with ids and model-specific response keys
        model_name = getattr(engine, "model_name", "model")
        response_key = f"{model_name}_response"
        reasoning_key = f"{model_name}_reasoning"
        results_jsonl = []
        for i, input, out in zip(inputs, out_list):
            record = {
                "traj_uid": meta[i].get("traj_uid") if i < len(meta) else None,
                "step": meta[i].get("step") if i < len(meta) else None,
                response_key: out.get("text", ""),
                reasoning_key: out.get("reasoning", ""),
                "input_messages": input,
            }
            results_jsonl.append(record)

        if args.output_path:
            out_dir = os.path.dirname(args.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                for rec in results_jsonl:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        engine.stop_server()


if __name__ == "__main__":
    main()


