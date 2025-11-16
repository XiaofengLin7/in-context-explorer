"""
Lightweight vLLM inference engine using the OpenAI-compatible HTTP API.

Usage examples:
1) Single prompt
   engine = VLLMEngine(model_name="qwen2.5-7b-instruct", base_url="http://localhost:8000/v1")
   result = engine.chat("Hello!", max_tokens=256)
   print(result["text"])

2) Batch prompts
   outputs = engine.generate_batch(["Hi", "How are you?"], max_tokens=64)
   for o in outputs:
       print(o["text"])

You can also run this file as a small CLI to test quickly:
   python -m in-context-explorer.inference.vllm_engine \\
     --model qwen2.5-7b-instruct \\
     --base-url http://localhost:8000/v1 \\
     --prompt "Say hello" \\
     --max-tokens 64
"""
from __future__ import annotations
import gc
import torch
import torch.distributed
import json
import subprocess
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from vllm.distributed import destroy_model_parallel
import requests
import os
from tqdm import tqdm

Messages = List[Dict[str, str]]


class VLLMEngine:
    """
    Minimal wrapper around a vLLM OpenAI-compatible server.
    Starts/stops a local server process for simple inference.
    """

    def __init__(self, model_path: str, base_url: str = "http://localhost:8000/v1", request_timeout: float = 120.0):
        """
        Args:
            model_path: Local path or hub id for the model to serve.
            base_url: Base URL of the vLLM server (e.g., http://localhost:8000/v1).
            request_timeout: Request timeout in seconds.
        """
        self.model_path = model_path
        # Use a simple served model name derived from the model path
        self.model_name = os.path.basename(model_path.rstrip("/"))
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.server_process: Optional[subprocess.Popen] = None
        self._last_server_cmd: Optional[List[str]] = None

    # ---------------------------- Server lifecycle ---------------------------- #
    def _wait_for_server(self, timeout_s: int = 1200, interval_s: int = 2) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                resp = requests.get(f"{self.base_url}/models", timeout=5, proxies={"http": None, "https": None})
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if self.server_process and self.server_process.poll() is not None:
                # Try to capture server logs to aid debugging
                try:
                    stdout, _ = self.server_process.communicate(timeout=1)
                except Exception:
                    stdout = ""
                cmd_str = " ".join(self._last_server_cmd or [])
                raise RuntimeError(
                    "vLLM server exited early with code "
                    f"{self.server_process.returncode}. Command: {cmd_str}\n--- vLLM stdout ---\n"
                    f"{stdout}\n--------------------"
                )
            time.sleep(interval_s)
        raise TimeoutError("Timed out waiting for vLLM server to become ready")

    def start_server(
        self,
        *,
        tensor_parallel_size: int = 1,
        extra_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        wait_timeout_s: int = 1200,
    ) -> None:
        """
        Launch a local vLLM OpenAI API server.
        """
        if self.server_process is not None:
            return
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        cmd: List[str] = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            '--served-model-name',
            self.model_name,
        ]

        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy() 
        # keep for error reporting
        self._last_server_cmd = cmd
        self.server_process = subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._wait_for_server(timeout_s=wait_timeout_s)
        print(f'vLLM server started with PID: {self.server_process.pid}')

    def stop_server(self) -> None:
        if self.server_process is None:
            return
        try:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        finally:
            self.server_process = None
        # Clean up all resources
        try:
            destroy_model_parallel()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f'Some cleanup steps failed:, but continuing...', e)

        print(f'Cleanup completed for model: {self.model_name}')

    def _post_chat(
        self,
        messages: Messages,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_special_tokens: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a single /chat/completions request and returns the parsed JSON.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "n": n,
            'skip_special_tokens': skip_special_tokens,
        }
        if extra:
            payload.update(extra)

        # Remove None values for cleanliness
        payload = {k: v for k, v in payload.items() if v is not None}

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.request_timeout,
            proxies={"http": None, "https": None},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM API error {resp.status_code}: {resp.text}")
        return resp.json()

    @staticmethod
    def _extract_choice_text(choice: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts content and optional reasoning (if present) from a completion choice.
        """
        message: Dict[str, Any] = choice.get("message", {})
        content: str = message.get("content", "") or ""
        reasoning: str = message.get("reasoning_content", "") or ""
        return {"text": content, "reasoning": reasoning}

    def chat(
        self,
        messages: Messages,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Single-turn completion call. Returns the first choice's text and reasoning if available.
        """
        data = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            max_tokens=max_tokens,
            extra=extra,
        )
        choices = data.get("choices", []) or []
        if not choices:
            return {"text": "", "reasoning": "", "raw": data}
        parsed = self._extract_choice_text(choices[0])
        return {"text": parsed["text"], "reasoning": parsed["reasoning"], "raw": data}

    def generate_batch(
        self,
        messages: Iterable[Messages],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        max_workers: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Simple parallel batched generation using threads.
        Returns a list of dicts with 'text', 'reasoning', and 'raw'.
        """
        messages = list(messages)
        results: List[Optional[Dict[str, Any]]] = [None] * len(messages)

        def _task(index_and_messages: Tuple[int, Messages]) -> Tuple[int, Dict[str, Any]]:
            idx, messages_i = index_and_messages
            out = self.chat(
                messages_i,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n=n,
                max_tokens=max_tokens,
                extra=extra,
            )
            return idx, out

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_task, (i, m)) for i, m in enumerate(messages)]
            with tqdm(total=len(messages), desc='Generating') as pbar:
                for fut in as_completed(futures):
                    idx, out = fut.result()
                    results[idx] = out

        # mypy: narrow
        return [r if r is not None else {"text": "", "reasoning": "", "raw": {}} for r in results]

    # Convenience for parity with other engines
    def generate_single(
        self,
        messages: Messages,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Single example generation. Alias for chat().
        """
        return self.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            max_tokens=max_tokens,
            extra=extra,
        )

