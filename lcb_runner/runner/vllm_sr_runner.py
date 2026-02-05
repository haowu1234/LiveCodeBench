"""
vLLM Semantic Router Runner for LiveCodeBench.

This runner calls the vLLM-SR HTTP API which is OpenAI-compatible.
It supports both direct model specification and MoM (Mixture of Models) routing.

Usage:
    python -m lcb_runner.runner.main --model vllm-sr-MoM --scenario codegeneration --evaluate

Environment variables:
    VLLM_SR_BASE_URL: Base URL for vLLM-SR (default: http://localhost:8899)
    VLLM_SR_API_KEY: API key for vLLM-SR (default: empty, not required)
    VLLM_SR_DEBUG: Set to "1" to enable debug logging
    VLLM_SR_CONCURRENCY: Number of concurrent requests (default: 8)
"""

import os
import traceback
from time import sleep
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai package is required. Install with: pip install openai") from e

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class VLLMSRRunner(BaseRunner):
    """
    Runner for vLLM Semantic Router.
    
    Uses OpenAI-compatible API to call vLLM-SR endpoints.
    Supports both "MoM" (automatic routing) and direct model specification.
    Supports concurrent batch processing for faster evaluation.
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Get vLLM-SR configuration from environment
        self.base_url = os.getenv("VLLM_SR_BASE_URL", "http://localhost:8899")
        api_key = os.getenv("VLLM_SR_API_KEY", "not-needed")
        self.debug = os.getenv("VLLM_SR_DEBUG", "0") == "1"
        self.concurrency = int(os.getenv("VLLM_SR_CONCURRENCY", "8"))
        
        # Store expected_n from args
        self.expected_n = args.n
        self.args = args
        
        # Extract model name from the full model identifier
        # Format: "vllm-sr-{model_name}" where model_name can be "MoM" or a specific model
        if args.model.startswith("vllm-sr-"):
            self.model_name = args.model[8:]  # Remove "vllm-sr-" prefix
        else:
            self.model_name = "MoM"  # Default to MoM (automatic routing)
        
        # Create OpenAI client pointing to vLLM-SR
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
        )
        
        # Build client kwargs based on model style
        self.client_kwargs: dict = {
            "model": self.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "n": args.n,
            "timeout": args.openai_timeout,
        }
        
        print(f"[vLLM-SR] Initialized:")
        print(f"  - base_url: {self.base_url}")
        print(f"  - model: {self.model_name}")
        print(f"  - expected_n: {self.expected_n}")
        print(f"  - timeout: {args.openai_timeout}s")
        print(f"  - concurrency: {self.concurrency}")
        print(f"  - debug: {self.debug}")

    def run_batch(self, prompts: list) -> list[list[str]]:
        """
        Override run_batch to use concurrent processing.
        Process prompts in parallel using ThreadPoolExecutor.
        """
        print(f"\n[vLLM-SR] Processing {len(prompts)} prompts with concurrency={self.concurrency}")
        
        outputs = [None] * len(prompts)
        
        def process_single(idx_prompt):
            idx, prompt = idx_prompt
            try:
                result = self._run_single(prompt)
                return idx, result
            except Exception as e:
                print(f"[vLLM-SR] Error processing prompt {idx}: {repr(e)}")
                traceback.print_exc()
                return idx, [""] * self.expected_n
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single, (idx, prompt)): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results with progress bar
            with tqdm(total=len(prompts), desc="[vLLM-SR Concurrent]") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result = future.result()
                        outputs[idx] = result
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        print(f"[vLLM-SR] Future error for prompt {idx}: {repr(e)}")
                        outputs[idx] = [""] * self.expected_n
                        pbar.update(1)
        
        # Fill any None results with empty strings
        for i, output in enumerate(outputs):
            if output is None:
                outputs[i] = [""] * self.expected_n
        
        return outputs

    def _run_single(self, prompt: list[dict[str, str]], retry_count: int = 10) -> list[str]:
        """
        Run a single prompt through vLLM-SR.
        
        Since vLLM-SR doesn't support n > 1, we loop and call it n times
        to generate multiple samples for Pass@k evaluation.
        """
        
        # Validate input
        if not isinstance(prompt, list):
            if self.debug:
                print(f"[vLLM-SR ERROR] Invalid prompt type: {type(prompt)}, expected list")
            return [""] * self.expected_n
        
        if len(prompt) == 0:
            if self.debug:
                print(f"[vLLM-SR ERROR] Empty prompt list")
            return [""] * self.expected_n

        # Loop to generate n samples (since vLLM-SR doesn't support n > 1)
        results = []
        for sample_idx in range(self.expected_n):
            result = self._call_api_once(prompt, sample_idx, retry_count)
            results.append(result)
            
        return results
    
    def _call_api_once(self, prompt: list[dict[str, str]], sample_idx: int, retry_count: int = 10) -> str:
        """Call vLLM-SR API once and return a single result."""
        
        if retry_count == 0:
            print(f"[vLLM-SR] Max retries reached for sample {sample_idx}. Returning empty.")
            return ""

        try:
            # Build kwargs with n=1 (since SR doesn't support n > 1)
            call_kwargs = {
                "model": self.model_name,
                "temperature": self.client_kwargs["temperature"],
                "max_tokens": self.client_kwargs["max_tokens"],
                "top_p": self.client_kwargs["top_p"],
                "n": 1,  # Force n=1 for each call
                "timeout": self.client_kwargs["timeout"],
            }
            
            response = self.client.chat.completions.create(
                messages=prompt,
                **call_kwargs,
            )
            
            # Extract result
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content if response.choices[0].message else ""
                return content or ""
            else:
                return ""
            
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[vLLM-SR] OpenAI Exception for sample {sample_idx}: {repr(e)}")
            sleep(5)  # Shorter sleep for concurrent mode
            return self._call_api_once(prompt, sample_idx, retry_count=retry_count - 1)
        except Exception as e:
            print(f"[vLLM-SR] Unexpected Exception for sample {sample_idx}: {repr(e)}")
            return ""
