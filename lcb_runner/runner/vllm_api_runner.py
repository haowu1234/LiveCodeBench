"""
vLLM API Runner for LiveCodeBench.

This runner calls a remote vLLM server via OpenAI-compatible API.
Unlike vllm_runner.py which loads models locally, this runner connects
to an existing vLLM server.

Usage:
    python -m lcb_runner.runner.main --model vllm-api-DeepSeek-V3.2 --scenario codegeneration --evaluate

Environment variables:
    VLLM_API_BASE_URL: Base URL for vLLM server (default: http://localhost:8002)
    VLLM_API_KEY: API key (default: "EMPTY")
    VLLM_API_DEBUG: Set to "1" to enable debug logging
    VLLM_API_CONCURRENCY: Number of concurrent requests (default: 8)
"""

import os
import traceback
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai package is required. Install with: pip install openai") from e

from lcb_runner.runner.base_runner import BaseRunner


class VLLMAPIRunner(BaseRunner):
    """
    Runner for remote vLLM server via OpenAI-compatible API.
    
    This is different from VLLMRunner which loads models locally.
    This runner connects to an existing vLLM server and supports n > 1.
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Get vLLM API configuration from environment
        self.base_url = os.getenv("VLLM_API_BASE_URL", "http://localhost:8002")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.debug = os.getenv("VLLM_API_DEBUG", "1") == "1"
        self.concurrency = int(os.getenv("VLLM_API_CONCURRENCY", "16"))
        
        # Store expected_n from args
        self.expected_n = args.n
        self.args = args
        
        # Extract model name from the full model identifier
        # Format: "vllm-api-{model_name}"
        if model.model_name.startswith("vllm-api-"):
            self.model_name = model.model_name[9:]  # Remove "vllm-api-" prefix
        else:
            self.model_name = model.model_name
        
        # Create OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
        )
        
        # Check if vLLM server supports n > 1
        self.supports_n = True  # vLLM natively supports n parameter
        
        print(f"[vLLM-API] Initialized:")
        print(f"  - base_url: {self.base_url}")
        print(f"  - model: {self.model_name}")
        print(f"  - expected_n: {self.expected_n}")
        print(f"  - timeout: {args.openai_timeout}s")
        print(f"  - concurrency: {self.concurrency}")
        print(f"  - supports_n: {self.supports_n}")

    def run_batch(self, prompts: list) -> list[list[str]]:
        """
        Process prompts in parallel using ThreadPoolExecutor.
        """
        print(f"\n[vLLM-API] Processing {len(prompts)} prompts with concurrency={self.concurrency}")
        
        outputs = [None] * len(prompts)
        
        def process_single(idx_prompt):
            idx, prompt = idx_prompt
            try:
                result = self._run_single(prompt)
                return idx, result
            except Exception as e:
                print(f"[vLLM-API] Error processing prompt {idx}: {repr(e)}")
                if self.debug:
                    traceback.print_exc()
                return idx, [""] * self.expected_n
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(process_single, (idx, prompt)): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            with tqdm(total=len(prompts), desc="[vLLM-API Concurrent]") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result = future.result()
                        outputs[idx] = result
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        print(f"[vLLM-API] Future error for prompt {idx}: {repr(e)}")
                        outputs[idx] = [""] * self.expected_n
                        pbar.update(1)
        
        # Fill any None results with empty strings
        for i, output in enumerate(outputs):
            if output is None:
                outputs[i] = [""] * self.expected_n
        
        return outputs

    def _run_single(self, prompt: list[dict[str, str]], retry_count: int = 5) -> list[str]:
        """
        Run a single prompt through vLLM API.
        
        vLLM supports n > 1 natively, so we can get multiple samples in one call.
        """
        
        if not isinstance(prompt, list) or len(prompt) == 0:
            if self.debug:
                print(f"[vLLM-API ERROR] Invalid prompt: {type(prompt)}")
            return [""] * self.expected_n

        if retry_count == 0:
            print(f"[vLLM-API] Max retries reached. Returning empty.")
            return [""] * self.expected_n

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                top_p=self.args.top_p,
                n=self.expected_n,  # vLLM supports n > 1
                timeout=self.args.openai_timeout,
            )
            
            # Extract all n results
            results = []
            if response.choices:
                for choice in response.choices:
                    content = choice.message.content if choice.message else ""
                    results.append(content or "")
            
            # Pad with empty strings if we got fewer results
            while len(results) < self.expected_n:
                results.append("")
            
            return results[:self.expected_n]
            
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[vLLM-API] OpenAI Exception: {repr(e)}")
            sleep(3)
            return self._run_single(prompt, retry_count=retry_count - 1)
        except Exception as e:
            print(f"[vLLM-API] Unexpected Exception: {repr(e)}")
            if self.debug:
                traceback.print_exc()
            return [""] * self.expected_n
