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
"""

import os
import traceback
from time import sleep
from typing import Optional

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
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Get vLLM-SR configuration from environment
        self.base_url = os.getenv("VLLM_SR_BASE_URL", "http://localhost:8899")
        api_key = os.getenv("VLLM_SR_API_KEY", "not-needed")
        self.debug = os.getenv("VLLM_SR_DEBUG", "0") == "1"
        
        # Store expected_n from args
        self.expected_n = args.n
        
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
        print(f"  - debug: {self.debug}")

    def _run_single(self, prompt: list[dict[str, str]], retry_count: int = 10) -> list[str]:
        """Run a single prompt through vLLM-SR."""
        
        # Debug: Log input
        if self.debug:
            print(f"\n[vLLM-SR DEBUG] _run_single called:")
            print(f"  - prompt type: {type(prompt)}")
            print(f"  - prompt length: {len(prompt) if isinstance(prompt, list) else 'N/A'}")
            print(f"  - retry_count: {retry_count}")
            print(f"  - expected_n: {self.expected_n}")
            if isinstance(prompt, list) and len(prompt) > 0:
                print(f"  - first message role: {prompt[0].get('role', 'N/A')}")
                print(f"  - first message content[:100]: {str(prompt[0].get('content', ''))[:100]}")
        
        # Validate input
        if not isinstance(prompt, list):
            print(f"[vLLM-SR ERROR] Invalid prompt type: {type(prompt)}, expected list")
            print(f"[vLLM-SR ERROR] Prompt value: {str(prompt)[:200]}")
            return [""] * self.expected_n
        
        if len(prompt) == 0:
            print(f"[vLLM-SR ERROR] Empty prompt list")
            return [""] * self.expected_n

        if retry_count == 0:
            print("[vLLM-SR] Max retries reached. Returning empty responses.")
            return [""] * self.expected_n

        try:
            if self.debug:
                print(f"[vLLM-SR DEBUG] Sending request to {self.base_url}/v1/chat/completions")
                print(f"[vLLM-SR DEBUG] client_kwargs: {self.client_kwargs}")
            
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
            
            if self.debug:
                print(f"[vLLM-SR DEBUG] Response received:")
                print(f"  - response type: {type(response)}")
                print(f"  - choices count: {len(response.choices) if response.choices else 0}")
            
            # Extract results from response
            results = []
            if response.choices:
                for i, c in enumerate(response.choices):
                    content = c.message.content if c.message and c.message.content else ""
                    results.append(content)
                    if self.debug:
                        print(f"  - choice[{i}] content[:100]: {content[:100] if content else 'EMPTY'}")
            else:
                print(f"[vLLM-SR WARNING] No choices in response!")
            
            # Ensure we return exactly the expected number of results
            if len(results) < self.expected_n:
                print(f"[vLLM-SR] Warning: Got {len(results)} results, expected {self.expected_n}. Padding with empty strings.")
                results.extend([""] * (self.expected_n - len(results)))
            elif len(results) > self.expected_n:
                print(f"[vLLM-SR] Warning: Got {len(results)} results, expected {self.expected_n}. Truncating.")
                results = results[:self.expected_n]
            
            if self.debug:
                print(f"[vLLM-SR DEBUG] Returning {len(results)} results")
            
            return results
            
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[vLLM-SR] OpenAI Exception: {repr(e)}")
            print(f"[vLLM-SR] Exception type: {type(e).__name__}")
            if self.debug:
                traceback.print_exc()
            print(f"[vLLM-SR] Sleeping for 10 seconds... (retries left: {retry_count - 1})")
            sleep(10)
            return self._run_single(prompt, retry_count=retry_count - 1)
        except Exception as e:
            print(f"[vLLM-SR] Unexpected Exception: {repr(e)}")
            print(f"[vLLM-SR] Exception type: {type(e).__name__}")
            print(f"[vLLM-SR] Prompt (first 200 chars): {str(prompt)[:200]}")
            traceback.print_exc()
            # Return empty results instead of raising to allow batch to continue
            print(f"[vLLM-SR] Returning {self.expected_n} empty results to continue batch.")
            return [""] * self.expected_n
