"""
vLLM Semantic Router Runner for LiveCodeBench.

This runner calls the vLLM-SR HTTP API which is OpenAI-compatible.
It supports both direct model specification and MoM (Mixture of Models) routing.

Usage:
    python -m lcb_runner.runner.main --model vllm-sr-MoM --scenario codegeneration --evaluate

Environment variables:
    VLLM_SR_BASE_URL: Base URL for vLLM-SR (default: http://localhost:8899)
    VLLM_SR_API_KEY: API key for vLLM-SR (default: empty, not required)
"""

import os
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
        base_url = os.getenv("VLLM_SR_BASE_URL", "http://localhost:8899")
        api_key = os.getenv("VLLM_SR_API_KEY", "not-needed")
        
        # Extract model name from the full model identifier
        # Format: "vllm-sr-{model_name}" where model_name can be "MoM" or a specific model
        if args.model.startswith("vllm-sr-"):
            self.model_name = args.model[8:]  # Remove "vllm-sr-" prefix
        else:
            self.model_name = "MoM"  # Default to MoM (automatic routing)
        
        # Create OpenAI client pointing to vLLM-SR
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url}/v1",
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
        
        print(f"[vLLM-SR] Initialized with base_url={base_url}, model={self.model_name}")

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        """Run a single prompt through vLLM-SR."""
        assert isinstance(prompt, list), "Prompt must be a list of messages"

        if n == 0:
            print("[vLLM-SR] Max retries reached. Returning empty response.")
            return []

        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
            
            # Log some useful headers if available (for debugging routing decisions)
            # Note: OpenAI SDK doesn't expose response headers directly
            
            return [c.message.content for c in response.choices]
            
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[vLLM-SR] Exception: {repr(e)}")
            print("[vLLM-SR] Sleeping for 10 seconds...")
            sleep(10)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"[vLLM-SR] Failed to run the model for {prompt[:100]}...")
            print(f"[vLLM-SR] Exception: {repr(e)}")
            raise e
