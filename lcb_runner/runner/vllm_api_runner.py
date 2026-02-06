"""
vLLM API Runner for LiveCodeBench.

This runner calls a remote vLLM server via OpenAI-compatible API.
Unlike vllm_runner.py which loads models locally, this runner connects
to an existing vLLM server.

Usage:
    python -m lcb_runner.runner.main --model vllm-api-DeepSeek-V3.2 --scenario codegeneration --evaluate
    
    # With high reasoning enabled:
    python -m lcb_runner.runner.main --model vllm-api-DeepSeek-V3.2 --high_reasoning --scenario codegeneration --evaluate

Environment variables:
    VLLM_API_BASE_URL: Base URL for vLLM server (default: http://localhost:8002)
    VLLM_API_KEY: API key (default: "EMPTY")
    VLLM_API_DEBUG: Set to "1" to enable debug logging
    VLLM_API_CONCURRENCY: Number of concurrent requests (default: 8)
    VLLM_API_REASONING_EFFORT: Reasoning effort level (low/medium/high, default: None)
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
    
    Supports high reasoning mode for models like DeepSeek-R1, Kimi-K2, etc.
    Enable via --high_reasoning flag or VLLM_API_REASONING_EFFORT env var.
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Get vLLM API configuration from environment
        self.base_url = os.getenv("VLLM_API_BASE_URL", "http://localhost:8002")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.debug = os.getenv("VLLM_API_DEBUG", "1") == "1"
        self.concurrency = int(os.getenv("VLLM_API_CONCURRENCY", "16"))
        
        # High reasoning support
        # Priority: args.high_reasoning > args.reasoning_effort > env var > None
        self.reasoning_effort = None
        if hasattr(args, 'high_reasoning') and args.high_reasoning:
            self.reasoning_effort = "high"
        elif hasattr(args, 'reasoning_effort') and args.reasoning_effort:
            self.reasoning_effort = args.reasoning_effort
        else:
            env_reasoning = os.getenv("VLLM_API_REASONING_EFFORT", "").lower()
            if env_reasoning in ("low", "medium", "high"):
                self.reasoning_effort = env_reasoning
        
        # Store expected_n from args
        self.expected_n = args.n
        self.args = args
        
        # Extract model name from the full model identifier
        # Format: "vllm-api-{model_name}" or "vllm-api-{model_name}__high"
        raw_model_name = model.model_name
        if raw_model_name.startswith("vllm-api-"):
            raw_model_name = raw_model_name[9:]  # Remove "vllm-api-" prefix
        
        # Check if model name contains reasoning effort suffix (e.g., "__high")
        if "__" in raw_model_name:
            self.model_name, suffix = raw_model_name.rsplit("__", 1)
            if suffix in ("low", "medium", "high") and self.reasoning_effort is None:
                self.reasoning_effort = suffix
        else:
            self.model_name = raw_model_name
        
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
        print(f"  - reasoning_effort: {self.reasoning_effort or 'disabled'}")

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
        Run a single prompt through vLLM API using OpenAI SDK.
        
        Uses native SDK parameters following OpenAI API specification:
        https://platform.openai.com/docs/api-reference/chat/create
        
        Key parameters used:
        - messages: The conversation messages
        - model: Model ID
        - temperature: Sampling temperature (0-2)
        - max_tokens: Maximum tokens to generate
        - top_p: Nucleus sampling threshold
        - n: Number of completions to generate
        - reasoning_effort: Constrains reasoning effort (low/medium/high) for reasoning models
        - stop: Stop sequences
        - timeout: Request timeout
        """
        
        if not isinstance(prompt, list) or len(prompt) == 0:
            if self.debug:
                print(f"[vLLM-API ERROR] Invalid prompt: {type(prompt)}")
            return [""] * self.expected_n

        if retry_count == 0:
            print(f"[vLLM-API] Max retries reached. Returning empty.")
            return [""] * self.expected_n

        try:
            # Use OpenAI SDK's native chat.completions.create() with proper parameters
            # Reference: https://platform.openai.com/docs/api-reference/chat/create
            response = self.client.chat.completions.create(
                # Required parameters
                messages=prompt,
                model=self.model_name,
                # Sampling parameters
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=self.expected_n,
                # Token limits
                max_tokens=self.args.max_tokens,
                # Stop sequences (if configured)
                stop=self.args.stop if hasattr(self.args, 'stop') and self.args.stop != ["###"] else None,
                # Reasoning effort for reasoning models (gpt-oss-120b, DeepSeek-R1, etc.)
                # Supported values: low, medium, high
                # Higher values = more reasoning tokens, better quality on complex tasks
                reasoning_effort=self.reasoning_effort if self.reasoning_effort else None,
                # Request timeout
                timeout=self.args.openai_timeout,
            )
            
            if self.debug and self.reasoning_effort:
                print(f"[vLLM-API] Request sent with reasoning_effort={self.reasoning_effort}")
            
            # Extract all n results
            results = []
            if response.choices:
                for choice in response.choices:
                    message = choice.message
                    if message:
                        content = message.content or ""
                        # For reasoning models, check for reasoning_content in response
                        # Some vLLM deployments return reasoning in message.reasoning_content
                        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
                        if reasoning and self.debug:
                            print(f"[vLLM-API] Reasoning tokens received: {len(reasoning)} chars")
                            print(f"[vLLM-API] Reasoning preview: {reasoning[:200]}...")
                        results.append(content)
                    else:
                        results.append("")
            
            # Log usage stats if available
            if self.debug and hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"[vLLM-API] Usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            
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
