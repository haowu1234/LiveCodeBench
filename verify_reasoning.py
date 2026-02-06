#!/usr/bin/env python3
"""
Verify reasoning_effort parameter is working correctly.

Usage:
    python verify_reasoning.py --base_url http://localhost:8002 --model gpt-oss-120b
    python verify_reasoning.py --model gpt-oss-120b --levels low high
    python verify_reasoning.py --model gpt-oss-120b --query "Prove sqrt(2) is irrational"
"""

import os
import time
import argparse
from openai import OpenAI


TEST_QUERIES = {
    "math": "Solve: Find all positive integers n such that n^2 + 2n + 4 is divisible by 7.",
    "logic": "A says 'B is lying'. B says 'C is lying'. C says 'A and B are both lying'. Who is telling the truth?",
    "aime": "Let S be the set of positive integers n where 3n has more divisors than n. Find the smallest element > 2023.",
    "simple": "What is 15 + 27?",
}


def test(client, model, query, reasoning_effort):
    """Test one reasoning level."""
    start = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=2048,
        reasoning_effort=reasoning_effort,  # SDK 原生支持
    )
    
    elapsed = time.time() - start
    content = response.choices[0].message.content or ""
    usage = response.usage
    
    return {
        "level": reasoning_effort or "none",
        "time": round(elapsed, 2),
        "tokens": usage.total_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "length": len(content),
        "preview": content[:300],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default=os.getenv('VLLM_API_BASE_URL', 'http://localhost:8002'))
    parser.add_argument('--model', default='gpt-oss-120b')
    parser.add_argument('--query', default=None)
    parser.add_argument('--preset', choices=TEST_QUERIES.keys(), default='math')
    parser.add_argument('--levels', nargs='+', default=['low', 'high'])
    args = parser.parse_args()
    
    client = OpenAI(api_key=os.getenv('VLLM_API_KEY', 'EMPTY'), base_url=f"{args.base_url}/v1")
    query = args.query or TEST_QUERIES[args.preset]
    
    print(f"Model: {args.model}")
    print(f"Query: {query[:80]}...")
    print("-" * 60)
    
    results = []
    for level in args.levels:
        effort = None if level == "none" else level
        print(f"Testing {level}...", end=" ", flush=True)
        r = test(client, args.model, query, effort)
        results.append(r)
        print(f"✓ {r['time']}s, {r['tokens']} tokens")
    
    # Compare
    print("\n" + "=" * 60)
    print(f"{'Level':<10} {'Time':<10} {'Tokens':<10} {'Length':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['level']:<10} {r['time']:<10} {r['tokens']:<10} {r['length']:<10}")
    
    if len(results) == 2:
        r1, r2 = results
        print(f"\nΔ Tokens: {r2['tokens'] - r1['tokens']:+d} ({r2['tokens']/r1['tokens']:.2f}x)")
        print(f"Δ Time: {r2['time'] - r1['time']:+.2f}s ({r2['time']/r1['time']:.2f}x)")
        
        if r2['tokens'] > r1['tokens'] * 1.2:
            print("\n✅ reasoning_effort 生效 (token 差异明显)")
        else:
            print("\n⚠️ 差异不明显，可能未生效或问题太简单")
    
    print("\n--- Responses ---")
    for r in results:
        print(f"\n[{r['level'].upper()}]: {r['preview']}...")


if __name__ == "__main__":
    main()
