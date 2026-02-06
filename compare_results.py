#!/usr/bin/env python3
"""
Compare LiveCodeBench evaluation results across multiple experiments.

Usage:
    python compare_results.py output/*/Scenario.codegeneration_*_eval_all.json
    python compare_results.py output/vLLM-API-GPT-OSS-120B/*_eval_all.json output/vLLM-SR-MoM/*_eval_all.json
"""

import json
import argparse
import os
from datetime import datetime
from collections import defaultdict
from tabulate import tabulate

try:
    from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k
except ImportError:
    # Fallback implementation
    import numpy as np
    def estimate_pass_at_k(num_samples, num_correct, k):
        """Estimates pass@k."""
        num_samples = np.array(num_samples)
        num_correct = np.array(num_correct)
        
        if (num_samples < k).any():
            # If not enough samples, use the available ones
            pass_k = num_correct / num_samples
        else:
            # Standard pass@k formula
            pass_k = 1.0 - np.prod(1.0 - k / num_samples * (num_correct / num_samples).clip(0, 1))
        return pass_k


def extract_experiment_name(filepath):
    """Extract a readable experiment name from filepath."""
    # e.g., output/vLLM-API-GPT-OSS-120B/Scenario.codegeneration_1_0.8_high_v2_eval_all.json
    # -> vLLM-API-GPT-OSS-120B / high_v2
    parts = filepath.split('/')
    model_name = parts[-2] if len(parts) >= 2 else "unknown"
    filename = parts[-1]
    
    # Extract run_id if present
    # Scenario.codegeneration_1_0.8_high_v2_eval_all.json -> high_v2
    # Scenario.codegeneration_1_0.8_eval_all.json -> default
    base = filename.replace('_eval_all.json', '').replace('Scenario.codegeneration_', '')
    # base: 1_0.8_high_v2 or 1_0.8
    parts = base.split('_')
    if len(parts) > 2:
        run_id = '_'.join(parts[2:])  # high_v2
    else:
        run_id = "default"
    
    n = parts[0]
    temp = parts[1] if len(parts) > 1 else "?"
    
    return f"{model_name} (n={n}, t={temp}, {run_id})"


def compute_metrics(results, k_values=[1]):
    """Compute Pass@k metrics from results."""
    metrics = {}
    
    # Filter valid results
    valid_results = [r for r in results if 'graded_list' in r and r['graded_list']]
    
    if not valid_results:
        return {'total': 0, 'pass@1': 0, 'easy_pass@1': 0, 'medium_pass@1': 0, 'hard_pass@1': 0}
    
    metrics['total'] = len(valid_results)
    
    # Overall metrics
    totals = [len(x['graded_list']) for x in valid_results]
    corrects = [sum(x['graded_list']) for x in valid_results]
    
    for k in k_values:
        try:
            metrics[f'pass@{k}'] = estimate_pass_at_k(totals, corrects, k).mean()
        except:
            # Fallback: simple accuracy
            metrics[f'pass@{k}'] = sum(corrects) / sum(totals) if sum(totals) > 0 else 0
    
    # By difficulty
    for difficulty in ['easy', 'medium', 'hard']:
        diff_results = [r for r in valid_results if r.get('difficulty') == difficulty]
        if diff_results:
            diff_totals = [len(x['graded_list']) for x in diff_results]
            diff_corrects = [sum(x['graded_list']) for x in diff_results]
            metrics[f'{difficulty}_count'] = len(diff_results)
            for k in k_values:
                try:
                    metrics[f'{difficulty}_pass@{k}'] = estimate_pass_at_k(diff_totals, diff_corrects, k).mean()
                except:
                    metrics[f'{difficulty}_pass@{k}'] = sum(diff_corrects) / sum(diff_totals) if sum(diff_totals) > 0 else 0
        else:
            metrics[f'{difficulty}_count'] = 0
            for k in k_values:
                metrics[f'{difficulty}_pass@{k}'] = 0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare LiveCodeBench results')
    parser.add_argument('files', nargs='+', help='eval_all.json files to compare')
    parser.add_argument('--start_date', type=str, default=None, help='Filter by start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Filter by end date (YYYY-MM-DD)')
    parser.add_argument('--k', type=int, nargs='+', default=[1], help='k values for Pass@k')
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table', help='Output format')
    args = parser.parse_args()
    
    all_metrics = []
    
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Parse dates and filter
        for res in results:
            if 'contest_date' in res and isinstance(res['contest_date'], str):
                try:
                    res['contest_date'] = datetime.fromisoformat(res['contest_date'])
                except:
                    pass
        
        if args.start_date:
            start = datetime.strptime(args.start_date, '%Y-%m-%d')
            results = [r for r in results if r.get('contest_date') and r['contest_date'] >= start]
        
        if args.end_date:
            end = datetime.strptime(args.end_date, '%Y-%m-%d')
            results = [r for r in results if r.get('contest_date') and r['contest_date'] <= end]
        
        exp_name = extract_experiment_name(filepath)
        metrics = compute_metrics(results, args.k)
        metrics['experiment'] = exp_name
        metrics['file'] = filepath
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid results found!")
        return
    
    # Output
    if args.format == 'json':
        print(json.dumps(all_metrics, indent=2, default=str))
    elif args.format == 'csv':
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    else:
        # Table format
        headers = ['Experiment', 'Total', 'Pass@1', 'Easy', 'Medium', 'Hard']
        rows = []
        for m in all_metrics:
            rows.append([
                m['experiment'],
                m['total'],
                f"{m.get('pass@1', 0)*100:.2f}%",
                f"{m.get('easy_pass@1', 0)*100:.2f}% ({m.get('easy_count', 0)})",
                f"{m.get('medium_pass@1', 0)*100:.2f}% ({m.get('medium_count', 0)})",
                f"{m.get('hard_pass@1', 0)*100:.2f}% ({m.get('hard_count', 0)})",
            ])
        
        print("\n" + "="*80)
        print("LiveCodeBench Results Comparison")
        print("="*80 + "\n")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print()
        
        # Also print a simple comparison if exactly 2 experiments
        if len(all_metrics) == 2:
            m1, m2 = all_metrics
            diff = (m2.get('pass@1', 0) - m1.get('pass@1', 0)) * 100
            print(f"Δ Pass@1: {diff:+.2f}% ({m2['experiment']} vs {m1['experiment']})")
            for difficulty in ['easy', 'medium', 'hard']:
                d1 = m1.get(f'{difficulty}_pass@1', 0)
                d2 = m2.get(f'{difficulty}_pass@1', 0)
                diff = (d2 - d1) * 100
                print(f"Δ {difficulty.capitalize()} Pass@1: {diff:+.2f}%")


if __name__ == '__main__':
    main()
