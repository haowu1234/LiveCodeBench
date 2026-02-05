#!/usr/bin/env python3
"""
LiveCodeBench 评测结果分析脚本

用法:
    python -m lcb_runner.analysis.analyze_results --model vLLM-SR-MoM
    python -m lcb_runner.analysis.analyze_results --file output/vLLM-SR-MoM/Scenario.codegeneration_1_0.2_eval_all.json
    python -m lcb_runner.analysis.analyze_results --compare vLLM-SR-MoM vllm-api-DeepSeek-V3.2
"""

import os
import json
import argparse
import glob
from typing import Dict, List, Optional
from collections import defaultdict


def find_eval_file(model_name: str, output_dir: str = "output") -> Optional[str]:
    """根据模型名查找评测结果文件"""
    pattern = os.path.join(output_dir, model_name, "*_eval_all.json")
    files = glob.glob(pattern)
    if files:
        # 返回最新的文件
        return max(files, key=os.path.getmtime)
    return None


def load_eval_data(filepath: str) -> List[dict]:
    """加载评测结果数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_results(data: List[dict]) -> Dict:
    """分析评测结果"""
    # 按难度统计
    difficulty_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'pass_sum': 0.0})
    
    for d in data:
        diff = d.get('difficulty', 'unknown')
        pass_at_1 = d.get('pass@1', 0)
        passed = pass_at_1 == 1.0
        
        difficulty_stats[diff]['total'] += 1
        difficulty_stats[diff]['pass_sum'] += pass_at_1
        if passed:
            difficulty_stats[diff]['passed'] += 1
    
    # 总体统计
    total = len(data)
    passed = sum(1 for d in data if d.get('pass@1', 0) == 1.0)
    pass_sum = sum(d.get('pass@1', 0) for d in data)
    
    return {
        'total': total,
        'passed': passed,
        'pass_sum': pass_sum,
        'pass_at_1': passed / total if total > 0 else 0,
        'pass_at_1_avg': pass_sum / total if total > 0 else 0,
        'difficulty_stats': dict(difficulty_stats)
    }


def print_results(model_name: str, results: Dict, filepath: str = None):
    """打印评测结果"""
    print('=' * 70)
    print(f'📋 {model_name} LiveCodeBench 评测结果')
    if filepath:
        print(f'   文件: {filepath}')
    print('=' * 70)
    
    print(f'\n📊 总体结果:')
    print(f'  总题数: {results["total"]}')
    print(f'  完全通过: {results["passed"]}')
    print(f'  Pass@1 (严格): {results["pass_at_1"]*100:.2f}%')
    print(f'  Pass@1 (平均): {results["pass_at_1_avg"]*100:.2f}%')
    
    print(f'\n📈 按难度分布:')
    difficulty_order = ['easy', 'medium', 'hard']
    
    # 打印已知难度
    for diff in difficulty_order:
        if diff in results['difficulty_stats']:
            stats = results['difficulty_stats'][diff]
            rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_rate = stats['pass_sum'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f'  {diff:8s}: {stats["passed"]:4d}/{stats["total"]:4d} = {rate:6.2f}% (avg: {avg_rate:.2f}%)')
    
    # 打印未知难度
    for diff, stats in results['difficulty_stats'].items():
        if diff not in difficulty_order:
            rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_rate = stats['pass_sum'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f'  {diff:8s}: {stats["passed"]:4d}/{stats["total"]:4d} = {rate:6.2f}% (avg: {avg_rate:.2f}%)')
    
    print('=' * 70)


def compare_models(models_data: Dict[str, Dict]):
    """对比多个模型的结果"""
    print('\n' + '=' * 90)
    print('📊 模型对比')
    print('=' * 90)
    
    # 表头
    header = f'{"模型":<30} {"总题数":>8} {"通过":>8} {"Pass@1":>10} {"Easy":>10} {"Medium":>10} {"Hard":>10}'
    print(header)
    print('-' * 90)
    
    for model_name, results in models_data.items():
        total = results['total']
        passed = results['passed']
        pass_at_1 = results['pass_at_1'] * 100
        
        # 各难度通过率
        easy_rate = medium_rate = hard_rate = '-'
        diff_stats = results['difficulty_stats']
        
        if 'easy' in diff_stats:
            s = diff_stats['easy']
            easy_rate = f'{s["passed"]/s["total"]*100:.1f}%' if s['total'] > 0 else '-'
        if 'medium' in diff_stats:
            s = diff_stats['medium']
            medium_rate = f'{s["passed"]/s["total"]*100:.1f}%' if s['total'] > 0 else '-'
        if 'hard' in diff_stats:
            s = diff_stats['hard']
            hard_rate = f'{s["passed"]/s["total"]*100:.1f}%' if s['total'] > 0 else '-'
        
        row = f'{model_name:<30} {total:>8} {passed:>8} {pass_at_1:>9.2f}% {easy_rate:>10} {medium_rate:>10} {hard_rate:>10}'
        print(row)
    
    print('=' * 90)


def export_csv(models_data: Dict[str, Dict], output_file: str):
    """导出结果到 CSV 文件"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Total', 'Passed', 'Pass@1', 'Easy', 'Medium', 'Hard'])
        
        for model_name, results in models_data.items():
            total = results['total']
            passed = results['passed']
            pass_at_1 = results['pass_at_1']
            
            diff_stats = results['difficulty_stats']
            easy = diff_stats.get('easy', {})
            medium = diff_stats.get('medium', {})
            hard = diff_stats.get('hard', {})
            
            easy_rate = easy['passed'] / easy['total'] if easy.get('total', 0) > 0 else 0
            medium_rate = medium['passed'] / medium['total'] if medium.get('total', 0) > 0 else 0
            hard_rate = hard['passed'] / hard['total'] if hard.get('total', 0) > 0 else 0
            
            writer.writerow([
                model_name, total, passed,
                f'{pass_at_1:.4f}',
                f'{easy_rate:.4f}',
                f'{medium_rate:.4f}',
                f'{hard_rate:.4f}'
            ])
    
    print(f'\n✅ 结果已导出到: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='LiveCodeBench 评测结果分析')
    parser.add_argument('--model', '-m', type=str, help='模型名称')
    parser.add_argument('--file', '-f', type=str, help='评测结果文件路径')
    parser.add_argument('--compare', '-c', nargs='+', help='对比多个模型')
    parser.add_argument('--output-dir', '-o', type=str, default='output', help='输出目录')
    parser.add_argument('--export-csv', type=str, help='导出 CSV 文件路径')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用的评测结果')
    
    args = parser.parse_args()
    
    # 列出所有可用结果
    if args.list:
        print('\n📂 可用的评测结果:')
        if os.path.exists(args.output_dir):
            for model_dir in os.listdir(args.output_dir):
                model_path = os.path.join(args.output_dir, model_dir)
                if os.path.isdir(model_path):
                    eval_files = glob.glob(os.path.join(model_path, '*_eval_all.json'))
                    if eval_files:
                        print(f'  - {model_dir}')
                        for ef in eval_files:
                            print(f'      {os.path.basename(ef)}')
        return
    
    models_data = {}
    
    # 对比模式
    if args.compare:
        for model_name in args.compare:
            filepath = find_eval_file(model_name, args.output_dir)
            if filepath:
                data = load_eval_data(filepath)
                results = analyze_results(data)
                models_data[model_name] = results
                print_results(model_name, results, filepath)
            else:
                print(f'⚠️  未找到模型 {model_name} 的评测结果')
        
        if len(models_data) > 1:
            compare_models(models_data)
    
    # 单模型模式
    elif args.model:
        filepath = find_eval_file(args.model, args.output_dir)
        if filepath:
            data = load_eval_data(filepath)
            results = analyze_results(data)
            models_data[args.model] = results
            print_results(args.model, results, filepath)
        else:
            print(f'⚠️  未找到模型 {args.model} 的评测结果')
    
    # 直接指定文件
    elif args.file:
        if os.path.exists(args.file):
            data = load_eval_data(args.file)
            results = analyze_results(data)
            model_name = os.path.basename(os.path.dirname(args.file))
            models_data[model_name] = results
            print_results(model_name, results, args.file)
        else:
            print(f'⚠️  文件不存在: {args.file}')
    
    else:
        parser.print_help()
        print('\n示例:')
        print('  python -m lcb_runner.analysis.analyze_results --model vLLM-SR-MoM')
        print('  python -m lcb_runner.analysis.analyze_results --compare vLLM-SR-MoM vllm-api-DeepSeek-V3.2')
        print('  python -m lcb_runner.analysis.analyze_results --list')
    
    # 导出 CSV
    if args.export_csv and models_data:
        export_csv(models_data, args.export_csv)


if __name__ == '__main__':
    main()
