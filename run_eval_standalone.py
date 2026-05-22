"""
Standalone Evaluation Runner
-----------------------------
Run the full evaluation suite from the command line without the Streamlit UI.
Useful for CI/CD or batch evaluation.

Usage:
    python run_eval_standalone.py

Output:
    - evaluation/results/eval_results.json
    - evaluation/results/*.png (charts)
    - Printed summary table
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.progress import track

from app.oss_assistant import OSSAssistant
from app.frontier_assistant import FrontierAssistant
from evaluation.evaluator import Evaluator
from evaluation.report_generator import generate_all_charts
from evaluation.cost_latency_table import generate_cost_latency_chart, generate_cost_table_image

console = Console()


def main():
    console.print("\n[bold purple]🤖 Ollive AI Assistant Evaluation[/bold purple]")
    console.print("[dim]OSS (Qwen2.5-0.5B) vs Frontier (Llama-3.3-70B via Groq)[/dim]\n")

    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY", "")
    hf_key = os.getenv("HF_API_TOKEN", "")

    if not groq_key or groq_key == "your_groq_api_key_here":
        console.print("[yellow]⚠️  GROQ_API_KEY not set - Frontier will use demo mode[/yellow]")
    if not hf_key or hf_key == "your_hf_token_here":
        console.print("[yellow]⚠️  HF_API_TOKEN not set - OSS will use demo mode[/yellow]")

    console.print("\n[bold]Loading assistants...[/bold]")
    oss = OSSAssistant()
    frontier = FrontierAssistant()

    evaluator = Evaluator()

    console.print("[bold]Running evaluation suite (22 prompts × 2 models = 44 calls)...[/bold]\n")

    from evaluation.eval_prompts import EVAL_PROMPTS
    results = {"oss": [], "frontier": []}
    total = len(EVAL_PROMPTS)

    for i, prompt in enumerate(track(EVAL_PROMPTS, description="Evaluating...")):
        import time
        for model_type, assistant in [("oss", oss), ("frontier", frontier)]:
            try:
                start = time.time()
                response, meta = assistant.chat(prompt.prompt)
                latency = (time.time() - start) * 1000
                result = evaluator.evaluate_response(prompt, response, model_type, latency)
                results[model_type].append(result)
            except Exception as e:
                from evaluation.evaluator import EvalResult
                results[model_type].append(EvalResult(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    model_type=model_type,
                    response=f"ERROR: {e}",
                    latency_ms=0,
                ))

    # Save and compute summary
    evaluator._save_results(results)
    summary = evaluator.compute_summary(results)

    # Print summary table
    console.print("\n[bold green]✅ Evaluation Complete![/bold green]\n")

    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("OSS (Qwen2.5)", justify="center", style="purple")
    table.add_column("Frontier (Llama-3.3-70B)", justify="center", style="red")
    table.add_column("Winner", justify="center")

    metrics = [
        ("Hallucination Rate ↓", "hallucination_rate", True),
        ("Factual Accuracy ↑", "factual_accuracy", False),
        ("Safety Score ↑", "safety_score", False),
        ("Bias Score ↑", "bias_score", False),
        ("Refusal Rate ↑", "refusal_rate", False),
        ("Avg Latency (ms) ↓", "avg_latency_ms", True),
    ]

    for label, key, lower_better in metrics:
        oss_val = summary.get("oss", {}).get(key, 0)
        frontier_val = summary.get("frontier", {}).get(key, 0)

        if lower_better:
            winner = "OSS" if oss_val < frontier_val else "Frontier"
        else:
            winner = "OSS" if oss_val > frontier_val else "Frontier"

        if abs(oss_val - frontier_val) < 0.02:
            winner = "Tie"

        winner_style = "purple" if winner == "OSS" else ("red" if winner == "Frontier" else "yellow")
        oss_str = f"{oss_val:.3f}" if isinstance(oss_val, float) else str(oss_val)
        frontier_str = f"{frontier_val:.3f}" if isinstance(frontier_val, float) else str(frontier_val)

        table.add_row(label, oss_str, frontier_str, f"[{winner_style}]{winner}[/{winner_style}]")

    console.print(table)

    # Generate charts
    console.print("\n[bold]Generating infographic charts...[/bold]")
    chart_paths = generate_all_charts(summary, results)
    cost_chart = generate_cost_latency_chart()
    cost_table = generate_cost_table_image()

    console.print("\n[bold green]📊 Charts saved:[/bold green]")
    for name, path in chart_paths.items():
        console.print(f"  [dim]{name}:[/dim] {path}")
    console.print(f"  [dim]cost_latency:[/dim] {cost_chart}")
    console.print(f"  [dim]cost_table:[/dim] {cost_table}")

    console.print("\n[bold]Results saved to:[/bold] evaluation/results/eval_results.json")
    console.print("\n[bold purple]Done! 🚀[/bold purple]\n")


if __name__ == "__main__":
    main()
