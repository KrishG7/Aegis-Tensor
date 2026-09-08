"""Aegis-Tensor Command Line Interface.

Unified CLI for static steganography detection and dynamic Trojan fuzzing in AI models.
"""

from pathlib import Path
from typing import Optional
import sys
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from aegis import (
    __version__,
    CORE_AVAILABLE,
    TORCH_AVAILABLE,
    scan_safetensors,
)

app = typer.Typer(
    name="aegis",
    help="🛡️ Aegis-Tensor: Advanced Adversarial Security Tool for AI Models (.safetensors)",
    add_completion=False,
)
console = Console()

BANNER = r"""[bold cyan]
    ___    ______ _____ _____ _____     _____ _____ _   _ _____ ___________ 
   / _ \  |  ____|  __ \_   _/ ____|   |_   _|  ___| \ | /  ___|  _  | ___ \
  / /_\ \ | |__  | |  \/ | || (___ ______| | | |__ |  \| \ `--.| | | | |_/ /
  |  _  | |  __| | | __  | | \___ \______| | |  __|| . ` |`--. \ | | |    / 
  | | | | | |____| |_\ \_| |_____) |     | | | |___| |\  /\__/ /\ \_/ / |\ \ 
  \_| |_/ |______|\____/_____|_____/     \_/ \____/\_| \_/\____/  \___/\_| \_|
[/bold cyan]
[dim]Adversarial Threat Scanner: Steganography & Sleeper Agent Trojan Detection[/dim]
"""


def print_banner():
    console.print(BANNER)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show Aegis-Tensor version."),
):
    if version:
        console.print(f"[bold cyan]Aegis-Tensor[/bold cyan] version [green]{__version__}[/green]")
        console.print(f"Rust Core Engine (`aegis_core`): {'[bold green]Loaded[/bold green]' if CORE_AVAILABLE else '[bold yellow]Not compiled (run `maturin develop`)[/bold yellow]'}")
        console.print(f"PyTorch Dynamic Engine: {'[bold green]Available[/bold green]' if TORCH_AVAILABLE else '[bold yellow]Not installed[/bold yellow]'}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("[yellow]Use [bold]aegis --help[/bold] to see available commands.[/yellow]")


@app.command(name="scan")
def scan_command(
    model_path: Path = typer.Argument(..., help="Path to the .safetensors model file to scan."),
    entropy_threshold: float = typer.Option(
        7.92,
        "--entropy-threshold",
        "-e",
        help="Shannon entropy threshold for flagging encrypted/compressed payloads (max 8.0).",
    ),
    benford_threshold: float = typer.Option(
        0.04,
        "--benford-threshold",
        "-b",
        help="Benford's Law MAD threshold for flagging non-natural float distributions.",
    ),
    json_output: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Save full scan report to a JSON file.",
    ),
):
    """Run zero-copy static security scan on a .safetensors model using Rust core."""
    print_banner()

    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found at: {model_path}")
        raise typer.Exit(code=1)

    if not CORE_AVAILABLE or scan_safetensors is None:
        console.print(
            Panel(
                "[bold red]Rust Core Engine (`aegis_core`) is not compiled![/bold red]\n\n"
                "Please build the native extension first by running:\n"
                "  [bold green]maturin develop[/bold green]\n"
                "or\n"
                "  [bold green]pip install -e .[/bold green]",
                title="⚠️ Native Extension Missing",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Scanning:[/bold] [cyan]{model_path.resolve()}[/cyan]")
    console.print(f"[dim]Thresholds: Shannon Entropy > {entropy_threshold:.2f} | Benford MAD > {benford_threshold:.4f}[/dim]\n")

    with console.status("[bold green]Memory-mapping and scanning tensors...[/bold green]"):
        try:
            results = scan_safetensors(str(model_path), entropy_threshold, benford_threshold)
        except Exception as e:
            console.print(f"[bold red]Scan failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Build results table
    table = Table(
        title=f"Aegis-Tensor Static Analysis Report ({len(results)} Tensors)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Tensor Name", style="cyan", overflow="ellipsis")
    table.add_column("Dtype", justify="center")
    table.add_column("Shape", justify="center")
    table.add_column("Entropy (Bits)", justify="right")
    table.add_column("Benford MAD", justify="right")
    table.add_column("Status", justify="center")

    suspicious_count = 0
    report_data = []

    for res in results:
        is_susp = res.is_suspicious
        if is_susp:
            suspicious_count += 1
            status_text = "[bold red]SUSPICIOUS[/bold red]"
        else:
            status_text = "[bold green]CLEAN[/bold green]"

        entropy_color = "red" if res.entropy > entropy_threshold else "white"
        benford_color = "red" if res.benford_mad > benford_threshold else "white"

        table.add_row(
            res.name,
            res.dtype,
            str(res.shape),
            f"[{entropy_color}]{res.entropy:.4f}[/{entropy_color}]",
            f"[{benford_color}]{res.benford_mad:.4f}[/{benford_color}]",
            status_text,
        )

        report_data.append({
            "name": res.name,
            "dtype": res.dtype,
            "shape": res.shape,
            "num_elements": res.num_elements,
            "entropy": res.entropy,
            "benford_mad": res.benford_mad,
            "is_suspicious": res.is_suspicious,
            "anomaly_reasons": res.anomaly_reasons,
        })

    console.print(table)

    if suspicious_count > 0:
        console.print(
            Panel(
                f"[bold red]ALERT: {suspicious_count} suspicious tensor(s) detected![/bold red]\n"
                "Tensors exhibit anomalous Shannon entropy or Benford's Law deviations indicative "
                "of hidden steganographic malware or injected cryptographic payloads.",
                title="🚨 Adversarial Threat Detected",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]No steganographic anomalies detected.[/bold green]\n"
                "All tensor entropy and Benford's Law distributions conform to expected neural network profiles.",
                title="✅ Model Clean",
                border_style="green",
            )
        )

    if json_output:
        with open(json_output, "w") as f:
            json.dump({
                "model_path": str(model_path.resolve()),
                "suspicious_count": suspicious_count,
                "tensors": report_data,
            }, f, indent=2)
        console.print(f"[dim]Report saved to: {json_output}[/dim]")


@app.command(name="doctor")
def doctor_command():
    """Diagnose local environment for Aegis-Tensor dependencies."""
    print_banner()
    table = Table(title="Aegis-Tensor Environment Diagnostics", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    # Python
    table.add_row("Python", "[green]OK[/green]", sys.version.split()[0])

    # Rust Core
    if CORE_AVAILABLE:
        table.add_row("Rust Core (`aegis_core`)", "[green]OK[/green]", "Native extension loaded")
    else:
        table.add_row("Rust Core (`aegis_core`)", "[yellow]NOT BUILT[/yellow]", "Run `maturin develop` to compile")

    # PyTorch
    if TORCH_AVAILABLE:
        import torch  # type: ignore
        table.add_row("PyTorch Engine", "[green]OK[/green]", f"v{torch.__version__}")
    else:
        table.add_row("PyTorch Engine", "[yellow]MISSING[/yellow]", "Install torch for dynamic Trojan fuzzing")

    console.print(table)


if __name__ == "__main__":
    app()
