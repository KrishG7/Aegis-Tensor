"""Aegis-Tensor Command Line Interface.

Unified CLI for static steganography detection and dynamic Trojan fuzzing in AI models.
"""

import sys

# Ensure UTF-8 output encoding across all platforms (including Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Optional
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich import box

from aegis import (
    __version__,
    CORE_AVAILABLE,
    TORCH_AVAILABLE,
    scan_safetensors,
    DynamicTrojanFuzzer,
    export_scan_sarif,
    export_fuzz_sarif,
    parse_hf_uri,
    inspect_hf_header,
    download_hf_model,
    HuggingFaceInspectorError,
)

def get_risk_badge(level: str) -> str:
    """Return styled security risk badge for terminal display."""
    if level == "CRITICAL":
        return "[bold white on red] [!] CRITICAL MALWARE [/bold white on red]"
    elif level == "SUSPICIOUS":
        return "[bold black on yellow] [?] SUSPICIOUS ANOMALY [/bold black on yellow]"
    elif level == "CLEAN":
        return "[bold white on green] [OK] CLEAN [/bold white on green]"
    return level

app = typer.Typer(
    name="aegis",
    help="Aegis-Tensor: Advanced Adversarial Security Tool for AI Models (.safetensors)",
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
    model_target: str = typer.Argument(
        ...,
        metavar="MODEL_PATH_OR_URI",
        help="Path to local .safetensors file or remote Hugging Face URI (e.g. 'hf://username/repo').",
    ),
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
    sarif_output: Optional[Path] = typer.Option(
        None,
        "--output-sarif",
        help="Save scan report in SARIF 2.1.0 format for GitHub Code Scanning.",
    ),
    download: bool = typer.Option(
        False,
        "--download",
        "-d",
        help="Download remote Hugging Face model to local cache for full static cryptanalysis scan.",
    ),
):
    """Run zero-copy static security scan on a .safetensors model using Rust core."""
    print_banner()

    # Check if target is a remote Hugging Face URI
    if model_target.startswith("hf://"):
        try:
            repo_id, filename, revision = parse_hf_uri(model_target)
        except (ValueError, HuggingFaceInspectorError) as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

        resolved_file = filename or "model.safetensors"
        resolved_rev = revision or "main"

        if not download:
            # Fast remote header inspection via HTTP Range Requests
            console.print(f"[bold cyan]Connecting to Hugging Face Hub:[/bold cyan] {repo_id} ({resolved_file})")
            with console.status("[bold green]Fetching remote safetensors header via HTTP Range Requests...[/bold green]"):
                try:
                    header_info = inspect_hf_header(repo_id, resolved_file, revision=resolved_rev)
                except Exception as e:
                    console.print(f"[bold red]Remote inspection failed:[/bold red] {e}")
                    raise typer.Exit(code=1)

            num_t = header_info.get("num_tensors", header_info.get("tensor_count", 0))
            tensors_dict = header_info.get("tensors", header_info.get("header", {}))

            table = Table(
                title=f"Hugging Face Remote Header: {repo_id} ({num_t} Tensors)",
                box=box.ROUNDED,
                header_style="bold magenta",
            )
            table.add_column("Tensor Name", style="cyan", overflow="ellipsis")
            table.add_column("Dtype", justify="center")
            table.add_column("Shape", justify="center")
            table.add_column("Data Offsets (Bytes)", justify="right")

            for t_name, t_info in list(tensors_dict.items())[:25]:
                table.add_row(
                    t_name,
                    t_info.get("dtype", "Unknown"),
                    str(t_info.get("shape", [])),
                    str(t_info.get("data_offsets", [])),
                )

            console.print(table)
            if num_t > 25:
                console.print(f"[dim]... and {num_t - 25} more tensors.[/dim]\n")

            console.print(
                Panel(
                    f"[bold green]Remote Header Verified:[/bold green] {num_t} tensor descriptors extracted without downloading weight payloads.\n"
                    f"Header Size: [cyan]{header_info['header_size_bytes']:,} bytes[/cyan].\n\n"
                    f"[bold yellow]To run full zero-copy static cryptanalysis (Shannon entropy & Benford's Law) on tensor weights, re-run with:[/bold yellow]\n"
                    f"  [bold green]aegis scan {model_target} --download[/bold green]",
                    title="Hugging Face Remote Inspection",
                    border_style="cyan",
                )
            )

            if json_output:
                with open(json_output, "w") as f:
                    json.dump(header_info, f, indent=2)
                console.print(f"[dim]Remote header report saved to: {json_output}[/dim]")
            return

        # Full download requested for deep cryptanalysis scan
        console.print(f"[bold cyan]Downloading remote Hugging Face model:[/bold cyan] {repo_id} ({resolved_file})")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Downloading model...", total=100)

            def dl_cb(downloaded: int, total: int):
                if total > 0:
                    progress.update(task, completed=int((downloaded / total) * 100))

            try:
                model_path = download_hf_model(
                    repo_id,
                    resolved_file,
                    revision=resolved_rev,
                    progress_callback=dl_cb,
                )
            except Exception as e:
                console.print(f"[bold red]Download failed:[/bold red] {e}")
                raise typer.Exit(code=1)

        console.print(f"[green]Download completed. Cached at:[/green] {model_path}\n")
    else:
        model_path = Path(model_target)
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
        is_critical = any(
            res.entropy > entropy_threshold and res.benford_mad > benford_threshold
            for res in results
        ) or suspicious_count > 1
        badge = get_risk_badge("CRITICAL" if is_critical else "SUSPICIOUS")
        console.print(
            Panel(
                f"{badge}\n\n"
                f"[bold red]Adversarial Threat Detected:[/bold red] {suspicious_count} suspicious tensor(s) found!\n"
                "Tensors exhibit anomalous Shannon entropy or Benford's Law deviations indicative "
                "of hidden steganographic malware or injected cryptographic payloads.",
                title="🚨 Static Steganography Scan Alert",
                border_style="red" if is_critical else "yellow",
            )
        )
    else:
        badge = get_risk_badge("CLEAN")
        console.print(
            Panel(
                f"{badge}\n\n"
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

    if sarif_output:
        export_scan_sarif(model_path, results, sarif_output)
        console.print(f"[dim]SARIF 2.1.0 log saved to: {sarif_output}[/dim]")


@app.command(name="fuzz")
def fuzz_command(
    model_path: Path = typer.Argument(
        ...,
        help="Path to PyTorch model file (.pt, .pth, or TorchScript) to fuzz.",
    ),
    spike_threshold: float = typer.Option(
        4.0,
        "--spike-threshold",
        "-t",
        help="Activation spike threshold (ratio of fuzzed L_infinity to baseline L_infinity).",
    ),
    iterations: int = typer.Option(
        50,
        "--iterations",
        "-n",
        help="Number of perturbation fuzzing iterations to execute.",
    ),
    input_shape: str = typer.Option(
        "1,10",
        "--input-shape",
        "-s",
        help="Comma-delimited input tensor dimensions (e.g. '1,10' or '1,3,224,224').",
    ),
    json_output: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Save full dynamic Trojan fuzzing report to a JSON file.",
    ),
    sarif_output: Optional[Path] = typer.Option(
        None,
        "--output-sarif",
        help="Save dynamic Trojan fuzzing report in SARIF 2.1.0 format for GitHub Code Scanning.",
    ),
):
    """Run dynamic Trojan backdoor fuzzing on a PyTorch model."""
    print_banner()

    # 1. Validate model file existence
    if not model_path.exists():
        console.print(
            Panel(
                f"[bold red]Model file not found:[/bold red] {model_path.resolve()}",
                title="❌ File Not Found",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # 2. Safetensors file guidance
    if model_path.suffix.lower() == ".safetensors":
        console.print(
            Panel(
                "[bold yellow]Safetensors weight container detected.[/bold yellow]\n\n"
                "`.safetensors` files contain raw tensor weights without an executable PyTorch computation graph.\n"
                "To perform dynamic activation fuzzing, provide a serialized PyTorch model (`.pt`, `.pth`) "
                "or compiled TorchScript model.\n\n"
                "Tip: To inspect steganography inside `.safetensors`, use:\n"
                f"  [bold green]aegis scan {model_path}[/bold green]",
                title="ℹ️ Unsupported File Format for Dynamic Fuzzing",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    # 3. Check PyTorch availability
    if not TORCH_AVAILABLE:
        console.print(
            Panel(
                "[bold red]PyTorch Engine is not installed![/bold red]\n\n"
                "Dynamic Trojan fuzzing requires PyTorch to hook forward activations.\n"
                "Please install PyTorch in your environment:\n"
                "  [bold green]pip install torch[/bold green]\n"
                "or install with optional dependencies:\n"
                "  [bold green]pip install -e .[fuzzer][/bold green]",
                title="⚠️ PyTorch Engine Missing",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    import torch

    # 4. Validate threshold and iterations
    if spike_threshold <= 0:
        console.print("[bold red]Error:[/bold red] `--spike-threshold` must be a positive number.")
        raise typer.Exit(code=1)
    if iterations <= 0:
        console.print("[bold red]Error:[/bold red] `--iterations` must be a positive integer.")
        raise typer.Exit(code=1)

    # 5. Parse input tensor shape
    try:
        shape = tuple(int(dim.strip()) for dim in input_shape.split(",") if dim.strip())
        if not shape or any(d <= 0 for d in shape):
            raise ValueError("All dimensions must be positive integers.")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Invalid input shape format:[/bold red] '{input_shape}'\n\n"
                "Shape must be comma-separated positive integers, for example:\n"
                "  --input-shape 1,10\n"
                "  --input-shape 1,3,224,224\n"
                f"[dim]Details: {e}[/dim]",
                title="❌ Invalid Argument",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # 6. Load model
    console.print(f"[bold]Target Model:[/bold] [cyan]{model_path.resolve()}[/cyan]")
    console.print(f"[dim]Fuzzing Config: {iterations} iterations | Spike Threshold: {spike_threshold:.1f}x | Input Shape: {shape}[/dim]\n")

    model = None
    with console.status("[bold green]Loading model architecture and weights...[/bold green]"):
        try:
            try:
                model = torch.jit.load(str(model_path), map_location="cpu")
            except Exception:
                model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Failed to load model file:[/bold red]\n{e}\n\n"
                    "[yellow]Ensure the file contains an executable PyTorch nn.Module or TorchScript model.[/yellow]",
                    title="❌ Model Load Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    if hasattr(model, "eval"):
        model.eval()

    # Determine target device and floating-point precision from model parameters
    target_device = torch.device("cpu")
    target_dtype = torch.float32
    if hasattr(model, "parameters"):
        try:
            first_param = next(model.parameters())
            target_device = first_param.device
            if first_param.is_floating_point():
                target_dtype = first_param.dtype
        except (StopIteration, Exception):
            pass

    # 7. Initialize Fuzzer
    fuzzer = DynamicTrojanFuzzer(model, spike_threshold=spike_threshold)
    baseline_input = torch.zeros(shape, dtype=target_dtype, device=target_device)

    def input_generator(step: int) -> torch.Tensor:
        scale = 1.0 + (step % 5) * 0.5
        noise = torch.randn(shape, dtype=torch.float32) * scale
        return noise.to(dtype=target_dtype, device=target_device)

    # 8. Execute Fuzzing with Live Rich Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fuzzing activations ({iterations} iterations)...", total=iterations)

        def on_step_progress(curr_iter: int, total_iter: int, top_layer: str, max_val: float):
            desc = f"Fuzzing [{curr_iter}/{total_iter}]"
            if top_layer:
                desc += f" [dim](Peak: {top_layer} = {max_val:.2f})[/dim]"
            progress.update(task, completed=curr_iter, description=desc)

        try:
            report = fuzzer.run_fuzzing(
                sample_inputs_generator=input_generator,
                baseline_input=baseline_input,
                num_iterations=iterations,
                progress_callback=on_step_progress,
            )
        except Exception as e:
            console.print(f"[bold red]Dynamic fuzzing failed during execution:[/bold red] {e}")
            raise typer.Exit(code=1)

    # 9. Aggregate per-layer statistics
    layer_peaks: dict[str, float] = {}
    for r in report.iteration_reports:
        for lname, stat in r.layer_stats.items():
            if lname not in layer_peaks or stat.l_inf_norm > layer_peaks[lname]:
                layer_peaks[lname] = stat.l_inf_norm

    overall_peak_layer = ""
    highest_layer_activation = -1.0
    for lname, peak_val in layer_peaks.items():
        if peak_val > highest_layer_activation:
            highest_layer_activation = peak_val
            overall_peak_layer = lname

    # 10. Render Results Table
    table = Table(
        title=f"Aegis-Tensor Dynamic Introspection Report ({len(layer_peaks)} Layers Audited)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Layer / Sub-Network", style="cyan", overflow="ellipsis")
    table.add_column("Peak Fuzzed L_inf", justify="right")
    table.add_column("Spike Ratio", justify="right")
    table.add_column("Layer Status", justify="center")

    base_val = max(report.baseline_l_inf, 1e-6)
    for lname in sorted(layer_peaks.keys()):
        peak_val = layer_peaks[lname]
        ratio = peak_val / base_val
        is_susp = lname in report.suspicious_layers
        is_peak = (lname == overall_peak_layer)

        if is_susp:
            status_str = "[bold red]SPIKE DETECTED[/bold red]"
            ratio_str = f"[bold red]{ratio:.2f}x[/bold red]"
        else:
            status_str = "[bold green]NOMINAL[/bold green]"
            ratio_str = f"[white]{ratio:.2f}x[/white]"

        layer_label = lname
        if is_peak:
            layer_label += " [bold magenta]★ PEAK[/bold magenta]"

        table.add_row(
            layer_label,
            f"{peak_val:.4f}",
            ratio_str,
            status_str,
        )

    console.print(table)

    # 11. Render Executive Summary & Risk Badge
    if report.suspected_trojan:
        is_critical = report.spike_ratio >= (spike_threshold * 2.0) or len(report.suspicious_layers) > 1
        badge = get_risk_badge("CRITICAL" if is_critical else "SUSPICIOUS")
        summary_text = (
            f"{badge}\n\n"
            f"• [bold red]Threat Alert:[/bold red] Internal activations exhibited anomalous spikes under perturbation.\n"
            f"• [bold]Peak Activation Layer:[/bold] [cyan]{overall_peak_layer or 'N/A'}[/cyan] ({max(highest_layer_activation, 0.0):.4f})\n"
            f"• [bold]Baseline L_inf:[/bold] {report.baseline_l_inf:.4f}  ──▶  [bold]Peak Fuzzed L_inf:[/bold] {report.peak_fuzzed_l_inf:.4f}\n"
            f"• [bold]Max Spike Multiplier:[/bold] [bold red]{report.spike_ratio:.2f}x[/bold red] (Threshold: {spike_threshold:.1f}x)\n"
            f"• [bold]Flagged Layers ({len(report.suspicious_layers)}):[/bold] {', '.join(report.suspicious_layers) if report.suspicious_layers else 'None'}\n\n"
            f"[dim yellow]Advisory: Disproportionate activation surges under random perturbations indicate dormant Sleeper Agent circuits.[/dim yellow]"
        )
        console.print(
            Panel(
                summary_text,
                title="🚨 Adversarial Trojan Trigger Detected",
                border_style="red" if is_critical else "yellow",
            )
        )
    else:
        badge = get_risk_badge("CLEAN")
        summary_text = (
            f"{badge}\n\n"
            f"• [bold green]Model Integrity Verified:[/bold green] No activation spikes exceeded the {spike_threshold:.1f}x threshold.\n"
            f"• [bold]Peak Monitored Layer:[/bold] [cyan]{overall_peak_layer or 'N/A'}[/cyan]\n"
            f"• [bold]Baseline L_inf:[/bold] {report.baseline_l_inf:.4f}  ──▶  [bold]Peak Fuzzed L_inf:[/bold] {report.peak_fuzzed_l_inf:.4f}\n"
            f"• [bold]Max Spike Multiplier:[/bold] [bold green]{report.spike_ratio:.2f}x[/bold green]\n"
            f"• [bold]Total Fuzzing Iterations:[/bold] {iterations}\n\n"
            f"[dim green]Advisory: Dynamic activations remain bounded within normal Gaussian envelopes.[/dim green]"
        )
        console.print(
            Panel(
                summary_text,
                title="✅ Dynamic Integrity Nominal",
                border_style="green",
            )
        )

    # 12. Optional JSON Telemetry Export
    if json_output:
        report_json = {
            "model_path": str(model_path.resolve()),
            "model_name": report.model_name,
            "num_fuzz_samples": report.num_fuzz_samples,
            "baseline_l_inf": report.baseline_l_inf,
            "peak_fuzzed_l_inf": report.peak_fuzzed_l_inf,
            "spike_ratio": report.spike_ratio,
            "spike_threshold": spike_threshold,
            "suspected_trojan": report.suspected_trojan,
            "suspicious_layers": report.suspicious_layers,
            "peak_layer": overall_peak_layer,
            "layer_peaks": layer_peaks,
        }
        with open(json_output, "w") as f:
            json.dump(report_json, f, indent=2)
        console.print(f"[dim]Fuzzing report saved to: {json_output}[/dim]")

    if sarif_output:
        export_fuzz_sarif(model_path, report, sarif_output)
        console.print(f"[dim]SARIF 2.1.0 log saved to: {sarif_output}[/dim]")


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
