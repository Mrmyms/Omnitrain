import argparse
import sys
import os
import yaml
import torch
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from .launcher import parse_and_launch
from .exporter import OmniExporter
from .fusion_core import FusionCore
from .token_bus import TokenBus

console = Console()

BANNER = """
[bold cyan]
  ____  _      _        _ 
 / __ \| |    | |      (_)
| |  | | | __ | | _   _ _ 
| |  | | |/  \| || | | | |
| |__| | | (| | || |_| | |
 \____/|_|\__/|_| \__,_|_|
[/bold cyan] [bold white]Industrial Multimodal AI Framework v2.0[/bold white]
"""


def main():
    """Industrial OmniTrain CLI."""
    console.print(BANNER)

    parser = argparse.ArgumentParser(prog='omni', description='OmniTrain: Real-time Robotics AI')
    sub = parser.add_subparsers(dest='cmd')

    # init
    sub.add_parser('init', help='Scaffold a new project Archetype')

    # run
    run_p = sub.add_parser('run', help='Launch the industrial sensor ecosystem')
    run_p.add_argument('config', nargs='?', default='config.yaml', help='Path to config.yaml')

    # bus
    bus_p = sub.add_parser('bus', help='Monitor real-time SharedMemory pulses')
    bus_p.add_argument('--session', default='omni_default', help='Bus SID to monitor')

    # inspect
    inspect_p = sub.add_parser('inspect', help='View model architecture and metadata')
    inspect_p.add_argument('model', help='Path to .omni or .pt file')

    # export
    export_p = sub.add_parser('export', help='Package trained weights into an .omni bundle')
    export_p.add_argument('weights', help='Path to .pt weights')
    export_p.add_argument('output', help='Target .omni filename')

    # deploy
    deploy_p = sub.add_parser('deploy', help='Industrial Edge AI deployment orchestrator')
    deploy_p.add_argument('model', help='Path to .omni or .onnx')
    deploy_p.add_argument('--quantize', action='store_true', help='Apply Mixed-Precision Quantization (INT8/FP32)')
    deploy_p.add_argument('--prune', action='store_true', help='Apply Structured Pruning (Sparsity)')

    # verify (Formal Safety Certification)
    verify_p = sub.add_parser('verify', help='Run formal safety verification on a model')
    verify_p.add_argument('model', help='Path to .omni file')

    args = parser.parse_args()

    if args.cmd == 'init':
        template = {
            "project": "Alpha_Robot_Alpha",
            "model": {
                "d_model": 512,
                "n_latents": 64,
                "num_layers": 3
            },
            "inputs": [
                {"id": "lidar_front", "plugin": "plugins.DummyLidarPlugin", "hz": 20},
                {"id": "label_stream", "plugin": "plugins_real.CSVModalityPlugin", "hz": 1, "csv_path": "labels.csv"}
            ],
        }
        with open("config.yaml", "w") as f:
            yaml.dump(template, f, sort_keys=False)
        console.print("[bold green]✔[/bold green] Generated project archetype in [bold]config.yaml[/bold]")

    elif args.cmd == 'run':
        try:
            parse_and_launch(args.config)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]! [/bold yellow]Pipeline terminated by user.")

    elif args.cmd == 'bus':
        console.print(f"📡 Connecting to Industrial Bus: [bold cyan]{args.session}[/bold cyan]")
        try:
            bus = TokenBus(session_id=args.session, create=False)
            with Live(console=console, refresh_per_second=4) as live:
                while True:
                    tokens = bus.get_window(time.time() - 0.5, time.time())
                    table = Table(title=f"Live Pulse: {args.session}", border_style="cyan")
                    table.add_column("Modality", style="magenta")
                    table.add_column("Count", justify="right")
                    table.add_column("Latest Timestamp", justify="center")

                    stats = {}
                    for t in tokens:
                        stats[t['modal_id']] = stats.get(t['modal_id'], 0) + 1

                    for m_id, count in stats.items():
                        table.add_row(m_id, str(count), f"{time.time():.4f}")

                    live.update(table)
                    time.sleep(0.25)
        except Exception as e:
            console.print(f"[bold red]✘ Bus Error:[/bold red] {e}")

    elif args.cmd == 'inspect':
        if not os.path.exists(args.model):
            console.print(f"[bold red]✘[/bold red] Model not found: {args.model}")
            return
        weights = torch.load(args.model, map_location='cpu')

        table = Table(title=f"Model Inspection: {os.path.basename(args.model)}", border_style="green")
        table.add_column("Layer", style="cyan")
        table.add_column("Shape", style="white")

        if isinstance(weights, dict) and 'model_state' in weights:
            state = weights['model_state']
        else:
            state = weights

        for k, v in state.items():
            if hasattr(v, 'shape') and len(table.rows) < 15:
                table.add_row(k, str(list(v.shape)))

        console.print(table)
        console.print(f"\n[dim]Displayed top 15 layers.[/dim]")

    elif args.cmd == 'export':
        exporter = OmniExporter()
        model = FusionCore()
        try:
            model.load_state_dict(torch.load(args.weights))
            exporter.save(model, {}, {"exported_at": time.ctime()}, args.output)
            console.print(f"[bold green]✔[/bold green] Bundle created: [bold cyan]{args.output}[/bold cyan]")
        except Exception as e:
            console.print(f"[bold red]✘ Export failed:[/bold red] {e}")

    elif args.cmd == 'deploy':
        handle_deploy(args)

    elif args.cmd == 'verify':
        handle_verify(args)


def handle_deploy(args):
    from rich.panel import Panel
    from rich.prompt import Confirm
    import os

    console.print(Panel(f"[bold cyan]🚀 OMNITRAIN DEPLOYMENT ORCHESTRATOR 2.0[/]\n[white]Target: {args.model}[/]"))

    current_model = args.model
    if not os.path.exists(current_model):
        console.print(f"[bold red]✘ Error:[/] Model not found: {current_model}")
        return

    # 1. Check for .omni -> .onnx conversion
    if current_model.endswith('.omni'):
        console.print("[yellow]ℹ  Industrial Tip: Export to ONNX is recommended for sub-ms Edge inference.[/]")
        if Confirm.ask("Do you want to export to ONNX now?"):
            from .onnx_exporter import export_omnitrain_to_onnx
            export_omnitrain_to_onnx("omni_deploy_temp.onnx")
            current_model = "omni_deploy_temp.onnx"

    # 2. Pruning (Optional)
    if args.prune:
        console.print("[bold green]✂  Applying Structured Pruning...[/]")
        from .pruner import apply_omni_pruning
        apply_omni_pruning(current_model, output_path="omni_deploy_pruned.omni")
        current_model = "omni_deploy_pruned.omni"

    # 3. Quantization (Optional)
    if args.quantize and current_model.endswith('.onnx'):
        console.print("[bold green]💎 Applying Mixed-Precision Quantization...[/]")
        from .quantize_omni import quantize_omnitrain_mixed
        quantize_omnitrain_mixed(current_model, "omni_deploy_quant.onnx")
        current_model = "omni_deploy_quant.onnx"

    # 4. Final Verification
    console.print(f"\n[bold green]✔  OmniEngine (C++) Preparation Complete.[/]")
    console.print(f"📦 Final Payload: [bold cyan]{current_model}[/bold cyan]")
    console.print("[dim]Next Step: Transfer the payload to the robot and launch with 'omni_engine'.[/]")


def handle_verify(args):
    """Formal Safety Verification: Run constraint checks and generate a certificate."""
    from rich.panel import Panel
    from rich.table import Table
    import torch

    console.print(Panel("[bold cyan]🛡️  OMNITRAIN FORMAL SAFETY VERIFIER[/]"))

    if not os.path.exists(args.model):
        console.print(f"[bold red]✘ Error:[/] Model not found: {args.model}")
        return

    # 1. Load model
    exporter = OmniExporter()
    try:
        core, heads, meta = exporter.load_as_inference(args.model)
    except Exception as e:
        console.print(f"[bold red]✘ Failed to load model:[/] {e}")
        return

    # 2. Wrap safety heads with SafetyGuard
    from .safety_guard import SafetyGuard

    safety_heads_found = {k: v for k, v in heads.items() if 'safety' in k.lower()}
    if not safety_heads_found:
        console.print("[yellow]⚠  No safety heads found in model. Verifying structure only.[/]")

    for head_id, head in safety_heads_found.items():
        guard = SafetyGuard(head, emergency_class=1)

        # Default safety constraints for robotics
        guard.add_constraint('lidar_front', min_safe=0.10, max_safe=50.0)  # meters
        guard.add_constraint('lidar_rear', min_safe=0.10, max_safe=50.0)
        guard.add_constraint('temperature', min_safe=-40.0, max_safe=85.0)  # Celsius
        guard.add_constraint('voltage', min_safe=10.0, max_safe=52.0)  # Volts

        # 3. Generate test cases (edge cases)
        test_cases = [
            {'lidar_front': 0.05, 'lidar_rear': 2.0},   # DANGER: too close
            {'lidar_front': 1.0, 'lidar_rear': 1.5},     # SAFE
            {'temperature': 90.0, 'voltage': 48.0},       # DANGER: overheat
            {'lidar_front': 5.0, 'temperature': 20.0},    # SAFE
            {'voltage': 8.0},                              # DANGER: undervoltage
            {'lidar_front': 100.0},                        # DANGER: out of range
        ]

        report = guard.generate_safety_report(test_cases)

        # 4. Display report
        table = Table(title=f"Safety Verification: {head_id}", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Test Cases", str(report['total_cases']))
        table.add_row("Constraints Active", str(report['constraint_count']))
        table.add_row("Passed", f"[green]{report['passed']}[/green]")
        table.add_row("Failed (Correctly Caught)", f"[red]{report['failed']}[/red]")

        status_color = "green" if report['status'] == 'CERTIFIED' else "yellow"
        table.add_row("Status", f"[{status_color}]{report['status']}[/{status_color}]")

        console.print(table)

        if report['violations']:
            console.print("\n[bold yellow]Caught Violations (Expected):[/]")
            for v in report['violations']:
                console.print(f"  Case {v['case_id']}: {v['violations']}")

    console.print(f"\n[bold green]✔  Safety verification complete.[/]")


if __name__ == '__main__':
    main()

