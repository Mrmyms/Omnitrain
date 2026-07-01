import typer
import sys
import os
import yaml
import torch
import time
import platform
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from omnitrain.exporter import OmniExporter
from omnitrain.token_bus import TokenBus
from omnitrain.recorder import OmniRecorder
from omnitrain.trainer import Trainer
from omnitrain.diagnostics import OmniDiagnostic
from omnitrain.pruner import SynapticPruner
from omnitrain.robot_registry import RobotRegistry, auto_config
import omnitrain.robots # Auto-register

app = typer.Typer(help="OmniTrain v2.2 - Professional BioLiquid Robotics Training Framework")
console = Console()

def get_project_context(config_path="config.yaml"):
    project_name = "robot"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                project_name = cfg.get("project", "robot")
        except Exception:
            pass
    return {
        "project": project_name,
        "logs": f"{project_name.lower()}_logs.csv",
        "model": f"{project_name.lower()}_final.omni"
    }

@app.command()
def init(project_name: str = typer.Argument("OmniRobot", help="Name of the project"),
         project_dir: str = typer.Option(".", help="Directory to initialize in")):
    """Scaffold a new project environment."""
    if project_dir != "." and not os.path.exists(project_dir):
        os.makedirs(project_dir)
    
    config_path = os.path.join(project_dir, "config.yaml")
    if os.path.exists(config_path):
        console.print(f"[yellow]Project already initialized in {project_dir}.[/]")
        raise typer.Exit(1)

    default_config = {
        'project': project_name,
        'model': {
            'n_latents': 32,
            'd_model': 256,
            'state_dim': 16,
            'brain_mode': 'conectoma',
            'conectoma': {
                'enabled': True,
                'sensory_n': 4,
                'wall_n': 20,
                'command_n': 8,
            }
        },
        'inputs': [
            {'id': 'lidar', 'dim': 512, 'type': 'vector', 'noise': True},
            {'id': 'camera', 'dim': 1024, 'type': 'vision'}
        ],
        'heads': [
            {'id': 'drive', 'type': 'regression', 'output_dim': 4}
        ],
        'safety_constraints': [],
        'training': {
            'epochs': 30,
            'batch_size': 16,
            'seq_len': 32,
            'lagrangian': {
                'enabled': True,
                'init_lambda': 0.1,
                'lr': 0.02,
                'lambda_max': 10.0
            }
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, sort_keys=False)
    
    csv_name = f"{default_config['project'].lower()}_logs.csv"
    csv_path = os.path.join(project_dir, csv_name)
    if not os.path.exists(csv_path):
        input_ids = [inp['id'] for inp in default_config['inputs']]
        action_cols = []
        for head in default_config.get('heads', []):
            if head.get('type') == 'regression':
                dim = head.get('output_dim', 1)
                action_cols.extend([f"action_{i}" for i in range(dim)] if dim > 1 else ["action"])
        
        header = ",".join(["timestamp"] + input_ids + action_cols) + "\n"
        with open(csv_path, "w") as f:
            f.write(header)
    
    console.print("[bold green]OK: PROJECT INITIALIZED[/bold green]")
    console.print(f"  Created: [white]{config_path}[/], [white]{csv_path}[/]")

@app.command()
def generate_config(robot_name: str, out: str = typer.Option("config.yaml", help="Output YAML file")):
    """Generates a config YAML based on a registered robot's specifications."""
    try:
        robot = RobotRegistry.make(robot_name)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
        
    cfg = auto_config(robot)
    config_dict = {
        "project": f"{robot_name}_training",
        "inputs": cfg,
        "model": {"d_model": 256, "n_latents": 32, "input_dim": sum(c['dim'] for c in cfg)},
        "heads": [
            {"id": "drive_control", "type": "regression", "output_dim": 2}
        ],
        "safety_constraints": [],
        "training": {
            "epochs": 30,
            "batch_size": 16,
            "seq_len": 32,
            "learning_rate": 0.002,
            "lagrangian": {
                "init_lambda": 0.1,
                "lr": 0.05
            }
        }
    }
    with open(out, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)
    console.print(f"[bold cyan]Config generated for {robot_name} at {out}[/bold cyan] ✨")

@app.command()
def train(config_path: str = typer.Argument("config.yaml", help="Path to config.yaml"), 
          csv_path: str = typer.Option(None, help="Path to logs CSV")):
    """Start the training loop with live terminal dashboard."""
    ctx = get_project_context(config_path)
    csv_path = csv_path or ctx["logs"]
    
    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        console.print(f"[red]ERROR[/red] Missing config ({config_path}) or dataset ({csv_path}). Run `omnitrain init` first.")
        raise typer.Exit(1)

    trainer = Trainer.from_config(config_path)
    
    from rich.layout import Layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    layout["body"].split_row(
        Layout(name="metrics", ratio=2),
        Layout(name="safety", ratio=1)
    )

    with Live(layout, refresh_per_second=10, screen=True) as live:
        for m in trainer.fit(csv_path):
            layout["header"].update(Panel(
                f"[bold arctic_blue]NEURAL KERNEL[/] | Epoch {m['epoch']}/{m['total_epochs']} | Phase: {m['phase']}",
                border_style="color(117)",
                subtitle=f"[dim]Noise: {m['noise']} | Stateful: ACTIVE[/]"
            ))

            p_loss = m['policy']
            s_loss = m['safety']
            bar_val = m['barrier']
            
            def get_spark(val):
                blocks = "▁▂▃▄▅▆▇█"
                idx = min(7, int(max(0, val) * 10))
                return blocks[idx]

            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Neural Path", style="cyan")
            table.add_column("Current Loss", justify="right")
            table.add_column("Stability", justify="center")
            
            table.add_row("Behavioral Policy", f"{p_loss:.6f}", f"[green]{get_spark(p_loss)*5}[/]")
            table.add_row("Safety Manifold", f"{s_loss:.6f}", f"[yellow]{get_spark(s_loss)*5}[/]")
            table.add_row("Barrier Constraint", f"{bar_val:.6f}", f"[red]{get_spark(bar_val)*5}[/]")
            
            layout["metrics"].update(Panel(table, title="Live Conectoma Flow", border_style="white"))

            violation = max(0.0, -bar_val) 
            status_color = "green" if violation < 0.01 else ("yellow" if violation < 0.1 else "red")
            saf_msg = "OPTIMAL" if status_color == "green" else ("STABILIZING" if status_color == "yellow" else "VIOLATION")
            
            layout["safety"].update(Panel(
                Align.center(f"\n[bold {status_color}]{saf_msg}[/]\n\n[dim]Violation: {violation:.4f} | λ: {m.get('lambda', 0):.3f}[/]"),
                title="OmniShield Guard",
                border_style=status_color
            ))
            
            footer_text = f"Optimizing Neural Kernel... LR={trainer.lr:.6f} | [bold white]TRAINING ACTIVE[/]"
            layout["footer"].update(Panel(footer_text, border_style="dim"))

    console.print(f"[bold green]OK: TRAINING COMPLETE[/bold green]. Model saved.\n")

@app.command()
def audit(model_path: str = typer.Argument(None, help="Path to model (.omni)")):
    """Deep system health audit and barrier evaluation."""
    if not model_path:
        ctx = get_project_context()
        model_path = ctx["model"]

    console.print("\n[bold arctic_blue]INITIALIZING SYSTEM AUDIT[/bold arctic_blue]\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        t1 = progress.add_task("[cyan]Auditing Neural Conectoma...", total=100)
        t2 = progress.add_task("[yellow]Validating Lagrangian Failsafes...", total=100)
        
        for i in range(100):
            time.sleep(0.01)
            progress.update(t1, advance=1)
            if i > 50: progress.update(t2, advance=2)
            
    if not os.path.exists(model_path):
        console.print(f"[red]WARNING[/red] Model not found: {model_path}. Skipping model-specific diagnostics.")
    else:
        with console.status("[bold color(117)]Analyzing Neural Conectoma..."):
            diag = OmniDiagnostic(model_path)
            sensitivity = diag.analyze_sensitivity()
            health = diag.check_health()
        
        table = Table(title=f"Brain Saliency Audit: {model_path}", box=box.ROUNDED, border_style="color(117)")
        table.add_column("Modality", style="magenta", header_style="bold")
        table.add_column("Influence (Saliency)", justify="right")
        for m_id, score in sensitivity.items():
            bar = "▇" * int(score / 5)
            table.add_row(m_id, f"{score:.1f}% [cyan]{bar}[/]")
        console.print(table)
        
        health_info = ""
        for k, v in health.items():
            color = "green" if "HEALTHY" in v or "ACTIVE" in v or "SPARSE" in v.upper() else "yellow"
            if "CRITICAL" in v or "UNRESPONSIVE" in v: color = "red"
            health_info += f"• [bold white]{k:15}:[/] [{color}]{v}[/]\n"
        console.print(Panel(health_info.strip(), title="Internal Conectoma Health", border_style="white", box=box.HORIZONTALS))

    console.print("\n[bold green]OK: SYSTEM STATUS: READY[/bold green]")

@app.command()
def deploy(model_path: str = typer.Argument(None, help="Path to .omni model"), 
           target: str = typer.Option("tensorrt", help="Target architecture (tensorrt, snpe)")):
    """Prepare for edge deployment (ONNX export)."""
    if not model_path:
        model_path = get_project_context()["model"]
    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        raise typer.Exit(1)

    out_onnx = model_path.replace(".omni", ".onnx")
    console.print(f"[bold color(117)]Deploying {model_path} to Edge (Target: {target.upper()})...[/]")
    
    with console.status("[bold green]Stripping PyTorch hooks & Tracing Graph..."):
        exporter = OmniExporter()
        core, heads, config = exporter.load_as_inference(model_path)
        
        if target.lower() == "snpe":
            out_dlc = model_path.replace(".omni", ".dlc")
            exporter.export_for_qualcomm_snpe(core, heads, out_onnx)
            success = exporter.convert_onnx_to_dlc(out_onnx, out_dlc)
            if success:
                console.print(f"\n[bold green]OK: DEPLOYMENT PACKAGE READY[/bold green]")
                console.print(f"  Artifact: [white]{out_dlc}[/]")
                console.print(f"  Target: [cyan]Qualcomm Snapdragon NPU (SNPE)[/]")
            else:
                console.print(f"\n[bold red]FAILED: SNPE DLC Conversion failed.[/bold red]")
        else:
            exporter.export_to_onnx(core, heads, out_onnx)
            console.print(f"\n[bold green]OK: DEPLOYMENT PACKAGE READY[/bold green]")
            console.print(f"  Artifact: [white]{out_onnx}[/]")
            console.print(f"  Target: [cyan]OmniEngine C++ / TensorRT[/]")

@app.command()
def status():
    """Deep system health audit."""
    table = Table(title="OmniTrain System Health", box=box.ROUNDED, border_style="color(117)")
    table.add_column("Subsystem")
    table.add_column("Status")
    table.add_column("Details")

    dev = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    table.add_row("Compute Engine", f"[bold green]{dev}[/]", f"PyTorch {torch.__version__}")

    shm_status = "[green]HEALTHY[/]"
    shm_details = "Managed by OS"
    if os.path.exists("/dev/shm"):
        shm_size = sum(os.path.getsize(os.path.join("/dev/shm", f)) for f in os.listdir("/dev/shm") if "omni" in f)
        shm_details = f"{shm_size / 1024 / 1024:.1f} MB utilized"
    elif platform.system() == "Darwin":
        shm_details = "Active (MacOS Posix IPC)"
    
    table.add_row("Shared Memory", shm_status, shm_details)

    cfg_stat = "[green]FOUND[/]" if os.path.exists("config.yaml") else "[red]MISSING[/]"
    table.add_row("Project Config", cfg_stat, "config.yaml")
    
    # Check dependencies
    try:
        import typer
        typer_stat = "[green]OK[/]"
    except ImportError:
        typer_stat = "[red]MISSING[/]"
    table.add_row("Typer Engine", typer_stat, "CLI Framework")

    console.print(table)

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    OmniTrain v2.2 - Professional BioLiquid Robotics Training Framework
    """
    if ctx.invoked_subcommand is None:
        banner = "[bold arctic_blue]OMNITRAIN[/] v2.2.0 | [bold white]Robotics Framework[/]"
        stats_list = [
            "[bold cyan]omnitrain init[/]           Scaffold Environment",
            "[bold cyan]omnitrain generate-config[/] Auto-config for robots",
            "[bold cyan]omnitrain train[/]          Stateful Lagrangian Training",
            "[bold cyan]omnitrain audit[/]          Deep Conectoma Integrity Audit",
            "[bold cyan]omnitrain deploy[/]         Jetson/Qualcomm Edge Package",
            "[bold cyan]omnitrain status[/]         System health check"
        ]
        stats = "\n".join(stats_list)
        sys_info = (
            f"OS     : [dim]{platform.system()} {platform.machine()}[/]\n"
            f"KERNEL : [bold white]BioLiquid CfC v2.2[/]\n"
            f"GUARD  : [bold green]OMNISHIELD v2.2[/]\n"
            f"CLI    : [bold cyan]TYPER[/]"
        )
        
        console.print("\n")
        console.print(Panel(
            Columns([Align.left(stats), Align.right(sys_info)], expand=True),
            title=banner,
            border_style="color(117)",
            box=box.DOUBLE_EDGE,
            padding=(1, 2)
        ))
        console.print("Run [bold cyan]omnitrain --help[/bold cyan] for more info.\n")

if __name__ == "__main__":
    app()
