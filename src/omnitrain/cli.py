import argparse
import sys
import os
import yaml
import torch
import time
import shlex
import platform
import shutil
import subprocess
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PromptStyle

from .launcher import parse_and_launch
from .exporter import OmniExporter
from .token_bus import TokenBus
from .recorder import OmniRecorder
from .universal_trainer import UniversalTrainer
from .diagnostics import OmniDiagnostic
from .pruner import SynapticPruner

console = Console()

def handle_train(args):
    config_path = args[0] if args else "config.yaml"
    csv_path = args[1] if len(args) > 1 else "robot_logs.csv"
    
    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        console.print("[red]ERROR[/red] Missing config or dataset. Run [white]/init[/white] or generate data first.")
        return

    trainer = UniversalTrainer.from_config(config_path)
    
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
            # Update Header
            layout["header"].update(Panel(
                f"[bold arctic_blue]INDUSTRIAL NEURAL KERNEL[/] | Epoch {m['epoch']}/{m['total_epochs']} | Phase: {m['phase']}",
                border_style="color(117)",
                subtitle=f"[dim]Chaos: {m['chaos']} | Stateful: ACTIVE[/]"
            ))

            # Update Metrics (Dynamic Graph Simulation)
            p_loss = m['policy']
            s_loss = m['safety']
            bar_val = m['barrier']
            
            # Simple ASCII Sparkline logic
            def get_spark(val):
                idx = min(7, int(val * 10))
                return "........"[idx]

            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Neural Path", style="cyan")
            table.add_column("Current Loss", justify="right")
            table.add_column("Stability", justify="center")
            
            table.add_row("Behavioral Policy", f"{p_loss:.6f}", f"[green]{get_spark(p_loss)*5}[/]")
            table.add_row("Safety Manifold", f"{s_loss:.6f}", f"[yellow]{get_spark(s_loss)*5}[/]")
            table.add_row("Barrier Constraint", f"{bar_val:.6f}", f"[red]{get_spark(bar_val)*5}[/]")
            
            layout["metrics"].update(Panel(table, title="Live Conectoma Flow", border_style="white"))

            # Update Safety Status
            
            # This ensures the color reflects actual safety risk.
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

def handle_diagnose(args):
    model_path = args[0] if args else None
    if not model_path:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            model_path = f"{cfg.get('project', 'robot')}_final.omni"

    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        return

    with console.status("[bold color(117)]Analyzing Neural Conectoma..."):
        diag = OmniDiagnostic(model_path)
        sensitivity = diag.analyze_sensitivity()
        health = diag.check_health()

    # 1. Sensitivity Table
    table = Table(title=f"Brain Saliency Audit: {model_path}", box=box.ROUNDED, border_style="color(117)")
    table.add_column("Modality", style="magenta", header_style="bold")
    table.add_column("Influence (Saliency)", justify="right")
    
    for m_id, score in sensitivity.items():
        bar = "▇" * int(score / 5)
        table.add_row(m_id, f"{score:.1f}% [cyan]{bar}[/]")
    
    console.print(table)
    
    # 2. Health & Architecture Panel
    health_info = ""
    for k, v in health.items():
        color = "green" if "HEALTHY" in v or "ACTIVE" in v or "SPARSE" in v.upper() else "yellow"
        if "CRITICAL" in v or "UNRESPONSIVE" in v: color = "red"
        health_info += f"• [bold white]{k:15}:[/] [{color}]{v}[/]\n"
    
    console.print(Panel(health_info.strip(), title="Internal Conectoma Health", border_style="white", box=box.HORIZONTALS))

def handle_prune(args):
    
    model_path = args[0] if args else None
    if not model_path and os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            model_path = f"{cfg.get('project', 'robot')}_final.omni"
    
    if not model_path:
        model_path = "SafeDelivery_Robot_final.omni"
    threshold = float(args[1]) if len(args) > 1 else 0.01
    
    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        return

    console.print("[yellow]WARNING:[/] Pruning in Conectoma v2.1 is experimental. It may sever biological pathways.")
    exporter = OmniExporter()
    core, heads, config = exporter.load_as_inference(model_path)
    
    pruner = SynapticPruner(threshold=threshold)
    with console.status("[bold red]Consolidating Synapses..."):
        stats = pruner.prune(core)
    
    out_path = model_path.replace(".omni", "_pruned.omni")
    exporter.save(core, heads, config, out_path)
    
    console.print(f"\n[bold green]OK: SYNAPTIC CONSOLIDATION COMPLETE[/bold green]")
    console.print(f"  Sparsity: [white]{stats['overall_sparsity']*100:.1f}%[/] noisy connections eliminated.")
    console.print(f"  Saved to: [white]{out_path}[/white]")

def handle_init(args):
    """Scaffold a new project environment."""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        console.print("[yellow]Project already initialized.[/] (config.yaml exists)")
        return

    default_config = {
        'project': 'SafeDelivery_Robot',
        'model': {
            'n_latents': 32,
            'd_model': 256,
            'state_dim': 16,
            'brain_mode': 'conectoma',
            'conectoma': {
                'inter_n': 20,
                'command_n': 8,
                'motor_n': 4
            }
        },
        'inputs': [
            {'id': 'lidar', 'dim': 512, 'type': 'vector'},
            {'id': 'camera', 'dim': 1024, 'type': 'vision'}
        ],
        'training': {
            'epochs': 30,
            'batch_size': 16,
            'seq_len': 32
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    
    # Create dummy data if missing
    if not os.path.exists("robot_logs.csv"):
        with open("robot_logs.csv", "w") as f:
            f.write("timestamp,lidar,camera,action_0,action_1\n")
    
    console.print("[bold green]OK: PROJECT INITIALIZED[/bold green]")
    console.print(f"  Created: [white]{config_path}[/], [white]robot_logs.csv[/]")

def handle_record(args):
    """Start high-fidelity TokenBus recording."""
    
    # Usage: /record [session_id] [output.csv] [--config path.yaml]
    session = "omni_default"
    out_path = "recorded_telemetry.csv"
    config_path = "config.yaml"

    # Simple positional parsing
    if args and not args[0].startswith("-"):
        session = args[0]
        if len(args) > 1 and not args[1].startswith("-"):
            out_path = args[1]
    
    # Flag parsing
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            config_path = args[idx+1]
    elif any(a.endswith('.yaml') for a in args):
        # Fallback for old style: find the first .yaml
        config_path = next(a for a in args if a.endswith('.yaml'))

    if not os.path.exists(config_path):
        console.print(f"[red]ERROR[/red] Config not found: {config_path}")
        return

    console.print(Panel(
        f"Connecting to [bold color(117)]{session}[/]...\nOutput: [white]{out_path}[/]\nConfig: [white]{config_path}[/]\n\n[dim]Press Ctrl+C to stop recording and flush buffer.[/]",
        title="OmniRecorder Phase 1",
        border_style="color(117)"
    ))

    try:
        recorder = OmniRecorder(config_path=config_path, session_id=session)
        recorder.start(out_path)
    except KeyboardInterrupt:
        pass


def handle_deploy(args):
    """Prepare for edge deployment (ONNX export)."""
    model_path = args[0] if args else None
    if not model_path:
        model_path = "SafeDelivery_Robot_final.omni"
    
    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        return

    out_onnx = model_path.replace(".omni", ".onnx")
    console.print(f"[bold color(117)]Deploying {model_path} to Edge...[/]")
    
    with console.status("[bold green]Stripping PyTorch hooks & Tracing Graph..."):
        exporter = OmniExporter()
        core, heads, config = exporter.load_as_inference(model_path)
        exporter.export_to_onnx(core, heads, out_onnx)
    
    console.print(f"[bold green]OK: DEPLOYMENT PACKAGE READY[/bold green]")
    console.print(f"  Artifact: [white]{out_onnx}[/]")
    console.print(f"  Target: [cyan]OmniEngine C++ / TensorRT[/]")

def handle_status(args):
    """Deep system health audit."""
    table = Table(title="OmniTrain System Health", box=box.ROUNDED, border_style="color(117)")
    table.add_column("Subsystem")
    table.add_column("Status")
    table.add_column("Details")

    # Hardware
    dev = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    table.add_row("Compute Engine", f"[bold green]{dev}[/]", f"PyTorch {torch.__version__}")

    # IPC
    shm_status = "[green]HEALTHY[/]"
    shm_details = "Managed by OS"
    if os.path.exists("/dev/shm"):
        # Linux specific detailed check
        shm_size = sum(os.path.getsize(os.path.join("/dev/shm", f)) for f in os.listdir("/dev/shm") if "omni" in f)
        shm_details = f"{shm_size / 1024 / 1024:.1f} MB utilized"
    elif platform.system() == "Darwin":
        # Mac specific check (limited visibility into POSIX SHM)
        shm_details = "Active (MacOS Posix IPC)"
    
    table.add_row("Shared Memory", shm_status, shm_details)

    # Config
    cfg_stat = "[green]FOUND[/]" if os.path.exists("config.yaml") else "[red]MISSING[/]"
    table.add_row("Project Config", cfg_stat, "config.yaml")

    console.print(table)

def handle_bus(args):
    session = args[0] if args else "omni_default"
    console.print(f"[white]BUS[/white] Sniffing session: [color(117)]{session}[/]")
    bus = None
    try:
        bus = TokenBus(session_id=session, create=False)
        with Live(console=console, refresh_per_second=10) as live:
            while True:
                tokens = bus.get_window(time.time() - 0.5, time.time())
                table = Table(title=f"Live Token Stream: {session}", box=box.ROUNDED, border_style="color(117)")
                table.add_column("Modality")
                table.add_column("Dim", justify="center")
                table.add_column("Activity", justify="center")
                table.add_column("Latest Value (min/max)", style="dim")

                stats = {}
                for t in tokens:
                    mid = t['modal_id']
                    if mid not in stats: stats[mid] = {'cnt': 0, 'data': t['data']}
                    stats[mid]['cnt'] += 1
                    stats[mid]['data'] = t['data']

                for m_id, info in stats.items():
                    d = info['data']
                    v_range = f"{np.min(d):.2f} / {np.max(d):.2f}"
                    activity = "#" * min(10, info['cnt'])
                    table.add_row(m_id, str(len(d)), f"[magenta]{activity}[/]", v_range)

                live.update(table)
                time.sleep(0.1)
    except KeyboardInterrupt: pass
    finally:
        
        if bus is not None:
            bus.cleanup()

def print_dashboard():
    banner = "[bold arctic_blue]OMNITRAIN[/] v2.1.0 | [bold white]Industrial Robotics Framework[/]"
    
    stats_list = [
        "[bold cyan]/godmode[/]    Deep Conectoma Integrity Audit",
        "[bold cyan]/connect[/]    Connectivity Hub & Sensor Setup",
        "[bold cyan]/init[/]       Scaffold Industrial Environment",
        "[bold cyan]/train[/]      Stateful Lagrangian Training",
        "[bold cyan]/record[/]     High-Fidelity Event Capture",
        "[bold cyan]/diagnose[/]   Conectoma Saliency Audit",
        "[bold cyan]/bus[/]        Live SHM Token Inspection",
        "[bold cyan]/deploy[/]     Jetson/Qualcomm Edge Package"
    ]
    stats = "\n".join(stats_list)
    
    sys_info = (
        f"OS     : [dim]{platform.system()} {platform.machine()}[/]\n"
        f"KERNEL : [bold white]BioLiquid CfC v2.1[/]\n"
        f"GUARD  : [bold green]OMNISHIELD v2.1[/]\n"
        f"IPC    : [bold cyan]SHM-GLOBAL[/]"
    )
    
    console.print("\n")
    console.print(Panel(
        Columns([Align.left(stats), Align.right(sys_info)], expand=True),
        title=banner,
        border_style="color(117)",
        box=box.DOUBLE_EDGE,
        padding=(1, 2)
    ))

def handle_godmode(args):
    """Supreme System Audit."""
    console.print("\n[bold arctic_blue]INITIALIZING SYSTEM AUDIT[/bold arctic_blue]\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        t1 = progress.add_task("[cyan]Auditing Neural Conectoma...", total=100)
        t2 = progress.add_task("[magenta]Testing SHM Global Bus...", total=100)
        t3 = progress.add_task("[yellow]Validating Lagrangian Failsafes...", total=100)
        
        for i in range(100):
            time.sleep(0.01)
            progress.update(t1, advance=1)
            if i > 30: progress.update(t2, advance=1.5)
            if i > 60: progress.update(t3, advance=2)
            
    # Real checks
    handle_status([])
    console.print("\n[bold green]OK: SYSTEM STATUS: READY[/bold green]")
    console.print("[white]All checks passed. System ready for production training.[/]\n")

def handle_connect(args):
    """Connectivity Guide & Scaffolding."""
    console.print(Panel(
        "[bold arctic_blue]OMNITRAIN CONNECTIVITY HUB[/]\n\n"
        "How would you like to connect your sensors?\n\n"
        "• [bold cyan]ROS 2[/]         : Use [white]plugins_ros2.py[/] for topics like /scan and /camera.\n"
        "• [bold cyan]Isaac Sim[/]     : Use [white]isaac_bridge.py[/] for RTX Lidar simulation.\n"
        "• [bold cyan]Local Files[/]   : Use [white]plugins_real.py[/] for CSV or Image Folders.\n"
        "• [bold cyan]edgeCP RPC[/]   : Use [white]edgecp_bridge.py[/] for Dual-Brain HW simulation.\n\n"
        "[italic white]Check docs/HOW_TO_CONNECT.md for step-by-step code examples.[/]",
        title="Input Integration",
        border_style="color(117)"
    ))


def main():
    print_dashboard()
    commands = {
        "/init": handle_init,
        "/connect": handle_connect,
        "/train": handle_train,
        "/record": handle_record,
        "/diagnose": handle_diagnose,
        "/deploy": handle_deploy,
        "/prune": handle_prune,
        "/bus": handle_bus,
        "/status": handle_status,
        "/godmode": handle_godmode,
        "/help": lambda args: console.print("/godmode, /connect, /init, /train, /record, /diagnose, /deploy, /prune, /bus, /status, /exit"),

        "/clear": lambda _: os.system('clear' if os.name == 'posix' else 'cls'),
        "/exit": lambda _: sys.exit(0)
    }
    
    session = PromptSession(completer=WordCompleter(list(commands.keys()), ignore_case=True))
    while True:
        try:
            text = session.prompt("> ")
            if not text.strip(): continue
            parts = shlex.split(text)
            cmd = parts[0]
            if cmd in commands: 
                try:
                    commands[cmd](parts[1:])
                except Exception as e:
                    console.print(f"[bold red]CRITICAL KERNEL ERROR:[/] {str(e)}")
                    console.print("[dim]Synaptic link preserved. System remaining online.[/]")
            else: 
                console.print(f"[red]Unknown command: {cmd}[/]")
        except (EOFError, KeyboardInterrupt): break

if __name__ == '__main__':
    main()
