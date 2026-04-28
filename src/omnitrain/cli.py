import argparse
import sys
import os
import yaml
import torch
import time
import shlex
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PromptStyle

from .launcher import parse_and_launch
from .exporter import OmniExporter
from .fusion_core import FusionCore
from .token_bus import TokenBus
from .recorder import OmniRecorder
from .universal_trainer import UniversalTrainer
 
import shutil
import platform
import subprocess

console = Console()

def handle_train(args):
    config_path = args[0] if args else "config.yaml"
    if not os.path.exists(config_path):
        console.print(f"[red]ERROR[/red] Config not found: {config_path}")
        return
    
    csv_path = args[1] if len(args) > 1 else "robot_logs.csv"
    if not os.path.exists(csv_path):
        console.print(f"[red]ERROR[/red] Training data not found: {csv_path}")
        return

    try:
        trainer = UniversalTrainer.from_config(config_path)
        trainer.fit(csv_path, epochs=30)
    except Exception as e:
        console.print(f"[red]ERROR[/red] Training failed: {e}")

def print_dashboard():
    mascot = """
   .---.
  ( @ @ )
   )   ( 
  /|||||\\
  " " " "
    """
    
    # Dynamic Project Info
    project_name = "N/A"
    last_train = "Never"
    if os.path.exists("config.yaml"):
        try:
            with open("config.yaml", 'r') as f:
                cfg = yaml.safe_load(f)
                project_name = cfg.get('project', 'Unknown')
        except: pass

    left_content = Align.center(
        f"[bold color(117)]{mascot}[/]\n"
        f"[bold white]OMNITRAIN v1.0.0[/]\n"
        f"[dim]Project: [white]{project_name}[/]\n"
        "[dim]Safety: [green]Active[/]",
        vertical="middle"
    )
    
    left_panel = Panel(
        left_content,
        border_style="color(117)",
        box=box.ROUNDED,
        height=12
    )
    
    tips = (
        "[color(117)]Quick Launch Tips[/]\n"
        "• [blue]/init[/]   : Scaffolding\n"
        "• [blue]/status[/] : Health & Resource Monitor\n"
        "• [blue]/train[/]  : Curriculum Pipeline\n"
        "• [blue]/test[/]   : Safety Audit\n"
    )
    
    activity = (
        "\n[color(117)]Environment[/]\n"
        f"OS: [dim]{platform.system()} {platform.release()}[/]\n"
        f"Path: [dim]...{os.getcwd()[-25:]}[/]\n"
    )
    
    right_panel = Panel(
        tips + activity,
        border_style="color(117)",
        box=box.ROUNDED,
        height=12
    )
    
    console.print(Columns([left_panel, right_panel], expand=True))
    console.print("[dim]Type [white]/help[/white] for commands or [white]/exit[/white] to quit.[/]\n")

def handle_init(args):
    """Interactive Project Scaffolding"""
    from rich.prompt import Prompt, IntPrompt, Confirm
    
    console.print("\n[bold color(117)]--- OmniTrain Project Scaffolding ---[/]")
    
    name = Prompt.ask("Project Name", default="Alpha_Robot")
    type_choice = Prompt.ask("Architecture Template", choices=["Industrial Arm", "Autonomous Vehicle", "Custom"], default="Industrial Arm")
    
    # Core Dims
    d_model = IntPrompt.ask("Model Dimension (d_model)", default=512)
    n_latents = IntPrompt.ask("Latent Tokens (n_latents)", default=64)
    
    ros2_enabled = Confirm.ask("Enable ROS 2 Humble/Iron Integration?", default=False)

    # Structure
    folders = ["data", "models", "logs", "scripts"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    template = {
        "project": name,
        "template": type_choice,
        "model": {
            "d_model": d_model,
            "n_latents": n_latents,
            "num_layers": 3
        },
        "inputs": [
            {"id": "sensor_primary", "plugin": "plugins_real.CSVModalityPlugin", "hz": 10, "csv_path": "data/sample_sensor.csv"}
        ],
    }
    
    if ros2_enabled:
        template["inputs"].append({
            "id": "ros_telemetry", 
            "plugin": "plugins_ros2.ROS2ModalityPlugin", 
            "hz": 50, 
            "topic_name": "/robot/telemetry"
        })
    
    # Create sample data for immediate training test
    sample_data_path = os.path.join("data", "sample_sensor.csv")
    with open(sample_data_path, "w") as f:
        f.write("timestamp,lidar_front,action\n")
        for i in range(100):
            f.write(f"{time.time()+i*0.1},{0.5+i*0.01},1.0\n")

    with open("config.yaml", "w") as f:
        yaml.dump(template, f, sort_keys=False)
        
    console.print(f"\n[color(117)]SUCCESS[/] Project [white]{name}[/white] initialized.")
    console.print(f"[dim]Folders created: {', '.join(folders)}[/]")
    console.print("[dim]Sample data generated in data/sample_sensor.csv[/]")

def handle_run(args):
    config_path = args[0] if args else "config.yaml"
    try:
        parse_and_launch(config_path)
    except KeyboardInterrupt:
        console.print("\n[yellow]ABORT[/yellow] Pipeline terminated by user.")

def handle_bus(args):
    session = args[0] if args else "omni_default"
    console.print(f"[white]BUS[/white] Connecting to session: [color(117)]{session}[/]")
    try:
        bus = TokenBus(session_id=session, create=False)
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                tokens = bus.get_window(time.time() - 0.5, time.time())
                table = Table(title=f"Bus: {session}", border_style="white", box=None)
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
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]ERROR[/red] Bus failure: {e}")

def handle_inspect(args):
    if not args:
        console.print("[red]ERROR[/red] Please specify a model path.")
        return
    model_path = args[0]
    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        return
    weights = torch.load(model_path, map_location='cpu')

    table = Table(title=f"Inspect: {os.path.basename(model_path)}", border_style="white", box=None)
    table.add_column("Layer", style="cyan")
    table.add_column("Shape", style="white")

    state = weights.get('model_state', weights) if isinstance(weights, dict) else weights
    for i, (k, v) in enumerate(state.items()):
        if hasattr(v, 'shape') and i < 15:
            table.add_row(k, str(list(v.shape)))

    console.print(table)
    console.print(f"\n[dim]Displayed top 15 layers.[/dim]")

def handle_deploy(args):
    if not args:
        console.print("[red]ERROR[/red] Please specify a model path.")
        return
    
    from rich.prompt import Confirm
    current_model = args[0]
    if not os.path.exists(current_model):
        console.print(f"[red]ERROR[/red] Model not found: {current_model}")
        return

    console.print(f"[white]DEPLOY[/white] [color(117)]{current_model}[/]")
    
    if current_model.endswith('.omni'):
        console.print("[dim]INFO[/dim] Export to ONNX is recommended for edge inference.")
        if Confirm.ask("Do you want to export to ONNX now?"):
            from .onnx_exporter import export_omnitrain_to_onnx
            export_omnitrain_to_onnx("omni_deploy_temp.onnx")
            current_model = "omni_deploy_temp.onnx"

    console.print(f"[color(117)]DONE[/] Payload ready: [white]{current_model}[/white]")
    console.print("[dim]Next Step: Transfer the payload to the robot and launch with 'omni_engine'.[/]")

def handle_record(args):
    """Data Recorder: /record <config> [--output <file>] [--hz N] [--session ID]"""
    config_path = args[0] if args else "config.yaml"
    output_path = "robot_logs.csv"
    hz = 10.0
    session = "omni_default"

    i = 1
    while i < len(args):
        if args[i] == "--output" and i+1 < len(args):
            output_path = args[i+1]
            i += 2
        elif args[i] == "--hz" and i+1 < len(args):
            hz = float(args[i+1])
            i += 2
        elif args[i] == "--session" and i+1 < len(args):
            session = args[i+1]
            i += 2
        else:
            i += 1

    if not os.path.exists(config_path):
        console.print(f"[red]ERROR[/red] Config not found: {config_path}")
        return

    try:
        recorder = OmniRecorder(config_path, session_id=session)
        recorder.record(output_path, hz=hz)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]ERROR[/red] Recording failed: {e}")

def handle_verify(args):
    if not args:
        console.print("[red]ERROR[/red] Please specify a model path.")
        return
    model_path = args[0]
    console.print(f"[white]VERIFY[/white] [color(117)]{model_path}[/]")

    if not os.path.exists(model_path):
        console.print(f"[red]ERROR[/red] Model not found: {model_path}")
        return

    exporter = OmniExporter()
    try:
        core, heads, meta = exporter.load_as_inference(model_path)
    except Exception as e:
        console.print(f"[red]ERROR[/red] Load failed: {e}")
        return

    from .safety_guard import SafetyGuard
    safety_heads = {k: v for k, v in heads.items() if 'safety' in k.lower()}
    if not safety_heads:
        console.print("[yellow]WARN[/yellow] No safety heads found.")
        return

    for head_id, head in safety_heads.items():
        guard = SafetyGuard(head, emergency_class=1)
        guard.add_constraint('lidar_front', min_safe=0.10, max_safe=50.0)
        test_cases = [{'lidar_front': 0.05}, {'lidar_front': 1.0}]
        report = guard.generate_safety_report(test_cases)

        table = Table(title=f"Safety: {head_id}", border_style="white", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total Test Cases", str(report['total_cases']))
        table.add_row("Passed", f"[color(117)]{report['passed']}[/]")
        console.print(table)

    console.print(f"\n[color(117)]DONE[/] Safety verification complete.")
 
def handle_capabilities(args):
    """Capabilities Paper Summary"""
    from rich.markdown import Markdown
    paper_path = "docs/CAPABILITIES.md"
    if not os.path.exists(paper_path):
        console.print(f"[red]ERROR[/red] Capabilities paper not found at {paper_path}")
        return

    with open(paper_path, 'r') as f:
        md_content = f.read()
    
    # Show a preview in a panel
    console.print(Panel(Markdown(md_content), title="OmniTrain Capabilities Paper", border_style="color(117)"))
    console.print(f"\n[dim]Full document located at: {os.path.abspath(paper_path)}[/]")

def handle_status(args):
    """System Health & Resource Monitor"""
    table = Table(title="OmniTrain System Health", border_style="color(117)", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # 1. Bus Check
    is_mac = platform.system() == "Darwin"
    shm_available = os.path.exists("/dev/shm") or is_mac
    bus_active = "[green]ONLINE[/]" if shm_available else "[yellow]LOCAL-ONLY[/]"
    table.add_row("TokenBus (IPC)", bus_active, "POSIX Shared Memory")

    # 2. Hardware
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    table.add_row("Inference Engine", f"[white]{device}[/]", f"Torch {torch.__version__}")

    # 3. Project
    proj = "[green]LOADED[/]" if os.path.exists("config.yaml") else "[red]MISSING[/]"
    table.add_row("Project Config", proj, "config.yaml")

    console.print(table)

def handle_audit(args):
    """Environment Industrialization Audit"""
    console.print("\n[bold]OmniTrain Industrial Audit[/]")
    checks = {
        "Python Version": sys.version.split()[0],
        "Platform": platform.platform(),
        "PyTorch": torch.__version__,
        "Shared Memory": "Available" if os.path.exists("/dev/shm") else "Emulated",
    }
    for k, v in checks.items():
        console.print(f"  [color(117)]•[/] {k:15}: [white]{v}[/]")
    
    # Check ROS2
    try:
        import rclpy
        console.print("  [color(117)]•[/] ROS 2           : [green]Found[/]")
    except:
        console.print("  [color(117)]•[/] ROS 2           : [yellow]Not Found (Optional)[/]")

def handle_config(args):
    from rich.prompt import Prompt, Confirm
    
    config_path = args[0] if args else "config.yaml"
    if not os.path.exists(config_path):
        console.print(f"[red]ERROR[/red] {config_path} not found. Run [white]/init[/white] first.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    console.print("\n[color(117)]--- OmniTrain Interactive Config Editor ---[/]")
    
    config['project'] = Prompt.ask("Project Name", default=config.get('project', 'Alpha_Robot'))
    
    model = config.get('model', {})
    console.print("\n[color(117)]Model Architecture[/]")
    model['d_model'] = int(Prompt.ask("  d_model (Dimension)", default=str(model.get('d_model', 512))))
    model['n_latents'] = int(Prompt.ask("  n_latents (Tokens)", default=str(model.get('n_latents', 64))))
    model['num_layers'] = int(Prompt.ask("  num_layers (Depth)", default=str(model.get('num_layers', 3))))
    config['model'] = model
    
    inputs = config.get('inputs', [])
    console.print(f"\n[color(117)]Sensors / Inputs[/] (Currently {len(inputs)} configured)")
    
    if Confirm.ask("Do you want to open the full file in your system editor for advanced changes?", default=False):
        editor = os.environ.get('EDITOR', 'nano')
        os.system(f"{editor} {config_path}")
        console.print(f"\n[color(117)]DONE[/] Editor closed.")
        return

    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    console.print("\n[color(117)]DONE[/] Configuration saved to [white]config.yaml[/white].")

def handle_help(args):
    table = Table(title="Available Commands", border_style="white", box=None)
    table.add_column("Command", style="color(117)")
    table.add_column("Description", style="dim")
    
    table.add_row("/init", "Scaffold a new project interactively")
    table.add_row("/status", "Monitor system health and resources")
    table.add_row("/audit", "Verify environment industrialization")
    table.add_row("/config", "Interactive YAML configuration editor")
    table.add_row("/record <config>", "Record TokenBus data to CSV")
    table.add_row("/train <config>", "Train a Liquid Neural Network (3-phase Curriculum)")
    table.add_row("/run <config>", "Launch real-time inference pipeline")
    table.add_row("/bus <session>", "Monitor live bus (default: omni_default)")
    table.add_row("/inspect <model>", "View model architecture")
    table.add_row("/deploy <model>", "Prepare for edge deployment")
    table.add_row("/test <model>", "Run safety and capability tests (alias for /verify)")
    table.add_row("/capabilities", "Show the System Capabilities White Paper")
    table.add_row("/clear", "Clear terminal screen")
    table.add_row("/exit", "Exit OmniTrain")
    
    console.print(table)

def main():
    print_dashboard()
    
    commands = {
        "/init": handle_init,
        "/status": handle_status,
        "/audit": handle_audit,
        "/config": handle_config,
        "/record": handle_record,
        "/train": handle_train,
        "/run": handle_run,
        "/bus": handle_bus,
        "/inspect": handle_inspect,
        "/deploy": handle_deploy,
        "/verify": handle_verify,
        "/test": handle_verify,
        "/capabilities": handle_capabilities,
        "/paper": handle_capabilities,
        "/help": handle_help,
        "/clear": lambda _: os.system('clear' if os.name == 'posix' else 'cls'),
        "/exit": lambda _: sys.exit(0)
    }
    
    completer = WordCompleter(list(commands.keys()), ignore_case=True)
    style = PromptStyle.from_dict({
        'prompt': '#87d7ff bold',
    })
    
    session = PromptSession(completer=completer, style=style)
    
    while True:
        try:
            text = session.prompt("> ")
            if not text.strip():
                continue
            
            # Handle slash commands
            if text.startswith('/'):
                parts = shlex.split(text)
                cmd = parts[0]
                args = parts[1:]
                
                if cmd in commands:
                    commands[cmd](args)
                else:
                    console.print(f"[red]ERROR[/red] Unknown command: {cmd}. Type [white]/help[/white] for list.")
            else:
                console.print("[dim]Enter a command starting with [/][white]/[/][dim] (try [/][white]/help[/][dim])[/]")
                
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == '__main__':
    main()
