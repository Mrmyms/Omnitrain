import os
import yaml
import time
import importlib
import multiprocessing
import signal
import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Generator

from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.token_bus import TokenBus
from omnitrain.trainer import Trainer
from omnitrain.exporter import OmniExporter
from omnitrain.omni_shield import OmniShieldGuard
from omnitrain.diagnostics_and_monitoring import run_monitor, OmniDiagnostic

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box

console = Console()


class ProjectManager:
    """
    Handles environment setup, folder scaffolding, and sample dataset generation.
    """

    @staticmethod
    def init_project(project_dir: str = ".") -> None:
        """
        Scaffolds a new OmniTrain project workspace with default config and logs.
        """
        os.makedirs(project_dir, exist_ok=True)
        
        # 1. Create default config.yaml
        config_path = os.path.join(project_dir, "config.yaml")
        if not os.path.exists(config_path):
            default_config = {
                "project": "OmniTrain_Project",
                "logs": "training_data.csv",
                "model": {
                    "d_model": 256,
                    "n_latents": 32,
                    "conectoma": {
                        "enabled": True,
                        "sensory_n": 8,
                        "wall_n": 16,
                        "command_n": 8
                    }
                },
                "transport": {
                    "session_id": "omni_session",
                    "max_tokens": 1000
                },
                "inputs": [
                    {
                        "id": "lidar",
                        "type": "sensor",
                        "hz": 10,
                        "dim": 512,
                        "plugin": "omnitrain.plugins.DummyLidarPlugin"
                    }
                ],
                "heads": [
                    {
                        "id": "drive",
                        "type": "regression",
                        "output_dim": 2
                    }
                ],
                "training": {
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.002,
                    "weight_decay": 0.0001,
                    "lagrangian": {
                        "init_lambda": 0.1,
                        "kp": 0.04,
                        "ki": 0.01,
                        "kd": 0.002
                    }
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            console.print(f"[green]✓[/] Created default config at: [white]{config_path}[/]")

        # 2. Create models directory
        os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)

        # 3. Create dummy dataset
        csv_path = os.path.join(project_dir, "training_data.csv")
        if not os.path.exists(csv_path):
            ProjectManager.generate_mock_dataset(csv_path)

        console.print("[bold green]OmniTrain Project Initialized successfully![/]")

    @staticmethod
    def generate_mock_dataset(file_path: str, num_samples: int = 200) -> None:
        """
        Generates a high-fidelity mock dataset CSV for training.
        """
        import csv
        LIDAR_DIM = 512
        DRIVE_DIM = 2
        
        headers = ['timestamp']
        for i in range(LIDAR_DIM): 
            headers.append(f'lidar_{i}')
        for i in range(DRIVE_DIM): 
            headers.append(f'drive_{i}')
        headers.append('safety')

        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for i in range(num_samples):
                timestamp = time.time() + (i * 0.1)
                base_dist = np.random.uniform(0.1, 5.0)
                lidar_beams = base_dist + np.random.normal(0, 0.05, LIDAR_DIM)
                
                if base_dist < 0.3:
                    target_speed = [0.0, 0.0]
                    safety_label = 1 # Danger / Emergency Stop
                else:
                    target_speed = [0.5, np.random.uniform(-0.1, 0.1)]
                    safety_label = 0 # Safe
                
                row = {'timestamp': timestamp}
                for j in range(LIDAR_DIM): 
                    row[f'lidar_{j}'] = lidar_beams[j]
                for j in range(DRIVE_DIM): 
                    row[f'drive_{j}'] = target_speed[j]
                row['safety'] = safety_label
                
                writer.writerow(row)
        console.print(f"[green]✓[/] Generated mock dataset at: [white]{file_path}[/]")


class LiquidTrainer:
    """
    High-level training curriculum orchestrator.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.trainer = Trainer.from_config(config_path)

    def fit(self, csv_path: Optional[str] = None, epochs: Optional[int] = None, verbose: bool = True, callback: Optional[callable] = None) -> dict:
        """
        Runs the stateful Lagrangian training process.
        """
        csv_path = csv_path or self.config.get("logs", "training_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV {csv_path} not found.")

        epochs = epochs or self.config.get("training", {}).get("epochs", 10)
        
        console.print(f"[bold cyan]Starting training: {epochs} epochs on {csv_path}[/]")
        
        last_metrics = {}
        
        if verbose:
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

            with Live(layout, refresh_per_second=5, screen=False):
                for metrics in self.trainer.fit(csv_path, epochs=epochs):
                    last_metrics = metrics
                    
                    layout["header"].update(Panel(
                        f"[bold cyan]OMNITRAIN SDK[/] | Epoch {metrics['epoch']}/{epochs} | Phase: {metrics['phase']}",
                        border_style="cyan"
                    ))

                    table = Table(box=box.SIMPLE, expand=True)
                    table.add_column("Neural Path", style="cyan")
                    table.add_column("Loss Value", justify="right")
                    
                    table.add_row("Behavioral Policy", f"{metrics['policy']:.6f}")
                    table.add_row("Safety Manifold", f"{metrics['safety']:.6f}")
                    table.add_row("Barrier Constraint", f"{metrics['barrier']:.6f}")
                    
                    layout["metrics"].update(Panel(table, title="Conectoma Training Flow", border_style="white"))

                    violation = max(0.0, -metrics['barrier'])
                    status_color = "green" if violation < 0.01 else ("yellow" if violation < 0.1 else "red")
                    saf_msg = "SAFE" if status_color == "green" else ("CORRECTING" if status_color == "yellow" else "VIOLATION")
                    
                    layout["safety"].update(Panel(
                        f"\n[bold {status_color}]{saf_msg}[/]\n\nViolation: {violation:.4f}\nλ: {metrics.get('lambda', 0):.3f}",
                        title="Safety Manifold Shield",
                        border_style=status_color
                    ))
                    
                    layout["footer"].update(Panel(f"Training active. LR={self.trainer.lr:.6f}", border_style="dim"))
                    
                    if callback:
                        callback(metrics)
        else:
            for metrics in self.trainer.fit(csv_path, epochs=epochs):
                last_metrics = metrics
                if callback:
                    callback(metrics)

        console.print("[bold green]Training Completed Successfully![/]")
        return last_metrics


class EdgeDeployer:
    """
    Handles model compilation, tracing, and static 8-bit quantization.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")

    def export(self, target: str = "tensorrt", output_path: Optional[str] = None) -> str:
        """
        Exports the model to ONNX or Snapdragon DLC formats.
        """
        exporter = OmniExporter()
        core, heads, config = exporter.load_as_inference(self.model_path)
        
        target = target.lower()
        if not output_path:
            ext = ".dlc" if target == "snpe" else ".onnx"
            output_path = self.model_path.replace(".omni", ext)

        console.print(f"[bold cyan]Exporting model target: {target.upper()} -> {output_path}[/]")

        if target == "snpe":
            onnx_temp = self.model_path.replace(".omni", ".temp.onnx")
            exporter.export_for_qualcomm_snpe(core, heads, onnx_temp)
            success = exporter.convert_onnx_to_dlc(onnx_temp, output_path)
            if os.path.exists(onnx_temp):
                os.remove(onnx_temp)
            if not success:
                raise RuntimeError("Failed converting ONNX to Snapdragon DLC format.")
        else:
            exporter.export_to_onnx(core, heads, output_path)
            
        console.print(f"[bold green]Model successfully exported to: {output_path}[/]")
        return output_path


class AgentRunner:
    """
    High-level runner managing TokenBus, plugins, and execution.
    """

    def __init__(self, config_path: str = "config.yaml", model_path: Optional[str] = None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = model_path or f"{self.config.get('project', 'robot')}_final.omni"
        self.bus = None
        self.workers = []
        self.write_ptr = multiprocessing.Value('i', 0)

    def start(self) -> None:
        """
        Initializes TokenBus and starts sensor plugins in background processes.
        """
        sid = self.config.get('transport', {}).get('session_id', 'omni_session')
        max_tokens = self.config.get('transport', {}).get('max_tokens', 1000)
        
        self.bus = TokenBus(max_tokens=max_tokens, create=True, session_id=sid)
        console.print(f"[cyan]TokenBus initialized (Session ID: {sid})[/]")

        for input_cfg in self.config.get('inputs', []):
            modal_id = input_cfg.get('id')
            freq = float(input_cfg.get('hz', 10.0))
            plugin_path = input_cfg.get('plugin')

            try:
                mod_name, cls_name = plugin_path.rsplit('.', 1)
                plugin_class = getattr(importlib.import_module(mod_name), cls_name)
                
                kwargs = {k: v for k, v in input_cfg.items() if k not in ['id', 'plugin', 'hz']}
                instance = plugin_class(self.bus, modal_id, freq, write_ptr=self.write_ptr, **kwargs)

                p = multiprocessing.Process(target=instance.run, name=f"Worker-{modal_id}", daemon=True)
                p.start()
                self.workers.append(p)
                console.print(f"[green]✓[/] Plugin started: [white]{modal_id}[/] ({freq}Hz)")
            except Exception as e:
                console.print(f"[red]✕[/] Failed to launch plugin [white]{modal_id}[/]: {e}")

    def run_telemetry(self, duration: Optional[float] = None) -> None:
        """
        Launches the interactive Rich dashboard to monitor TokenBus telemetry.
        """
        if not self.bus:
            raise RuntimeError("AgentRunner must be started with .start() before running telemetry.")
        
        console.print("[bold cyan]Running telemetry console... Press Ctrl+C to stop.[/]")
        
        def handle_signal(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        
        try:
            run_monitor(self.bus, duration=duration)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """
        Gracefully terminates background processes and releases Shared Memory blocks.
        """
        console.print("[yellow]Stopping AgentRunner background processes...[/]")
        for p in self.workers:
            if p.is_alive():
                p.terminate()
        self.workers = []
        
        if self.bus:
            self.bus.cleanup()
            self.bus = None
        console.print("[green]✓[/] Cleanup complete.")
