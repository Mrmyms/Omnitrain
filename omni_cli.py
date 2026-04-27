import sys
import torch
import time
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Add src to path so imports work from the root folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from omnitrain.exporter import OmniExporter
from omnitrain.omni_shield import OmniShieldGuard

console = Console()

def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]OmniTrain v4.0 - Interactive Liquid Brain[/bold cyan]\n"
        "[dim]Simulate physical sensors in real-time. Type 'exit' to quit.[/dim]", 
        border_style="cyan"
    ))
    
    # Auto-load the default model or allow custom path
    model_path = "SafeDelivery_Robot_trained.omni"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    try:
        with console.status(f"[bold green]Loading brain file {model_path}..."):
            core, heads, meta = OmniExporter().load_as_inference(model_path)
            # Rebuild shield
            shield = OmniShieldGuard.from_config(meta, heads['drive_control'], d_model=core.d_model)
    except Exception as e:
        console.print(f"[bold red]❌ Error loading brain:[/bold red] {e}")
        return
        
    core_name = 'LiquidFusionCore (CfC)' if hasattr(core, 'liquid_cell') else 'FusionCore (Transformer)'
    
    # System Info Header
    info_table = Table(show_header=False, box=None)
    info_table.add_row("🧠 Architecture:", f"[bold magenta]{core_name}[/bold magenta]")
    info_table.add_row("🛡️  OmniShield:", "[bold green]Active (3-Tiers)[/bold green]")
    info_table.add_row("⚡ Status:", "[bold yellow]Waiting for telemetry...[/bold yellow]")
    console.print(Panel(info_table, border_style="green"))
    
    prev_latents = None
    last_time = time.perf_counter()
    
    while True:
        try:
            console.print("\n[bold dim]─── Sensor Inputs ───────────────────────────────────────[/bold dim]")
            user_input = Prompt.ask("📡 [bold cyan]Lidar Distance[/bold cyan] (meters, e.g. 1.5)")
            if user_input.lower() in ['exit', 'q', 'quit']:
                break
            lidar_val = float(user_input)
            
            user_batt = Prompt.ask("🔋 [bold cyan]Battery Level[/bold cyan] (percentage, e.g. 80)")
            if user_batt.lower() in ['exit', 'q', 'quit']:
                break
            batt_val = float(user_batt) / 100.0
            
            # Compute exact dt since last command
            now = time.perf_counter()
            dt_val = now - last_time
            last_time = now
            
            # --- NEURAL INFERENCE ---
            # Liquid networks thrive on precise timing
            dt = torch.tensor([dt_val], dtype=torch.float32)
            
            # Simulate rich sensor inputs from simple numbers
            lidar_tensor = torch.randn(1, 32, 1) * 0.05 + lidar_val
            batt_tensor = torch.tensor([batt_val]).view(1, 1, 1)
            
            # Evolve liquid state
            l1 = core(lidar_tensor, dt, modal_id="lidar", prev_latents=prev_latents)
            l2 = core(batt_tensor, dt, modal_id="battery", prev_latents=l1)
            prev_latents = l2.detach()
            
            # Shield Audit
            hw_sensors = torch.tensor([[lidar_val, batt_val]])
            result = shield(l2, sensor_batch=hw_sensors)
            
            action = result['action'][0].detach().numpy()
            tier = result['tier']
            h_x = result['h_x'].mean().item()
            
            # --- UI RENDERING ---
            table = Table(title="📊 Live Telemetry", show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Metric", style="dim", width=20)
            table.add_column("Value")
            table.add_column("System Status")
            
            # Time Metric
            table.add_row("Delta Time (dt)", f"{dt_val*1000:.1f} ms", "⏳ Kernel Physics")
            
            # Action Metric
            action_str = f"Vel: {action[0]:.2f}m/s | Turn: {action[1]:.2f}rad"
            if tier > 0:
                table.add_row("Motor Command", f"[bold red]{action_str}[/bold red]", "🛑 INTERVENED")
            else:
                table.add_row("Motor Command", f"[bold green]{action_str}[/bold green]", "✅ AUTONOMOUS")
                
            # Safety Metric
            if tier == 1:
                saf_color = "red"
                saf_msg = "⚠️ HARDWARE EMERGENCY (Tier 1)"
            elif tier == 2:
                saf_color = "yellow"
                saf_msg = "🚧 CBF CORRECTION (Tier 2)"
            else:
                saf_color = "green"
                saf_msg = "✅ CLEAR PATH (Tier 0)"
                
            table.add_row("Safety Margin", f"[{saf_color}]h(x) = {h_x:.4f}[/{saf_color}]", f"[bold {saf_color}]{saf_msg}[/bold {saf_color}]")
            
            console.print(Panel(table, border_style="blue"))
            
        except ValueError:
            console.print("[bold red]❌ Please enter a valid number.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[dim]Exiting Interactive Brain...[/dim]")
            break

if __name__ == "__main__":
    main()
