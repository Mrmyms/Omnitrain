import time
import json
import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from .token_bus import TokenBus
from .telemetry import OmniHealthMonitor

def generate_dashboard(bus: TokenBus, monitor: OmniHealthMonitor) -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="head", size=3), Layout(name="body"), Layout(name="foot", size=5))
    layout["head"].update(Panel("OmniTrain Fleet Monitor v4.0", style="bold cyan"))

    now = time.time()
    tokens, _ = bus.get_since_index(max(0, bus.ptr_store[0] - 100))
    diag = monitor.get_diagnostics()
    
    # 1. Modality Table
    table = Table(title="Live Sensor Streams", box=None, border_style="cyan")
    table.add_column("MODALITY")
    table.add_column("HZ", justify="right")
    table.add_column("LATENCY (ms)", justify="right")
    table.add_column("STATUS", justify="center")

    stats = {}
    for t in tokens:
        mid = t['modal_id']
        if mid not in stats: stats[mid] = {"cnt": 0, "lats": []}
        stats[mid]["cnt"] += 1
        stats[mid]["lats"].append((now - t['timestamp']) * 1000)

    for mid, data in stats.items():
        ls = data["lats"]
        avg_lat = sum(ls) / len(ls) if ls else 0.0
        status = "[green]LIVE[/green]" if avg_lat < 100 else "[yellow]LAG[/yellow]"
        table.add_row(mid, f"{data['cnt']*4} Hz", f"{avg_lat:.1f}", status)

    layout["body"].update(table)

    # 2. Diagnostics Footer
    diag_text = (
        f"System Status: {diag['status']} | Active Nodes: {diag['active_nodes']} | "
        f"Stale: {len(diag['stale_nodes'])} | SID: {bus.sid}\n"
        f"Memory Usage: {bus.max_tokens * bus.token_dim * 4 / 1e6:.2f} MB"
    )
    layout["foot"].update(Panel(diag_text, title="Health & Diagnostics", border_style="green" if diag['status'] == "HEALTHY" else "red"))
    
    # Export for cloud/external tools
    with open("status.json", "w") as f:
        json.dump(diag, f)
        
    return layout

def run_monitor(bus: TokenBus):
    console = Console()
    monitor = OmniHealthMonitor(bus)
    
    try:
        with Live(generate_dashboard(bus, monitor), refresh_per_second=4, console=console) as live:
            while True:
                time.sleep(0.25)
                live.update(generate_dashboard(bus, monitor))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]🛑 Telemetry Monitor stopping...[/bold yellow]")
