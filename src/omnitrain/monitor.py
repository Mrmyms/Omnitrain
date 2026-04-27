import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from .token_bus import TokenBus

def generate_dashboard(bus: TokenBus) -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="head", size=3), Layout(name="body"), Layout(name="foot", size=3))
    layout["head"].update(Panel("OmniTrain Engine Status", style="color(117)"))

    now = time.time()
    tokens = bus.get_window(now - 1.0, now)
    
    table = Table(title="Live Modalities", box=None, border_style="color(117)")
    table.add_column("ID")
    table.add_column("HZ", justify="right")
    table.add_column("LATENCY", justify="right")

    stats = {}
    for t in tokens:
        mid = t['modal_id']
        if mid not in stats:
            stats[mid] = {"cnt": 0, "lats": []}
        
        stats[mid]["cnt"] += 1
        stats[mid]["lats"].append((now - t['timestamp']) * 1000)

    for mid, data in stats.items():
        ls = data["lats"]
        avg_lat = sum(ls) / len(ls) if ls else 0.0
        table.add_row(mid, f"{data['cnt']} Hz", f"{avg_lat:.2f}")

    layout["body"].update(table)
    layout["foot"].update(Panel(f"Capacity: {bus.buffer_size()} slots", style="color(117)"))
    return layout

def run_monitor(bus: TokenBus):
    console = Console()
    with Live(generate_dashboard(bus), refresh_per_second=4, console=console) as live:
        while True:
            time.sleep(0.25)
            live.update(generate_dashboard(bus))
