import argparse
import yaml
import json
import csv
import time
import sys
from pathlib import Path

try:
    import serial
except ImportError:
    print("[Error] 'pyserial' is not installed. Run: pip install pyserial")
    sys.exit(1)

from rich.console import Console
console = Console()

class ESP32SerialLogger:
    """
    Smart Data Collection Tool for OmniTrain.
    Reads JSON lines from an ESP32 over USB Serial, maps the keys to the OmniTrain config.yaml,
    and automatically flattens them into a synchronized dataset.csv ready for training.
    """
    def __init__(self, port: str, baud: int, config_path: str, output_csv: str):
        self.port = port
        self.baud = baud
        self.config_path = Path(config_path)
        self.output_csv = Path(output_csv)
        self.schema = []
        self.header = ["timestamp"]
        
        self._load_schema()

    def _load_schema(self):
        """Reads config.yaml to understand expected modalities."""
        if not self.config_path.exists():
            console.print(f"[bold red]Error: Config file {self.config_path} not found.[/bold red]")
            sys.exit(1)
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        inputs = config.get('inputs', [])
        targets = config.get('targets', []) # Or outputs
        
        console.print("[bold cyan]Loaded Schema Mapping:[/bold cyan]")
        
        for item in inputs:
            sid = item['id']
            dim = item.get('dim', 1)
            self.schema.append({'id': sid, 'dim': dim, 'type': 'input'})
            for i in range(dim):
                self.header.append(f"{sid}_{i}")
            console.print(f"  - Input: {sid} (Dim: {dim})")
            
        for item in targets:
            sid = item['id']
            dim = item.get('dim', 1)
            self.schema.append({'id': sid, 'dim': dim, 'type': 'target'})
            for i in range(dim):
                self.header.append(f"{sid}_{i}")
            console.print(f"  - Target: {sid} (Dim: {dim})")

    def _parse_packet(self, data: dict, timestamp: float) -> list:
        """Flattens the JSON dictionary into the CSV row format defined by the schema."""
        row = [f"{timestamp:.3f}"]
        
        for schema_item in self.schema:
            sid = schema_item['id']
            dim = schema_item['dim']
            
            if sid not in data:
                # If a sensor dropped out or was missed, we pad with zeros or NaNs
                row.extend([0.0] * dim)
                continue
                
            val = data[sid]
            
            if isinstance(val, (list, tuple)):
                if len(val) != dim:
                    console.print(f"[yellow]Warning: {sid} expected dim {dim} but got {len(val)}[/yellow]")
                
                # Fill up to expected dim
                for i in range(dim):
                    if i < len(val):
                        row.append(val[i])
                    else:
                        row.append(0.0)
            else:
                # Scalar value
                row.append(val)
                if dim > 1:
                    row.extend([0.0] * (dim - 1))
                    
        return row

    def start_recording(self):
        console.print(f"\n[bold green]Opening Serial Port {self.port} @ {self.baud} baud...[/bold green]")
        
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except serial.SerialException as e:
            console.print(f"[bold red]Failed to open port {self.port}:[/bold red] {e}")
            sys.exit(1)
            
        # Ensure output directory exists
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        packets_saved = 0
        
        with open(self.output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.header)
            
            console.print(f"[bold green]Recording to {self.output_csv}... Press Ctrl+C to stop.[/bold green]\n")
            
            try:
                while True:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        # Try parsing JSON
                        try:
                            data = json.loads(line)
                            current_time = time.time() - start_time
                            
                            row = self._parse_packet(data, current_time)
                            writer.writerow(row)
                            packets_saved += 1
                            
                            # Print live preview (throttle to avoid spamming console)
                            if packets_saved % 10 == 0:
                                sys.stdout.write(f"\r[Live] Logged {packets_saved} packets. Last T: {current_time:.2f}s  ")
                                sys.stdout.flush()
                                
                        except json.JSONDecodeError:
                            # Safely ignore corrupted serial lines
                            pass
                            
            except KeyboardInterrupt:
                console.print(f"\n\n[bold yellow]Recording stopped by user.[/bold yellow]")
                
            finally:
                ser.close()
                console.print(f"[bold green]Saved {packets_saved} total frames to {self.output_csv}.[/bold green]")
                console.print("[bold green]Data is ready for OmniTrain Phase 1 (Imitation Learning)![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="OmniTrain Smart Serial Logger (JSON Schema-Driven)")
    parser.add_argument("--port", type=str, required=True, help="Serial Port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to OmniTrain config.yaml")
    parser.add_argument("--out", type=str, default="dataset.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    logger = ESP32SerialLogger(
        port=args.port,
        baud=args.baud,
        config_path=args.config,
        output_csv=args.out
    )
    logger.start_recording()

if __name__ == "__main__":
    main()
