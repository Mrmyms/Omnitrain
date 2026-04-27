import csv
import time
import os
import yaml
import numpy as np
from typing import Dict, List, Optional
from .token_bus import TokenBus
from rich.console import Console

console = Console()

class OmniRecorder:
    """
    OmniTrain Data Recorder.
    Captures live streams from TokenBus and serializes them into a structured CSV
    based on the project's config.yaml.
    """

    def __init__(self, config_path: str, session_id: str = "omni_default"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.bus = TokenBus(session_id=session_id, create=False)
        self.headers = self._build_headers()
        self.latest_data: Dict[str, Any] = {h: 0.0 for h in self.headers}
        self.active = False

    def _build_headers(self) -> List[str]:
        """Build CSV headers dynamically from config."""
        headers = ['timestamp']
        
        # Add Input Sensors
        for input_cfg in self.config.get('inputs', []):
            modal_id = input_cfg.get('id')
            # If it's a multi-dimensional sensor like Lidar, we might want to record 'min'
            # as discussed, but for now let's support a simple mapping.
            headers.append(modal_id)
            
        # Add Output Heads (Labels/Targets)
        for head_cfg in self.config.get('heads', []):
            head_id = head_cfg.get('id')
            out_dim = head_cfg.get('output_dim', 1)
            num_classes = head_cfg.get('num_classes', 0)
            
            if num_classes > 0:
                headers.append(head_id) # Classification: single int
            else:
                for i in range(out_dim):
                    headers.append(f"{head_id}_{i}") # Regression: multiple floats
                    
        return headers

    def record(self, output_path: str, hz: float = 10.0):
        """
        Main recording loop.
        Syncs data from TokenBus and writes to CSV at a fixed frequency.
        """
        self.active = True
        file_exists = os.path.exists(output_path)
        
        console.print(f"[bold green]RECORDING STARTED[/bold green] -> [white]{output_path}[/white] at {hz}Hz")
        console.print(f"[dim]Headers: {', '.join(self.headers)}[/dim]")

        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if not file_exists:
                writer.writeheader()

            dt = 1.0 / hz
            try:
                while self.active:
                    start_time = time.time()
                    
                    # Fetch window of data since last check
                    # (Simplified: just take the latest tokens for each modality)
                    tokens = self.bus.get_window(time.time() - 0.5, time.time())
                    
                    current_row = {'timestamp': time.time()}
                    
                    # Group tokens by modality
                    latest_tokens = {}
                    for t in tokens:
                        latest_tokens[t['modal_id']] = t['data']
                    
                    # Fill inputs
                    for input_cfg in self.config.get('inputs', []):
                        m_id = input_cfg.get('id')
                        if m_id in latest_tokens:
                            data = latest_tokens[m_id]
                            # Strategy: if dimension > 1, take the minimum (for lidar) 
                            # or just the first element. This matches our Dataset expectation.
                            if data.size > 1:
                                val = np.min(data)
                            else:
                                val = float(data[0])
                            self.latest_data[m_id] = val
                        current_row[m_id] = self.latest_data[m_id]

                    # Fill labels (heads)
                    for head_cfg in self.config.get('heads', []):
                        h_id = head_cfg.get('id')
                        out_dim = head_cfg.get('output_dim', 1)
                        num_classes = head_cfg.get('num_classes', 0)
                        
                        if h_id in latest_tokens:
                            data = latest_tokens[h_id]
                            if num_classes > 0:
                                self.latest_data[h_id] = int(data[0])
                            else:
                                for i in range(min(len(data), out_dim)):
                                    self.latest_data[f"{h_id}_{i}"] = float(data[i])
                        
                        # Apply to row
                        if num_classes > 0:
                            current_row[h_id] = self.latest_data[h_id]
                        else:
                            for i in range(out_dim):
                                current_row[f"{h_id}_{i}"] = self.latest_data.get(f"{h_id}_{i}", 0.0)

                    writer.writerow(current_row)
                    f.flush() # Ensure data is written even if interrupted

                    # Precision timing
                    elapsed = time.time() - start_time
                    sleep_time = max(0, dt - elapsed)
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.active = False
                console.print("\n[bold yellow]RECORDING STOPPED[/bold yellow]")
