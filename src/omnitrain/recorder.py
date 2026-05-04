import csv
import time
import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Any
from .token_bus import TokenBus
from rich.console import Console

console = Console()

class OmniRecorder:
    """
    Industrial OmniTrain Recorder (High-Efficiency).
    Uses pointer-based retrieval to avoid O(N) memory scans.
    Operates in 'Event-Stream' mode: captures every state change without data loss.
    """

    def __init__(self, config_path: str, session_id: str = "omni_default"):
        self.config_path = config_path
        self.session_id = session_id
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.headers = self._build_headers()
        
        self.latest_data: Dict[str, Any] = {}
        self._init_latest_data()
        
        self.last_read_idx = 0
        self.active = False
        self.bus = None


    def _init_latest_data(self):
        for input_cfg in self.config.get('inputs', []):
            m_id = input_cfg['id']
            dim = input_cfg.get('dim', 1)
            self.latest_data[m_id] = np.zeros(dim)
        
        for head_cfg in self.config.get('heads', []):
            h_id = head_cfg['id']
            dim = head_cfg.get('output_dim', 1)
            if head_cfg.get('num_classes', 0) > 0:
                self.latest_data[h_id] = 0
            else:
                self.latest_data[h_id] = np.zeros(dim)

    def _build_headers(self) -> List[str]:
        headers = ['timestamp']
        for input_cfg in self.config.get('inputs', []):
            m_id = input_cfg['id']
            dim = input_cfg.get('dim', 1)
            if dim == 1: headers.append(m_id)
            else:
                for i in range(dim): headers.append(f"{m_id}_{i}")
            
        for head_cfg in self.config.get('heads', []):
            h_id = head_cfg['id']
            dim = head_cfg.get('output_dim', 1)
            if head_cfg.get('num_classes', 0) > 0: headers.append(h_id)
            else:
                for i in range(dim): headers.append(f"{h_id}_{i}")
        return headers

    def start(self, output_path: str):
        """
        High-Fidelity Event-Driven Recording Loop.
        """
        # Lazy initialization of the bus to support multiprocessing pickling
        if self.bus is None:
            self.bus = TokenBus(session_id=self.session_id, create=False)

        self.active = True

        file_exists = os.path.exists(output_path)
        
        console.print(f"[bold arctic_blue]INDUSTRIAL RECORDER (EVENT-MODE)[/bold arctic_blue] -> [white]{output_path}[/white]")
        
        # Pre-calculate mapping for faster row building
        input_info = []
        for inp in self.config.get('inputs', []):
            input_info.append((inp['id'], inp.get('dim', 1)))
        
        head_info = []
        for head in self.config.get('heads', []):
            head_info.append((head['id'], head.get('output_dim', 1), head.get('num_classes', 0)))

        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if not file_exists:
                writer.writeheader()

            try:
                row_count = 0
                while self.active:
                    # 1. Get ALL tokens since last check (using the new global SHM pointer)
                    new_tokens, next_idx = self.bus.get_since_index(self.last_read_idx)
                    
                    if not new_tokens:
                        time.sleep(0.002) # Ultra-short sleep
                        continue

                    # 2. Process every token INDIVIDUALLY to create one row per event
                    for t in new_tokens:
                        m_id = t['modal_id']
                        # Update the global ZOH state
                        if m_id in self.latest_data:
                            self.latest_data[m_id] = t['data']
                        
                        # 3. Build and write the row for THIS SPECIFIC event
                        current_row = {'timestamp': t['timestamp']}
                        
                        for mid, dim in input_info:
                            data = self.latest_data[mid]
                            if dim == 1:
                                current_row[mid] = float(data[0]) if hasattr(data, '__len__') else float(data)
                            else:
                                for i in range(dim):
                                    current_row[f"{mid}_{i}"] = float(data[i]) if i < len(data) else 0.0

                        for hid, dim, n_classes in head_info:
                            data = self.latest_data[hid]
                            if n_classes > 0:
                                current_row[hid] = int(data[0]) if hasattr(data, '__len__') else int(data)
                            else:
                                for i in range(dim):
                                    current_row[f"{hid}_{i}"] = float(data[i]) if i < len(data) else 0.0

                        writer.writerow(current_row)
                        row_count += 1
                        
                        # Periodic flush to ensure data isn't lost if crashed
                        if row_count % 100 == 0:
                            f.flush()
                    
                    self.last_read_idx = next_idx

            except KeyboardInterrupt:
                self.active = False
                console.print(f"\n[bold green]✔ RECORDING STOPPED[/bold green] (Saved {row_count} events)")

