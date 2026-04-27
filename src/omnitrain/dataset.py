import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import yaml
from typing import Dict, List
from .curriculum import SignalCorruptor


class OmniLogDataset(Dataset):
    """
    Dynamic OmniTrain Dataset.
    Loads data from CSV based on the provided config.yaml.
    Automatically maps columns to inputs and targets.
    """

    def __init__(
        self,
        csv_path: str,
        config: dict,
        seq_len: int = 32,
        chaos_level: int = 0,
    ):
        self.config = config
        self.seq_len = seq_len
        self.chaos_level = chaos_level

        # Map inputs and heads from config to CSV columns
        self.input_mappings = []
        for input_cfg in config.get('inputs', []):
            self.input_mappings.append({
                'id': input_cfg['id'],
                'dim': input_cfg.get('dim', 1),
                'type': input_cfg.get('type', 'sensor')
            })

        self.head_mappings = []
        for head_cfg in config.get('heads', []):
            h_id = head_cfg['id']
            out_dim = head_cfg.get('output_dim', 1)
            num_classes = head_cfg.get('num_classes', 0)
            self.head_mappings.append({
                'id': h_id,
                'dim': out_dim,
                'num_classes': num_classes
            })

        self._parse_csv(csv_path)
        self.num_sequences = max(1, len(self.timestamps) - seq_len)

    def _parse_csv(self, csv_path: str):
        """Read the CSV and map columns to internal arrays."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.timestamps = np.array([float(r['timestamp']) for r in rows], dtype=np.float64)
        self.dt = np.diff(self.timestamps, prepend=self.timestamps[0] - 0.1)
        self.dt = np.clip(self.dt, 0.001, 1.0).astype(np.float32)

        self.data_store = {}
        
        # Parse inputs
        for mapping in self.input_mappings:
            m_id = mapping['id']
            # We assume the CSV has a column with the same name as the sensor ID
            # In our Recorder, for multi-dim sensors, we recorded 'min'.
            # The dataset will expand this back to 'dim' features.
            self.data_store[m_id] = np.array([float(r[m_id]) for r in rows], dtype=np.float32)

        # Parse heads (labels)
        for mapping in self.head_mappings:
            h_id = mapping['id']
            if mapping['num_classes'] > 0:
                self.data_store[h_id] = np.array([int(r[h_id]) for r in rows], dtype=np.int64)
            else:
                dims = mapping['dim']
                for d in range(dims):
                    col = f"{h_id}_{d}"
                    self.data_store[col] = np.array([float(r[col]) for r in rows], dtype=np.float32)

    def _expand_sensor(self, val: float, dim: int) -> np.ndarray:
        """Expand a single scalar value to a dimensional tensor (e.g. Lidar scan)."""
        if dim == 1:
            return np.array([val], dtype=np.float32)
        
        # For Lidar-like sensors, create a spatial distribution
        angles = np.linspace(-1, 1, dim)
        spatial_factor = 1.0 + 0.5 * np.abs(angles)
        base = val * spatial_factor
        noise = np.random.normal(0, 0.02, dim)
        return np.clip(base + noise, 0.05, 20.0).astype(np.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len

        batch = {
            'dt': torch.tensor(self.dt[start:end], dtype=torch.float32),
            'inputs': {},
            'targets': {}
        }

        for mapping in self.input_mappings:
            m_id = mapping['id']
            dim = mapping['dim']
            vals = self.data_store[m_id][start:end]
            
            expanded = np.stack([self._expand_sensor(v, dim) for v in vals]) # (T, dim)
            # Add feature dimension if needed
            if expanded.ndim == 2:
                expanded = expanded[:, :, np.newaxis] # (T, dim, 1)
            
            t = torch.tensor(expanded, dtype=torch.float32)
            
            # Apply chaos
            if self.chaos_level >= 1 and 'lidar' in m_id:
                t = SignalCorruptor.apply_dropout(t, drop_prob=0.1, fill_value=20.0)
            if self.chaos_level >= 2 and 'lidar' in m_id:
                t = SignalCorruptor.apply_gaussian_noise(t, std=0.3, clamp_min=0.05, clamp_max=20.0)
                
            batch['inputs'][m_id] = t

        for mapping in self.head_mappings:
            h_id = mapping['id']
            if mapping['num_classes'] > 0:
                batch['targets'][h_id] = torch.tensor(self.data_store[h_id][start:end], dtype=torch.long)
            else:
                target_list = []
                for d in range(mapping['dim']):
                    target_list.append(self.data_store[f"{h_id}_{d}"][start:end])
                batch['targets'][h_id] = torch.tensor(np.stack(target_list, axis=1), dtype=torch.float32)

        # Raw hardware sensors for Tier 1 checks
        hw_list = []
        for mapping in self.input_mappings:
            hw_list.append(self.data_store[mapping['id']][start:end])
        batch['hw_sensors'] = torch.tensor(np.stack(hw_list, axis=1), dtype=torch.float32)

        return batch
