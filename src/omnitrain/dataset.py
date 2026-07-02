import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import yaml
from typing import Dict, List
from .curriculum import SignalCorruptor


class OmniLogDataset(Dataset):
    """
    High-Fidelity OmniTrain Dataset.
    Supports multi-column sensor data (vision_0, vision_1, ...) and
    adaptive normalization for robotics.
    """

    def __init__(
        self,
        csv_path: str,
        config: dict,
        seq_len: int = 32,
        noise_level: int = 0,
    ):
        self.config = config
        self.seq_len = seq_len
        self.noise_level = noise_level

        # Map inputs and heads from config to CSV columns
        self.input_mappings = []
        for input_cfg in config.get('inputs', []):
            self.input_mappings.append({
                'id': input_cfg['id'],
                'dim': input_cfg.get('dim', 1),
                'type': input_cfg.get('type', 'sensor'),
                'range': input_cfg.get('range', [0.0, 1.0]),
                'noise': input_cfg.get('noise', False)
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
        
        self.num_sequences = max(0, len(self.timestamps) - seq_len + 1)

    def _parse_csv(self, csv_path: str):
        """Read the CSV and map columns to internal arrays (handling multi-dim)."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.timestamps = np.array([float(r['timestamp']) for r in rows], dtype=np.float64)
        
        if len(self.timestamps) == 0:
            raise ValueError("OmniTrain Dataset Error: The provided CSV file contains no data rows.")
        elif len(self.timestamps) == 1:
            self.dt = np.array([0.1], dtype=np.float32)
        else:
            self.dt = np.diff(self.timestamps, prepend=self.timestamps[0] - 0.1)
            self.dt = np.clip(self.dt, 0.001, 1.0).astype(np.float32)

        self.dt = torch.tensor(self.dt, dtype=torch.float32)

        self.data_store = {}
        self.raw_data_store = {}

        # Parse and Normalize inputs (Robust Z-Score Normalization)
        # z-score + ±5σ clip. This ensures OOD sensor readings (noise, spikes)
        # never saturate CfC tanh/softplus activations.
        for mapping in self.input_mappings:
            m_id = mapping['id']
            dim = mapping['dim']
            
            # Find the input config to update it with stats
            target_cfg = next((c for c in self.config.get('inputs', []) if c['id'] == m_id), None)

            if dim == 1:
                # Scalar sensor
                raw = np.array([float(r[m_id]) for r in rows], dtype=np.float32)
                mean = float(raw.mean())
                std  = float(raw.std() + 1e-6)
                
                norm = np.clip((raw - mean) / std, -5.0, 5.0)
                self.data_store[m_id] = torch.tensor(norm, dtype=torch.float32).unsqueeze(1)
                self.raw_data_store[m_id] = torch.tensor(raw, dtype=torch.float32).unsqueeze(1)
                
                if target_cfg:
                    target_cfg['norm_mean'] = mean
                    target_cfg['norm_std'] = std
            else:
                # Multi-dimensional sensor (expanded columns)
                storage = []
                means = []
                stds = []
                raw_list = []
                for i in range(dim):
                    col_name = f"{m_id}_{i}"
                    if col_name in rows[0]:
                        raw = np.array([float(r[col_name]) for r in rows], dtype=np.float32)
                    else:
                        raw = np.array([float(r[m_id]) for r in rows], dtype=np.float32)

                    m = float(raw.mean())
                    s = float(raw.std() + 1e-6)
                    means.append(m)
                    stds.append(s)
                    
                    storage.append(np.clip((raw - m) / s, -3.0, 3.0))
                    raw_list.append(raw)

                self.data_store[m_id] = torch.tensor(np.stack(storage, axis=1), dtype=torch.float32)
                self.raw_data_store[m_id] = torch.tensor(np.stack(raw_list, axis=1), dtype=torch.float32)
                
                if target_cfg:
                    target_cfg['norm_mean'] = means
                    target_cfg['norm_std'] = stds

        # Parse heads (labels)
        for mapping in self.head_mappings:
            h_id = mapping['id']
            if mapping['num_classes'] > 0:
                self.data_store[h_id] = torch.tensor([int(r[h_id]) for r in rows], dtype=torch.long)
            else:
                dims = mapping['dim']
                storage = []
                for d in range(dims):
                    col = f"{h_id}_{d}"
                    storage.append(np.array([float(r[col]) for r in rows], dtype=np.float32))
                self.data_store[h_id] = torch.tensor(np.stack(storage, axis=1), dtype=torch.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len

        batch = {
            'dt': self.dt[start:end],
            'inputs': {},
            'targets': {}
        }

        for mapping in self.input_mappings:
            m_id = mapping['id']
            t = self.data_store[m_id][start:end]
            
            # Targeted Noise
            if mapping['noise']:
                if self.noise_level >= 1:
                    t = SignalCorruptor.apply_dropout(t, drop_prob=0.1, fill_value=1.0)
                if self.noise_level >= 2:
                    t = SignalCorruptor.apply_gaussian_noise(t, std=0.05)
                
            batch['inputs'][m_id] = t

        for mapping in self.head_mappings:
            h_id = mapping['id']
            batch['targets'][h_id] = self.data_store[h_id][start:end]

        # Hardware sensors for failsafe (Tier 1)
        hw_list = []
        for mapping in self.input_mappings:
            hw_list.append(self.raw_data_store[mapping['id']][start:end])
        
        # Concatenate all raw inputs along dimension 1 (Batch, Seq, TotalRawDim)
        # Shield will use self.state_extractor to pick relevant indices or pool them.
        batch['hw_sensors'] = torch.cat(hw_list, dim=-1)
        
        # is_start=True if this is the beginning of the entire log.
        # When shuffle=False, this allows the trainer to propagate state.
        batch['is_start'] = (idx == 0)

        return batch