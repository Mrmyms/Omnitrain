import time
import logging
import uuid
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Optional
import os
import platform
import json
import zlib

try:
    # Attempt to load the high-performance C++ backend
    import omni_bus_core
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

# Path for tracking active SHM sessions for cleanup
SHM_REGISTRY_DIR = os.path.expanduser("~/.omnitrain/shm_registry")


class TokenBus:
    """
    OmniTrain TokenBus: High-performance multimodal transport.
    Automatically uses C++ Posix SHM backend on Ubuntu/Linux if compiled.
    Falls back to Python SharedMemory for development and non-compiled environments.

    Includes a shape registry for Auto-Modality support: automatically tracks
    the input dimensions of each sensor modality for dynamic projection.
    """

    def __init__(self, max_tokens=2048, token_dim=512, modal_id_len=32, create=True, session_id=None):
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.modal_id_len = modal_id_len
        
        # Security: Include OS User ID in session to prevent cross-user data leakage
        import os as _os
        uid = _os.getuid() if hasattr(_os, 'getuid') else 0
        self.sid = f"{uid}_{session_id or uuid.uuid4().hex[:8]}"
        
        self.create = create
        # Auto-Modality: Track discovered sensor shapes
        self._shape_registry: Dict[str, int] = {}

        if HAS_CPP:
            logging.info(f"[TokenBus] C++ Backend Active (SID: {self.sid})")
            self.backend = omni_bus_core.NativeTokenBus(max_tokens, token_dim, modal_id_len, self.sid, create)
        else:
            logging.info(f"[TokenBus] Standard Python Backend Active (SID: {self.sid})")
            self._attach_python_backend()
            
        # Register session for robust cleanup (Reaper)
        if create:
            self._register_session()

    def _register_session(self):
        """Record SID and PID for the reaper to clean up after crashes."""
        os.makedirs(SHM_REGISTRY_DIR, exist_ok=True)
        reg_file = os.path.join(SHM_REGISTRY_DIR, f"{self.sid}.json")
        with open(reg_file, 'w') as f:
            json.dump({'pid': os.getpid(), 'sid': self.sid, 'created': time.time()}, f)

    def _attach_python_backend(self):
        from multiprocessing import shared_memory
        self.prefix = f"omni_shm_{self.sid}"

        data_size = self.max_tokens * self.token_dim * 4
        ts_size = self.max_tokens * 8
        id_size = self.max_tokens * self.modal_id_len
        crc_size = self.max_tokens * 4
        meta_size = 1024 * (self.modal_id_len + 4) 
        hb_size = 1024 * 8

        if self.create:
            self.shm_data = shared_memory.SharedMemory(name=f"{self.prefix}_data", create=True, size=data_size)
            self.shm_ts = shared_memory.SharedMemory(name=f"{self.prefix}_ts", create=True, size=ts_size)
            self.shm_id = shared_memory.SharedMemory(name=f"{self.prefix}_id", create=True, size=id_size)
            self.shm_crc = shared_memory.SharedMemory(name=f"{self.prefix}_crc", create=True, size=crc_size)
            self.shm_meta = shared_memory.SharedMemory(name=f"{self.prefix}_meta", create=True, size=meta_size)
            self.shm_hb = shared_memory.SharedMemory(name=f"{self.prefix}_hb", create=True, size=hb_size)
            self.shm_ptr = shared_memory.SharedMemory(name=f"{self.prefix}_ptr", create=True, size=8)
            self.ptr_store = np.ndarray((1,), dtype='int64', buffer=self.shm_ptr.buf)
            self.ptr_store[0] = 0
        else:
            self.shm_data = shared_memory.SharedMemory(name=f"{self.prefix}_data")
            self.shm_ts = shared_memory.SharedMemory(name=f"{self.prefix}_ts")
            self.shm_id = shared_memory.SharedMemory(name=f"{self.prefix}_id")
            self.shm_crc = shared_memory.SharedMemory(name=f"{self.prefix}_crc")
            self.shm_meta = shared_memory.SharedMemory(name=f"{self.prefix}_meta")
            self.shm_hb = shared_memory.SharedMemory(name=f"{self.prefix}_hb")
            self.shm_ptr = shared_memory.SharedMemory(name=f"{self.prefix}_ptr")
            self.ptr_store = np.ndarray((1,), dtype='int64', buffer=self.shm_ptr.buf)

        self.data_store = np.ndarray((self.max_tokens, self.token_dim), dtype='float32', buffer=self.shm_data.buf)
        self.ts_store = np.ndarray((self.max_tokens,), dtype='float64', buffer=self.shm_ts.buf)
        self.id_store = np.ndarray((self.max_tokens, self.modal_id_len), dtype='|S1', buffer=self.shm_id.buf)
        self.crc_store = np.ndarray((self.max_tokens,), dtype='u4', buffer=self.shm_crc.buf)
        self.meta_store = np.ndarray((1024,), dtype=[('id', f'S{self.modal_id_len}'), ('dim', 'i4')], buffer=self.shm_meta.buf)
        self.hb_store = np.ndarray((1024,), dtype='f8', buffer=self.shm_hb.buf)

    def get_modality_shapes(self) -> Dict[str, int]:
        """Return discovered modality shapes from cross-process SHM registry."""
        if HAS_CPP: return {}
        shapes = {}
        for entry in self.meta_store:
            # Decode and strip null bytes
            m_id = entry['id'].decode('utf-8').split('\x00')[0]
            if m_id:
                shapes[m_id] = int(entry['dim'])
        return shapes

    def _register_modality_shm(self, modal_id: str, dim: int):
        """Register a modality in the shared metadata segment."""
        m_bytes = modal_id.encode('utf-8')[:self.modal_id_len]
        
        # Check if already registered or find empty slot
        for entry in self.meta_store:
            # Numpy 'S' types auto-strip nulls on access
            existing_id = entry['id']
            if existing_id == m_bytes:
                return
            if not existing_id:
                entry['id'] = m_bytes
                entry['dim'] = dim
                return

    def publish(self, data: np.ndarray, timestamp: float, modal_id: str, write_ptr_obj: Any = None):
        """
        Publish sensor data to the shared memory bus.
        Uses vectorized circular writes and wait-free pointer updates.
        """
        # Auto-Modality: Record the raw input dimension
        raw_dim = data.flatten().shape[0] if data.ndim == 1 else data.shape[-1]
        if not HAS_CPP:
            self._register_modality_shm(modal_id, raw_dim)

        if HAS_CPP:
            # (C++ logic remains same)
            flat = data.flatten()
            rows = (len(flat) + self.token_dim - 1) // self.token_dim
            padded = np.zeros(rows * self.token_dim, dtype='float32')
            padded[:len(flat)] = flat
            self.backend.publish(padded.reshape(rows, self.token_dim), timestamp, modal_id)
        else:
            # 1. Prepare data rows
            flat = data.flatten()
            num_rows = (len(flat) + self.token_dim - 1) // self.token_dim
            padded = np.zeros(num_rows * self.token_dim, dtype='float32')
            padded[:len(flat)] = flat
            rows = padded.reshape(num_rows, self.token_dim)
            
            # 2. Vectorized Circular Write (No loops)
            # We use the shm_ptr directly (wait-free) or write_ptr_obj (legacy sync)
            start_idx = write_ptr_obj.value if write_ptr_obj else self.ptr_store[0]
            
            # Calculate indices
            end_idx = (start_idx + num_rows) % self.max_tokens
            
            if start_idx + num_rows <= self.max_tokens:
                # Continuous block
                self.data_store[start_idx : start_idx + num_rows] = rows
                self.ts_store[start_idx : start_idx + num_rows] = timestamp
                # Compute CRC32 for the block
                for i in range(num_rows):
                    self.crc_store[start_idx + i] = zlib.crc32(rows[i].tobytes()) & 0xFFFFFFFF
                # Bulk ID write
                id_bytes = modal_id.encode('utf-8')[:self.modal_id_len].ljust(self.modal_id_len, b'\x00')
                id_array = np.frombuffer(id_bytes, dtype='|S1')
                self.id_store[start_idx : start_idx + num_rows] = id_array
            else:
                # Wrapping block
                first_part = self.max_tokens - start_idx
                second_part = num_rows - first_part
                
                # Part 1
                self.data_store[start_idx:] = rows[:first_part]
                self.ts_store[start_idx:] = timestamp
                for i in range(first_part):
                    self.crc_store[start_idx + i] = zlib.crc32(rows[i].tobytes()) & 0xFFFFFFFF
                id_bytes = modal_id.encode('utf-8')[:self.modal_id_len].ljust(self.modal_id_len, b'\x00')
                id_array = np.frombuffer(id_bytes, dtype='|S1')
                self.id_store[start_idx:] = id_array
                
                # Part 2
                self.data_store[:second_part] = rows[first_part:]
                self.ts_store[:second_part] = timestamp
                for i in range(second_part):
                    self.crc_store[i] = zlib.crc32(rows[first_part + i].tobytes()) & 0xFFFFFFFF
                self.id_store[:second_part] = id_array

            # 3. Heartbeat Update
            # Use PID as a simple index hash for heartbeat
            hb_idx = os.getpid() % 1024
            self.hb_store[hb_idx] = time.time()

            # 4. Atomic Pointer Update
            if write_ptr_obj:
                write_ptr_obj.value = end_idx
            else:
                self.ptr_store[0] = end_idx

    def _write_token(self, idx: int, data: np.ndarray, timestamp: float, modal_id: str):
        """Internal low-level write to SHM buffers."""
        self.data_store[idx] = data[:self.token_dim]
        self.ts_store[idx] = timestamp
        id_bytes = modal_id.encode('utf-8')[:self.modal_id_len]
        id_array = np.frombuffer(id_bytes, dtype='|S1')
        self.id_store[idx, :len(id_array)] = id_array
        if len(id_array) < self.modal_id_len:
            self.id_store[idx, len(id_array):] = b'\x00'


    def get_window(self, start_time: float, end_time: float, write_ptr_obj: Any = None) -> List[Dict]:
        """Retrieve all tokens within a time window, sorted by timestamp."""
        if HAS_CPP:
            return self.backend.get_window(start_time, end_time)
        else:
            # Use the same lock as the publisher to prevent inconsistent reads
            lock = write_ptr_obj.get_lock() if write_ptr_obj else None
            
            def perform_read():
                mask = (self.ts_store >= start_time) & (self.ts_store <= end_time)
                indices = np.where(mask)[0]
                results = []
                for idx in indices:
                    mid = b"".join(self.id_store[idx]).decode('utf-8', errors='ignore').strip('\x00')
                    results.append({
                        'data': self.data_store[idx].copy(),
                        'timestamp': self.ts_store[idx],
                        'modal_id': mid
                    })
                return sorted(results, key=lambda x: x['timestamp'])

            if lock:
                with lock:
                    return perform_read()
            else:
                return perform_read()

    def get_since_index(self, last_idx: int, write_ptr_obj: Any = None) -> Tuple[List[Dict], int]:
        """
        Get all tokens between last_idx and current write_ptr.
        Includes overrun detection.
        """
        if HAS_CPP:
            return [], last_idx
        
        # Capture current pointer (Atomic Read)
        current_idx = write_ptr_obj.value if write_ptr_obj else self.ptr_store[0]
        
        if current_idx == last_idx:
            return [], last_idx
        
        # Overrun detection: if we are too far behind, we've lost data
        diff = (current_idx - last_idx) % self.max_tokens
        if diff > self.max_tokens * 0.9:
            # Log overrun
            logging.warning(f"[TokenBus] Consumer overrun detected ({diff} tokens behind). Data lost.")

        results = []
        if current_idx > last_idx:
            indices = range(last_idx, current_idx)
        else:
            indices = list(range(last_idx, self.max_tokens)) + list(range(0, current_idx))
        
        # Batch extract IDs for speed
        m_ids = [b"".join(self.id_store[i]).decode('utf-8', errors='ignore').strip('\x00') for i in indices]
        
        for i, idx in enumerate(indices):
            if self.ts_store[idx] == 0: continue
            results.append({
                'data': self.data_store[idx].copy(),
                'timestamp': self.ts_store[idx],
                'modal_id': m_ids[i]
            })
        return results, current_idx


    def buffer_size(self) -> int:
        """Return the number of active slots in the buffer."""
        if HAS_CPP:
            return self.max_tokens  # C++ backend manages its own internal state
        return int((self.ts_store > 0).sum())

    def cleanup(self):
        """Release shared memory resources."""
        if HAS_CPP:
            self.backend.cleanup(self.create)
        else:
            self.shm_data.close()
            self.shm_ts.close()
            self.shm_id.close()
            self.shm_crc.close()
            self.shm_meta.close()
            self.shm_hb.close()
            self.shm_ptr.close()
            if self.create:
                try:
                    self.shm_data.unlink()
                    self.shm_ts.unlink()
                    self.shm_id.unlink()
                    self.shm_crc.unlink()
                    self.shm_meta.unlink()
                    self.shm_hb.unlink()
                    self.shm_ptr.unlink()
                    # Remove registry entry
                    reg_file = os.path.join(SHM_REGISTRY_DIR, f"{self.sid}.json")
                    if os.path.exists(reg_file): os.remove(reg_file)
                except Exception:
                    pass


    def __getstate__(self):
        state = self.__dict__.copy()
        if not HAS_CPP:
            del state['shm_data']
            del state['shm_ts']
            del state['shm_id']
            del state['shm_crc']
            del state['shm_meta']
            del state['shm_hb']
            del state['shm_ptr']
            del state['data_store']
            del state['ts_store']
            del state['id_store']
            del state['crc_store']
            del state['meta_store']
            del state['hb_store']
            del state['ptr_store']

        state['create'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if HAS_CPP:
            self.backend = omni_bus_core.NativeTokenBus(self.max_tokens, self.token_dim, self.modal_id_len, self.sid, False)
        else:
            self._attach_python_backend()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @staticmethod
    def force_cleanup_zombies():
        """
        FIX #22: Robust SHM Reaper. Checks active PIDs and unlinks orphaned segments.
        """
        if HAS_CPP: return
        
        from multiprocessing import shared_memory
        import glob
        
        # 1. Clean via registry (Fast & Precise)
        if os.path.exists(SHM_REGISTRY_DIR):
            for reg_file in glob.glob(os.path.join(SHM_REGISTRY_DIR, "*.json")):
                try:
                    with open(reg_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if PID is still alive
                    pid_exists = False
                    try:
                        os.kill(data['pid'], 0)
                        pid_exists = True
                    except OSError:
                        pid_exists = False
                    
                    if not pid_exists:
                        sid = data['sid']
                        prefix = f"omni_shm_{sid}"
                        for suffix in ["data", "ts", "id", "meta", "ptr"]:
                            try:
                                shm = shared_memory.SharedMemory(name=f"{prefix}_{suffix}")
                                shm.unlink()
                                shm.close()
                            except: pass
                        os.remove(reg_file)
                        print(f"[TokenBus] Reaper: Cleaned orphaned session {sid}")
                except: pass

        # 2. Fallback: Brute force sweep of /dev/shm (Linux only)
        if os.path.exists('/dev/shm'):
            for f in glob.glob('/dev/shm/omni_shm_*'):
                name = os.path.basename(f)
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                    shm.close()
                except: pass
