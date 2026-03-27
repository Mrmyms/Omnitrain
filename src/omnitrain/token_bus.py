import time
import uuid
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any

try:
    # Attempt to load the high-performance C++ backend
    import omni_bus_core
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


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
        self.sid = session_id or uuid.uuid4().hex[:8]
        self.create = create
        # Auto-Modality: Track discovered sensor shapes
        self._shape_registry: Dict[str, int] = {}

        if HAS_CPP:
            print(f"[TokenBus] Industrial C++ Backend Active (SID: {self.sid})")
            self.backend = omni_bus_core.NativeTokenBus(max_tokens, token_dim, modal_id_len, self.sid, create)
        else:
            print(f"[TokenBus] Standard Python Backend Active (SID: {self.sid})")
            from multiprocessing import shared_memory
            self._attach_python_backend()

    def _attach_python_backend(self):
        from multiprocessing import shared_memory
        self._lock = mp.Lock()
        self.prefix = f"omni_{self.sid}"

        data_size = self.max_tokens * self.token_dim * 4
        ts_size = self.max_tokens * 8
        id_size = self.max_tokens * self.modal_id_len

        if self.create:
            self.shm_data = shared_memory.SharedMemory(name=f"{self.prefix}_data", create=True, size=data_size)
            self.shm_ts = shared_memory.SharedMemory(name=f"{self.prefix}_ts", create=True, size=ts_size)
            self.shm_id = shared_memory.SharedMemory(name=f"{self.prefix}_id", create=True, size=id_size)
        else:
            self.shm_data = shared_memory.SharedMemory(name=f"{self.prefix}_data")
            self.shm_ts = shared_memory.SharedMemory(name=f"{self.prefix}_ts")
            self.shm_id = shared_memory.SharedMemory(name=f"{self.prefix}_id")

        self.data_store = np.ndarray((self.max_tokens, self.token_dim), dtype='float32', buffer=self.shm_data.buf)
        self.ts_store = np.ndarray((self.max_tokens,), dtype='float64', buffer=self.shm_ts.buf)
        self.id_store = np.ndarray((self.max_tokens, self.modal_id_len), dtype='|S1', buffer=self.shm_id.buf)

    def get_modality_shapes(self) -> Dict[str, int]:
        """Return discovered modality shapes for Auto-Modality configuration."""
        return dict(self._shape_registry)

    def publish(self, data: np.ndarray, timestamp: float, modal_id: str, write_ptr_obj: Any = None):
        """Publish sensor data to the shared memory bus. Auto-registers modality shape."""
        # Auto-Modality: Record the raw input dimension for this modality
        raw_dim = data.flatten().shape[0] if data.ndim == 1 else data.shape[-1]
        self._shape_registry[modal_id] = raw_dim

        if HAS_CPP:
            # Ensure data is padded to token_dim before C++ ingest
            flat = data.flatten()
            rows = (len(flat) + self.token_dim - 1) // self.token_dim
            padded = np.zeros(rows * self.token_dim, dtype='float32')
            padded[:len(flat)] = flat
            self.backend.publish(padded.reshape(rows, self.token_dim), timestamp, modal_id)
        else:
            rows = data.reshape(-1, self.token_dim)
            for row in rows:
                with write_ptr_obj.get_lock():
                    idx = write_ptr_obj.value
                    self.data_store[idx] = row[:self.token_dim]
                    self.ts_store[idx] = timestamp
                    id_bytes = modal_id.encode('utf-8')[:self.modal_id_len]
                    id_array = np.frombuffer(id_bytes, dtype='|S1')
                    self.id_store[idx, :len(id_array)] = id_array
                    if len(id_array) < self.modal_id_len:
                        self.id_store[idx, len(id_array):] = b'\x00'
                    write_ptr_obj.value = (idx + 1) % self.max_tokens

    def get_window(self, start_time: float, end_time: float) -> List[Dict]:
        """Retrieve all tokens within a time window, sorted by timestamp."""
        if HAS_CPP:
            return self.backend.get_window(start_time, end_time)
        else:
            with self._lock:
                mask = (self.ts_store >= start_time) & (self.ts_store <= end_time)
                indices = np.where(mask)[0]
                results = []
                for idx in indices:
                    mid = b"".join(self.id_store[idx]).decode('utf-8').strip('\x00')
                    results.append({
                        'data': self.data_store[idx].copy(),
                        'timestamp': self.ts_store[idx],
                        'modal_id': mid
                    })
                return sorted(results, key=lambda x: x['timestamp'])

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
            if self.create:
                try:
                    self.shm_data.unlink()
                    self.shm_ts.unlink()
                    self.shm_id.unlink()
                except Exception:
                    pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if not HAS_CPP:
            del state['shm_data']
            del state['shm_ts']
            del state['shm_id']
            del state['data_store']
            del state['ts_store']
            del state['id_store']
        state['create'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if HAS_CPP:
            self.backend = omni_bus_core.NativeTokenBus(self.max_tokens, self.token_dim, self.modal_id_len, self.sid, False)
        else:
            self._attach_python_backend()
