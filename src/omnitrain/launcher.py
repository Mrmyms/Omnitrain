import yaml
import importlib
import multiprocessing
import signal
import sys
import os
from .token_bus import TokenBus
from .monitor import run_monitor


def parse_and_launch(yaml_path: str):
    """
    OmniTrain process orchestrator. Manages the lifecycle of plugins and monitoring.
    """
    if not os.path.exists(yaml_path):
        print(f"[Launcher] Config error: {yaml_path} not found.")
        return

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Extract Model Config
    model_cfg = config.get('model', {})
    d_model = model_cfg.get('d_model', 512)
    n_latents = model_cfg.get('n_latents', 128)

    print(f"🧠 Initializing FusionCore (d={d_model}, latents={n_latents})")
    # Lazy import to avoid CUDA initialization in main process if not needed
    from .fusion_core import FusionCore
    model = FusionCore(n_latents=n_latents, d_model=d_model)
    model.eval()

    # 3. Initialize Transport
    bus = TokenBus(max_tokens=2048, create=True)
    write_ptr = multiprocessing.Value('i', 0)

    workers = []

    for input_cfg in config.get('inputs', []):
        # Flexible key naming for modal_id and hz
        modal_id = input_cfg.get('id') or input_cfg.get('modal_id', 'unknown')
        freq = float(input_cfg.get('frequency') or input_cfg.get('hz', 10.0))
        plugin_path = input_cfg.get('plugin')

        try:
            mod_name, cls_name = plugin_path.rsplit('.', 1)
            plugin_class = getattr(importlib.import_module(mod_name), cls_name)

            kwargs = {k: v for k, v in input_cfg.items() if k not in ['modal_id', 'id', 'plugin', 'hz', 'frequency']}
            instance = plugin_class(bus, modal_id, freq, write_ptr=write_ptr, **kwargs)

            p = multiprocessing.Process(target=instance.run, name=f"Worker-{modal_id}", daemon=True)
            p.start()
            workers.append(p)
            print(f"[Launcher] OK: {modal_id} at {freq}Hz")

        except Exception as e:
            print(f"[Launcher] Error spawning {modal_id}: {e}")

    def graceful_exit(sig, frame):
        print("\n[Launcher] System signal received. Cleaning up...")
        for p in workers:
            p.terminate()
        bus.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    try:
        run_monitor(bus)
    except Exception as e:
        print(f"[Launcher] Monitor Error: {e}")
        graceful_exit(None, None)
