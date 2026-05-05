import yaml
import importlib
import multiprocessing
import signal
import sys
import os
from .token_bus import TokenBus
from .monitor import run_monitor
from rich.console import Console

console = Console()


def parse_and_launch(yaml_path: str):
    """
    OmniTrain process orchestrator. Manages the lifecycle of plugins and monitoring.
    """
    if not os.path.exists(yaml_path):
        console.print(f"[red]ERROR[/] Config not found: [white]{yaml_path}[/]")
        return

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Extract Model Config
    model_cfg = config.get('model', {})
    d_model = model_cfg.get('d_model', 512)
    n_latents = model_cfg.get('n_latents', 128)

    console.print(f"[white]INFO[/] LiquidFusionCore: d={d_model}, latents={n_latents}")
    # Lazy import to avoid CUDA initialization in main process if not needed
    from .fusion_core import LiquidFusionCore
    model = LiquidFusionCore(n_latents=n_latents, d_model=d_model, config=config)
    model.eval()

    # 3. Initialize Transport
    
    
    sid = config.get('transport', {}).get('session_id')
    bus = TokenBus(max_tokens=1000, create=True, session_id=sid)
    write_ptr = multiprocessing.Value('i', 0)

    workers = []

    for input_cfg in config.get('inputs', []):
        # Flexible key naming for modal_id and hz
        modal_id = input_cfg.get('id') or input_cfg.get('modal_id', 'unknown')
        freq = float(input_cfg.get('frequency') or input_cfg.get('hz', 10.0))
        plugin_path = input_cfg.get('plugin')

        try:
            mod_name, cls_name = plugin_path.rsplit('.', 1)
            
            # Security: Whitelist allowed plugin sources to prevent arbitrary code execution
            
            allowed_prefixes = ['omnitrain.']
            is_allowed = any(mod_name.startswith(p) for p in allowed_prefixes)
            
            if not is_allowed:
                raise ImportError(f"Unauthorized plugin source: {mod_name}. Only plugins from {allowed_prefixes} are allowed.")

            plugin_class = getattr(importlib.import_module(mod_name), cls_name)

            kwargs = {k: v for k, v in input_cfg.items() if k not in ['modal_id', 'id', 'plugin', 'hz', 'frequency']}
            instance = plugin_class(bus, modal_id, freq, write_ptr=write_ptr, **kwargs)

            p = multiprocessing.Process(target=instance.run, name=f"Worker-{modal_id}", daemon=True)
            p.start()
            workers.append(p)
            console.print(f"[color(117)]OK[/] Worker started: [white]{modal_id}[/] at {freq}Hz")

        except Exception as e:
            console.print(f"[red]ERROR[/] Worker spawn failure: [white]{modal_id}[/] ({e})")

    def graceful_exit(sig, frame):
        console.print("\n[yellow]ABORT[/] Terminating workers and cleaning bus...")
        for p in workers:
            p.terminate()
        bus.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    try:
        run_monitor(bus)
    except Exception as e:
        console.print(f"[red]ERROR[/] Monitor failure: {e}")
        graceful_exit(None, None)
