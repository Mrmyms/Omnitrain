# OmniTrain 1.0.0: Liquid Intelligence
### Industrial Liquid Neural Networks & Formal Safety for Robotics

---

OmniTrain is a state-of-the-art framework for building **Liquid Neural Networks (CfC)** and **Input Convex Neural Networks (ICNN)** for industrial robotics. It focuses on sub-millisecond latency, time-continuous reasoning, and provable safety.

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/Mrmyms/Omnitrain.git
cd omnitrain
pip install -r requirements.txt
```

### 2. Launch the Console
```bash
# Using the 1omni alias (recommended)
1omni

# Or manually
PYTHONPATH=src python3 -m omnitrain.cli
```

### 3. Universal Ingestion
```python
from omnitrain.omni_stream import OmniStream

stream = OmniStream(core, shield)
result = stream.send({"lidar": 1.5, "camera": "frame.jpg"})
action = result['action'] # Safe action via OmniShield
```

---

## Core Pillars

*   **Liquid Brain (CfC)**: Closed-form Continuous-time networks for extreme OOD robustness and temporal stability.
*   **OmniShield (ICNN)**: Formal safety verification using Input Convex Neural Networks to guarantee safe operational boundaries.
*   **OmniStream**: Universal sensor ingestion that auto-detects and transforms any data type (Vision, Vector, Scalar).
*   **TokenBus C++**: High-speed Shared Memory transport for sub-millisecond inter-process communication.

---

## Architecture

```mermaid
graph TD
    A[Universal Sensors] -->|OmniStream| B(LiquidFusionCore)
    B -->|Latents| C{OmniShield v2}
    C -->|Safe| D[Motor Command]
    C -->|Danger| E[Emergency Stop / Projection]
    style C fill:#f96,stroke:#333,stroke-width:2px
```

---

## Resources

*   **[Technical Deep Dive](docs/DETAILS.md)**: CfC cells and ICNN barriers.
*   **[Theoretical Frameworks](docs/THEORETICAL_FRAMEWORKS.md)**: Liquid Networks, ICNNs, and CTMT math.
*   **[System Capabilities](docs/CAPABILITIES.md)**: Full feature specification and Capabilities Paper.
*   **[Training Pipeline](docs/TRAINING_PIPELINE.md)**: 3-phase curriculum (Imitation, Safety, Chaos).
*   **[CLI Reference](docs/DETAILS.md#cli-reference)**: Management console usage.

---

**OmniTrain Team**
"Fuse Everything. Trust Nothing. Verify Formally."
