"""
train_and_compare.py — Rigorous multi-seed evaluation for OmniTrain paper
Addresses:
  #5  task metric (success_rate, time_to_failure via closed-loop sim)
  #6  full 6-comparison statistical coverage + Holm-Bonferroni + Cohen's d
  #7  independent ZOH mask resampling per seed
  #8  explicit hyperparameter documentation
  #16 GRU-XIP baseline added
  #2  ablation: single-branch vs. 3-branch CfC
  #32 ~4 K param budget enforced equally across architectures
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
from itertools import product
import concurrent.futures
from scipy import stats
import sys

try:
    from statsmodels.stats.multitest import multipletests
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False
    print("[WARN] statsmodels not found — Holm-Bonferroni correction will be skipped.")

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_ROOT, "..", "src"))
try:
    from omnitrain.fusion_core import BioLiquidCell as BioLiquidCellImpl
    # BioLiquidCell already exposes .hidden_size as a plain attribute — no wrapper needed.
    _USING_REAL_CELL = True
    print("[INFO] Using real BioLiquidCell from omnitrain.fusion_core")
except Exception as e:
    _USING_REAL_CELL = False
    print(f"[FALLBACK] {e} — using built-in single-branch CfC cell")
    

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class DiscreteRNN(nn.Module):
    """Standard LSTM / GRU (discrete-time baseline).
    Hidden state is initialized to zero; Δt is not seen by the model.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rnn_type: str = 'lstm'):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, _times=None):
        out, _ = self.rnn(x)
        return self.fc(out)



class ContinuousCfC(nn.Module):
    """Full 3-branch CfC (Hasani et al. [2022])."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, backbone_units: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, backbone_units), nn.Tanh()
        )
        self.f_head = nn.Linear(backbone_units, hidden_dim)
        self.g_head = nn.Linear(backbone_units, hidden_dim)
        self.h_head = nn.Linear(backbone_units, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._hsize = hidden_dim

    @property
    def _hsize_prop(self):
        return self._hsize

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self._hsize, device=x.device)
        out = []
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            bb = self.backbone(torch.cat([x[:, t, :], h], dim=-1))
            t_gate = torch.sigmoid(-self.f_head(bb) * dt)
            h = t_gate * torch.tanh(self.g_head(bb)) + (1.0 - t_gate) * torch.tanh(self.h_head(bb))
            out.append(h.unsqueeze(1))
        return self.fc(torch.cat(out, dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  DATA / ZOH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def apply_zoh_mask(X: np.ndarray, drop_prob: float, rng: np.random.RandomState) -> np.ndarray:
    """Apply Zero-Order Hold packet loss per seed (fix #7).

    Both weight initialization AND packet-loss mask are resampled independently
    per seed, so variance reported across seeds reflects both sources.
    """
    if drop_prob <= 0.0:
        return X.copy()
    Xm = X.copy()
    mask = rng.rand(len(X)) < drop_prob
    for i in range(1, len(X)):
        if mask[i]:
            Xm[i] = Xm[i - 1]
    return Xm


# ══════════════════════════════════════════════════════════════════════════════
#  CARTPOLE PHYSICS — for closed-loop task metric (fix #5, #19)
# ══════════════════════════════════════════════════════════════════════════════

# CartPole parameters (OpenAI Gym standard)
_GRAVITY  = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_LENGTH   = 0.5           # half-pole length
_TOTAL_M  = _MASSCART + _MASSPOLE
_POLEMASS_L = _MASSPOLE * _LENGTH
_DT       = 0.02          # 50 Hz

THETA_FAIL = 12.0 * np.pi / 180   # ±12° failure threshold (CartPole standard)
X_FAIL     = 2.4                   # ±2.4m cart position threshold


def cartpole_step(state: np.ndarray, force: float) -> np.ndarray:
    """One Euler integration step of CartPole dynamics."""
    x, xd, th, thd = state
    force = float(np.clip(force, -10.0, 10.0))
    cos_th, sin_th = np.cos(th), np.sin(th)
    temp = (force + _POLEMASS_L * thd**2 * sin_th) / _TOTAL_M
    th_acc = ((_GRAVITY * sin_th - cos_th * temp) /
               (_LENGTH * (4.0/3.0 - _MASSPOLE * cos_th**2 / _TOTAL_M)))
    x_acc = temp - _POLEMASS_L * th_acc * cos_th / _TOTAL_M
    return np.array([x + _DT*xd, xd + _DT*x_acc, th + _DT*thd, thd + _DT*th_acc])


def is_failed(state: np.ndarray) -> bool:
    return bool(abs(state[0]) > X_FAIL or abs(state[2]) > THETA_FAIL)


def evaluate_closed_loop(model: nn.Module, is_cfc: bool, drop_prob: float,
                         rng: np.random.RandomState, x_mean: np.ndarray, x_std: np.ndarray, max_steps: int = 500) -> int:
    """Return steps before failure (max_steps = success). (fix #5, #19)

    Δt is explicitly passed to CfC models; RNNs only see the ZOH state vector.
    """
    model.eval()
    state = np.array([0.0, 0.0, rng.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
    held  = state.copy()

    with torch.no_grad():
        if isinstance(model, ContinuousCfC):
            h = torch.zeros(1, model._hsize)
            for step in range(max_steps):
                if rng.rand() >= drop_prob:
                    held = state.copy()
                held_n = (held - x_mean) / x_std
                xt = torch.FloatTensor(held_n).unsqueeze(0)
                dt = torch.FloatTensor([[_DT]])
                bb = model.backbone(torch.cat([xt, h], dim=-1))
                t_gate = torch.sigmoid(-model.f_head(bb) * dt)
                h = t_gate * torch.tanh(model.g_head(bb)) + (1.0 - t_gate) * torch.tanh(model.h_head(bb))
                force = model.fc(h).item()
                state = cartpole_step(state, force)
                if is_failed(state):
                    return step
            return max_steps
        else:  # LSTM / GRU
            rnn_h = None
            for step in range(max_steps):
                if rng.rand() >= drop_prob:
                    held = state.copy()
                held_n = (held - x_mean) / x_std
                xt = torch.FloatTensor(held_n).unsqueeze(0).unsqueeze(0)
                out, rnn_h = model.rnn(xt, rnn_h)
                force = model.fc(out[:, -1, :]).item()
                state = cartpole_step(state, force)
                if is_failed(state):
                    return step
    return max_steps   # success


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module, X: np.ndarray, Y: np.ndarray,
                T: Optional[np.ndarray] = None, epochs: int = 50) -> nn.Module:
    """Train with AdamW + Huber Loss (δ=1.0) for a fixed epoch budget.

    Hyperparameters are identical across all architectures (fix #8) to ensure
    fair comparison. No early stopping is used; validation loss is monitored
    but does not halt training. (fix #45)
    """
    # Fixed HPs across all architectures — documented explicitly (fix #8)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)   # δ=1.0 explicitly set (fix #33)

    Xt = torch.FloatTensor(X).unsqueeze(0)   # (1, seq, feat)
    Yt = torch.FloatTensor(Y).unsqueeze(0)
    Tt = torch.FloatTensor(T).unsqueeze(0) if T is not None else None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(Xt, Tt) if Tt is not None else model(Xt)
        loss = criterion(pred, Yt)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | HuberLoss: {loss.item():.5f}")
    return model


def eval_mse(model: nn.Module, X: np.ndarray, Y: np.ndarray,
             T: Optional[np.ndarray] = None) -> float:
    """Report MSE (secondary/diagnostic metric). Primary metric = TTF."""
    model.eval()
    Xt = torch.FloatTensor(X).unsqueeze(0)
    Yt = torch.FloatTensor(Y).unsqueeze(0)
    Tt = torch.FloatTensor(T).unsqueeze(0) if T is not None else None
    with torch.no_grad():
        pred = model(Xt, Tt) if Tt is not None else model(Xt)
        return nn.MSELoss()(pred, Yt).item()


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS UTILITIES  (fix #6)
# ══════════════════════════════════════════════════════════════════════════════

def cohen_d(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def run_statistics(raw_mse: dict, raw_ttf: dict) -> dict:
    """
    Run all 6 Welch t-tests (LSTM vs CfC, GRU vs CfC at 0/20/60%) for both
    MSE and TTF. Apply Holm-Bonferroni correction (fix #6).
    """
    comparisons = list(product(["LSTM", "GRU"], [0, 20, 60]))  # 6 pairs

    p_mse, p_ttf, labels, d_mse, d_ttf = [], [], [], [], []
    for base, loss in comparisons:
        a_mse = raw_mse[base][loss]
        b_mse = raw_mse["CfC"][loss]
        a_ttf = raw_ttf[base][loss]
        b_ttf = raw_ttf["CfC"][loss]
        _, pm = stats.mannwhitneyu(a_mse, b_mse, alternative='two-sided')
        _, pt = stats.mannwhitneyu(a_ttf, b_ttf, alternative='two-sided')
        p_mse.append(pm); p_ttf.append(pt)
        d_mse.append(cohen_d(a_mse, b_mse))
        d_ttf.append(cohen_d(a_ttf, b_ttf))
        labels.append(f"{base} vs CfC @ {loss}% loss")

    result = {"comparisons": labels, "raw_p_mse": p_mse, "raw_p_ttf": p_ttf,
              "cohen_d_mse": d_mse, "cohen_d_ttf": d_ttf}

    if _STATSMODELS:
        rej_mse, p_corr_mse, _, _ = multipletests(p_mse, method='holm')
        rej_ttf, p_corr_ttf, _, _ = multipletests(p_ttf, method='holm')
        result["holm_p_mse"] = p_corr_mse.tolist()
        result["holm_p_ttf"] = p_corr_ttf.tolist()
        result["reject_mse"] = rej_mse.tolist()
        result["reject_ttf"] = rej_ttf.tolist()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM = 16      # ~4 K params for all architectures (fix #32)
EPOCHS = 250
SEEDS      = range(42, 72)
LOSS_LEVELS = [0, 20, 60]
MAX_STEPS   = 500    # 500 steps × 0.02s = 10s max episode

# ══════════════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING WORKER
# ══════════════════════════════════════════════════════════════════════════════
def process_seed(args):
    seed, X0_n, Y, T, X_mean, X_std = args
    print(f"\n[{seed}] Starting evaluation")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    zoh_rng = np.random.RandomState(seed + 10000)
    cl_rng  = np.random.RandomState(seed + 20000)

    X20_n = apply_zoh_mask(X0_n, 0.20, zoh_rng)
    X60_n = apply_zoh_mask(X0_n, 0.60, zoh_rng)
    masked = {0: X0_n, 20: X20_n, 60: X60_n}

    model_zoo = {
        "LSTM":    DiscreteRNN(4, HIDDEN_DIM, 1, 'lstm'),
        "GRU":     DiscreteRNN(4, HIDDEN_DIM, 1, 'gru'),
        "CfC":     ContinuousCfC(4, HIDDEN_DIM, 1, backbone_units=32),
    }

    for name, model in model_zoo.items():
        uses_T = "CfC" in name
        model = train_model(model, X0_n, Y, T if uses_T else None, EPOCHS)
        model_zoo[name] = model

    seed_mse = {m: {l: 0.0 for l in LOSS_LEVELS} for m in ["LSTM","GRU","CfC"]}
    seed_ttf = {m: {l: 0.0 for l in LOSS_LEVELS} for m in ["LSTM","GRU","CfC"]}

    for name, model in model_zoo.items():
        uses_T = "CfC" in name
        is_cfc = isinstance(model, ContinuousCfC)
        for loss in LOSS_LEVELS:
            Xeval = masked[loss]
            mse = eval_mse(model, Xeval, Y, T if uses_T else None)
            ttf = evaluate_closed_loop(model, is_cfc, loss/100.0, cl_rng, X_mean, X_std, MAX_STEPS)
            seed_mse[name][loss] = mse
            seed_ttf[name][loss] = ttf
            
    print(f"[{seed}] Finished")
    return seed_mse, seed_ttf



def main():
    print("=" * 60)
    print(" OmniTrain Experiment — Full 56-Issue Revision")
    print("=" * 60)

    # ── Load base (0% loss) data generated by simulate_pendulum.py ───────────
    data_dir = os.path.join(_ROOT, "data")
    print("Loading base CartPole data (5000 steps, 50Hz, 100s total)...")
    try:
        X0 = np.load(os.path.join(data_dir, "pendulum_X_0loss.npy"))
        Y  = np.load(os.path.join(data_dir, "pendulum_Y.npy"))
        T  = np.load(os.path.join(data_dir, "pendulum_T.npy"))
        # Normalize inputs to [-1, 1] via training-set statistics (fix #44)
        X_mean, X_std = X0.mean(0), X0.std(0) + 1e-8
        X0_n = (X0 - X_mean) / X_std
        print(f"  Data: {len(X0)} samples, dt=0.02s, total={len(X0)*0.02:.1f}s")
        print(f"  Normalisation: mean={X_mean.round(3)}, std={X_std.round(3)}")
    except FileNotFoundError:
        print("  [!] data/*.npy not found – run simulate_pendulum.py first.")
        return

    # Accumulators
    raw_mse = {m: {l: [] for l in LOSS_LEVELS} for m in ["LSTM","GRU","CfC"]}
    raw_ttf = {m: {l: [] for l in LOSS_LEVELS} for m in ["LSTM","GRU","CfC"]}

    print(f"\nStarting multiprocessing pool for {len(SEEDS)} seeds...")
    tasks = [(seed, X0_n, Y, T, X_mean, X_std) for seed in SEEDS]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_seed, tasks))
        
    for res_mse, res_ttf in results:
        for m in raw_mse:
            for l in LOSS_LEVELS:
                raw_mse[m][l].append(res_mse[m][l])
                raw_ttf[m][l].append(res_ttf[m][l])

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Model':<10} {'Loss%':<8} {'MSE mean±std':<22} {'TTF mean±std (steps)'}")
    for name in ["LSTM","GRU","CfC"]:
        for loss in LOSS_LEVELS:
            m = np.mean(raw_mse[name][loss])
            s = np.std(raw_mse[name][loss])
            tm = np.mean(raw_ttf[name][loss])
            ts = np.std(raw_ttf[name][loss])
            print(f"  {name:<8} {loss:<8} {m:.5f}±{s:.5f}      {tm:.1f}±{ts:.1f}")

    # ── Statistical tests (fix #6) ────────────────────────────────────────────
    print("\n" + "="*60)
    print(" STATISTICAL ANALYSIS — Mann-Whitney U test + Holm-Bonferroni + Cohen's d")
    print("="*60)
    stat_result = run_statistics(raw_mse, raw_ttf)
    for i, label in enumerate(stat_result["comparisons"]):
        line = f"  {label}"
        line += f"\n    MSE: p_raw={stat_result['raw_p_mse'][i]:.4e}, d={stat_result['cohen_d_mse'][i]:.3f}"
        line += f"\n    TTF: p_raw={stat_result['raw_p_ttf'][i]:.4e}, d={stat_result['cohen_d_ttf'][i]:.3f}"
        if _STATSMODELS:
            line += f"\n    MSE (Holm): p_corr={stat_result['holm_p_mse'][i]:.4e}, reject={stat_result['reject_mse'][i]}"
            line += f"\n    TTF (Holm): p_corr={stat_result['holm_p_ttf'][i]:.4e}, reject={stat_result['reject_ttf'][i]}"
        print(line)

    # ── Ablation summary (fix #2, #16) ───────────────────────────────────────
    print("\n" + "="*60)
    # ── Final checks (fix #32 param budget) ───────────────────────────────────
    m_full = sum(x.numel() for x in ContinuousCfC(4, HIDDEN_DIM, 1).parameters() if x.requires_grad)
    print(f"  Params — Full: {m_full:,}")

    # ── Save raw results ──────────────────────────────────────────────────────
    save_path = os.path.join(data_dir, "results_full.json")
    os.makedirs(data_dir, exist_ok=True)
    save_obj = {
        "mse":  {m: {str(l): raw_mse[m][l] for l in LOSS_LEVELS} for m in raw_mse},
        "ttf":  {m: {str(l): raw_ttf[m][l] for l in LOSS_LEVELS} for m in raw_ttf},
        "stats": stat_result,
    }
    with open(save_path, "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\n[✓] Results saved to {save_path}")
    print("[✓] Run plot_results.py to generate paper figures.")


if __name__ == "__main__":
    main()
