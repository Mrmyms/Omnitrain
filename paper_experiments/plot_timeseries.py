import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Ensure omnitrain is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train_and_compare import ContinuousCfC, DiscreteRNN, train_model

def generate_plot():
    print("Loading data...")
    X_60 = np.load("data/pendulum_X_60loss.npy")
    X_0 = np.load("data/pendulum_X_0loss.npy")
    Y = np.load("data/pendulum_Y.npy")
    T = np.load("data/pendulum_T.npy")
    
    # Normalize inputs! (Fix)
    X_mean, X_std = X_0.mean(0), X_0.std(0) + 1e-8
    X_0_n = (X_0 - X_mean) / X_std
    X_60_n = (X_60 - X_mean) / X_std
    
    hidden_dim = 16
    torch.manual_seed(42)
    epochs = 150
    
    # Train CfC
    print("Training CfC...")
    cfc = ContinuousCfC(4, hidden_dim, 1, backbone_units=32)
    cfc = train_model(cfc, X_0_n, Y, T, epochs=epochs)
    
    # Train LSTM
    print("Training LSTM...")
    lstm = DiscreteRNN(4, hidden_dim, 1, 'lstm')
    lstm = train_model(lstm, X_0_n, Y, epochs=epochs)

    # Train GRU
    print("Training GRU...")
    gru = DiscreteRNN(4, hidden_dim, 1, 'gru')
    gru = train_model(gru, X_0_n, Y, epochs=epochs)
    
    print("Evaluating sequences...")
    cfc.eval()
    lstm.eval()
    gru.eval()
    
    X_60_full = torch.FloatTensor(X_60_n).unsqueeze(0)
    T_full = torch.FloatTensor(T).unsqueeze(0)
    
    with torch.no_grad():
        cfc_pred_full = cfc(X_60_full, T_full).squeeze().numpy()
        lstm_pred_full = lstm(X_60_full).squeeze().numpy()
        gru_pred_full = gru(X_60_full).squeeze().numpy()
        
    start_idx = 500
    end_idx = 650
    
    Y_slice = Y[start_idx:end_idx, 0]
    time_axis = T[start_idx:end_idx, 0]
    cfc_pred = cfc_pred_full[start_idx:end_idx]
    lstm_pred = lstm_pred_full[start_idx:end_idx]
    gru_pred = gru_pred_full[start_idx:end_idx]
        
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.plot(time_axis, Y_slice, label="Ground Truth", color='black', linewidth=2)
    plt.plot(time_axis, lstm_pred, label="LSTM (60% Loss)", color='red', linestyle='--')
    plt.plot(time_axis, gru_pred, label="GRU (60% Loss)", color='orange', linestyle='-.')
    plt.plot(time_axis, cfc_pred, label="CfC (60% Loss)", color='blue')
    
    plt.title("Qualitative Time-Series Comparison Under 60% Packet Loss")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Force")
    plt.legend()
    plt.grid(True)
    
    out_path = "data/timeseries_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    generate_plot()
