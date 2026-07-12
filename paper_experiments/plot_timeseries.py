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
    
    hidden_dim = 16
    torch.manual_seed(42)
    
    # Train CfC properly
    print("Training CfC...")
    cfc = ContinuousCfC(4, hidden_dim, 1)
    cfc = train_model(cfc, X_0, Y, T, epochs=50)
    
    # Train LSTM properly
    print("Training LSTM...")
    lstm = DiscreteRNN(4, hidden_dim, 1, 'lstm')
    lstm = train_model(lstm, X_0, Y, epochs=50)
    
    print("Evaluating sequences...")
    cfc.eval()
    lstm.eval()
    
    # Evaluate on the full sequence first to maintain hidden state
    X_60_full = torch.FloatTensor(X_60).unsqueeze(0)
    T_full = torch.FloatTensor(T).unsqueeze(0)
    
    with torch.no_grad():
        cfc_pred_full = cfc(X_60_full, T_full).squeeze().numpy()
        lstm_pred_full = lstm(X_60_full).squeeze().numpy()
        
    # Take a 150-step slice for the plot
    start_idx = 500
    end_idx = 650
    
    Y_slice = Y[start_idx:end_idx, 0]
    time_axis = T[start_idx:end_idx, 0]
    cfc_pred = cfc_pred_full[start_idx:end_idx]
    lstm_pred = lstm_pred_full[start_idx:end_idx]
        
    # Plotting
    plt.figure(figsize=(10, 5))
    time_axis = T[start_idx:end_idx, 0]
    
    plt.plot(time_axis, Y_slice, label="Ground Truth (Target Force)", color='black', linewidth=2)
    plt.plot(time_axis, lstm_pred, label="LSTM (60% Packet Loss)", color='red', linestyle='--')
    plt.plot(time_axis, cfc_pred, label="CfC (60% Packet Loss)", color='blue')
    
    plt.title("Qualitative Time-Series Comparison Under 60% Packet Loss")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Force")
    plt.legend()
    plt.grid(True)
    
    out_path = "../ai_context/timeseries_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    generate_plot()
