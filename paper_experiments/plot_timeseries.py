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
    
    # Train CfC fast
    print("Training CfC...")
    cfc = ContinuousCfC(4, hidden_dim, 1)
    cfc = train_model(cfc, X_0, Y, T, epochs=20)
    
    # Train LSTM fast
    print("Training LSTM...")
    lstm = DiscreteRNN(4, hidden_dim, 1, 'lstm')
    lstm = train_model(lstm, X_0, Y, epochs=20)
    
    print("Evaluating sequences...")
    cfc.eval()
    lstm.eval()
    
    # Take a 100-step slice for the plot
    start_idx = 500
    end_idx = 650
    
    X_60_slice = torch.FloatTensor(X_60).unsqueeze(0)[:, start_idx:end_idx, :]
    T_slice = torch.FloatTensor(T).unsqueeze(0)[:, start_idx:end_idx, :]
    Y_slice = Y[start_idx:end_idx, 0]
    
    with torch.no_grad():
        cfc_pred = cfc(X_60_slice, T_slice).squeeze().numpy()
        lstm_pred = lstm(X_60_slice).squeeze().numpy()
        
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
