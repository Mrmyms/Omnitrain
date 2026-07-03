import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    if not os.path.exists("data/results_mse.npy"):
        print("Error: results_mse.npy not found. Run train_and_compare.py first.")
        exit(1)
        
    results = np.load("data/results_mse.npy", allow_pickle=True).item()
    
    packet_loss_levels = [0, 20, 60]
    
    plt.figure(figsize=(8, 6))
    
    colors = {"LSTM": "red", "GRU": "orange", "CfC": "blue"}
    markers = {"LSTM": "x", "GRU": "^", "CfC": "o"}
    linestyles = {"LSTM": "--", "GRU": "-.", "CfC": "-"}
    
    for name, mses in results.items():
        plt.plot(packet_loss_levels, mses, 
                 label=f"{name} (TFLite vs Zero-Copy)" if name=="CfC" else name,
                 color=colors[name],
                 marker=markers[name],
                 linestyle=linestyles[name],
                 linewidth=2.5,
                 markersize=8)
                 
    plt.title("Temporal Resilience: MSE vs Packet Loss (Inverted Pendulum PiL)", fontsize=14, pad=15)
    plt.xlabel("Packet Loss / Temporal Jitter (%)", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.xticks(packet_loss_levels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("data/temporal_resilience_chart.png", dpi=300)
    print("Saved plot to data/temporal_resilience_chart.png")
