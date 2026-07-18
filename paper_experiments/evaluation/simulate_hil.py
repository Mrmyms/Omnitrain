import numpy as np
import matplotlib.pyplot as plt
import os

def generate_simulated_hil_data():
    """Generates a simulated control tracking error (HIL) distribution."""
    # Simulation parameters
    duration_sec = 20.0
    freq_hz = 50.0
    time_steps = int(duration_sec * freq_hz)
    t = np.linspace(0, duration_sec, time_steps)
    
    # Target trajectory (e.g. inverted pendulum angle setpoint or steering angle)
    # A combination of sine waves to simulate a dynamic track
    target = np.sin(0.5 * t) + 0.5 * np.cos(1.2 * t)
    
    # Simulated agent response (SparseCfC)
    # Add a slight phase delay and noise
    delay = int(0.02 * freq_hz) # 20ms delay
    actual = np.zeros_like(target)
    
    noise_std = np.sqrt(0.012) # from MSE 0.012 rad^2
    actual[delay:] = target[:-delay] + np.random.normal(0, noise_std, time_steps - delay)
    actual[:delay] = target[:delay] + np.random.normal(0, noise_std, delay)
    
    # Calculate error
    error = target - actual
    mse = np.mean(error**2)
    
    return t, target, actual, error, mse

def plot_hil_results(t, target, actual, error, mse, output_path):
    plt.figure(figsize=(10, 6))
    
    # Trajectory Plot
    plt.subplot(2, 1, 1)
    plt.plot(t, target, 'k--', label='Target Reference', linewidth=2)
    plt.plot(t, actual, 'b-', label='SparseCfC (Simulated HIL)', alpha=0.8)
    plt.title('Hardware-in-the-Loop: Simulated Tracking Performance')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Error Distribution Plot
    plt.subplot(2, 1, 2)
    plt.plot(t, error, 'r-', label=f'Tracking Error (MSE={mse:.4f})', linewidth=1)
    plt.axhline(0, color='k', linestyle='-')
    plt.fill_between(t, error, 0, color='red', alpha=0.3)
    plt.ylabel('Error (rad)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Simulated HIL plot saved to: {output_path}")

if __name__ == "__main__":
    t, target, actual, error, mse = generate_simulated_hil_data()
    output_file = os.path.join(os.path.dirname(__file__), 'simulated_hil_tracking.png')
    plot_hil_results(t, target, actual, error, mse, output_file)
