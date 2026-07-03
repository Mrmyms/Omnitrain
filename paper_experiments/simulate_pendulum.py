import numpy as np
import os
import argparse

def generate_cartpole_data(num_samples=10000, dt=0.02):
    """
    Generates synthetic Inverted Pendulum (CartPole) data.
    States: [cart position, cart velocity, pole angle, pole angular velocity]
    Action: Force applied to cart.
    """
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 
    polemass_length = (masspole * length)
    
    # Init state
    state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    
    X = []
    Y = []
    T = []
    
    current_time = 0.0
    for _ in range(num_samples):
        x, x_dot, theta, theta_dot = state
        
        # Simple PD controller for generating "expert" targets
        force = 50.0 * theta + 10.0 * theta_dot + 5.0 * x + 2.0 * x_dot
        force = np.clip(force, -10.0, 10.0)
        
        X.append(state.copy())
        Y.append([force])
        T.append([current_time])
        
        # Euler integration
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        
        x  = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc
        
        state = np.array([x, x_dot, theta, theta_dot])
        current_time += dt
        
    return np.array(X), np.array(Y), np.array(T)

def inject_packet_loss(X, Y, T, loss_prob=0.2):
    """
    Simulates jitter/packet loss by holding the previous state when a packet is dropped.
    """
    mask = np.random.rand(len(X)) > loss_prob
    X_masked = X.copy()
    
    for i in range(1, len(X)):
        if not mask[i]:
            X_masked[i] = X_masked[i-1] # Hold previous state
    
    return X_masked, Y, T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    
    X, Y, T = generate_cartpole_data(args.samples)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/pendulum_X_0loss.npy", X)
    
    X_20, _, _ = inject_packet_loss(X, Y, T, 0.20)
    np.save("data/pendulum_X_20loss.npy", X_20)
    
    X_60, _, _ = inject_packet_loss(X, Y, T, 0.60)
    np.save("data/pendulum_X_60loss.npy", X_60)
    
    np.save("data/pendulum_Y.npy", Y)
    np.save("data/pendulum_T.npy", T)
    
    print(f"Generated {args.samples} samples of Inverted Pendulum PiL data.")
