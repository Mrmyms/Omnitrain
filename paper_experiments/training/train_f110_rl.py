import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
import torch
import os
import copy
import multiprocessing as mp
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
# Add src to path so we can import omnitrain
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask

# Hyperparameters for RL
POPULATION_SIZE = 16 # Reduce to 16 for faster evaluation
GENERATIONS = 10000
MUTATION_RATE = 0.05
NOISE_STD = 0.02
MAX_STEPS = 10000 # Max steps per generation (Multiple laps)
d_in = 25
d_out = 2

# Global vars for workers
mean_X_w = None
std_X_w = None
conf_w = None
map_path_w = "../data/maps/maps/vegas"
base_mask_w = None

def init_worker():
    global mean_X_w, std_X_w, conf_w, base_mask_w
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    base_mask_w = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)

def mutate_model(model):
    """Mutates a given model by adding Gaussian noise to its weights"""
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for param in mutated.parameters():
            # Only mutate 5% of the parameters
            mask = (torch.rand_like(param) < MUTATION_RATE).float()
            noise = torch.randn_like(param) * NOISE_STD
            param.add_(mask * noise)
            
        # Re-enforce zero weights for non-header output paths
        n_sen_R = 50
        n_pro_R = 25
        mutated.fc.weight[:, :n_sen_R + n_pro_R] = 0.0
    return mutated

def evaluate_agent(model_state_dict):
    """Worker function to evaluate a single agent in the F1TENTH Gym"""
    # Rebuild model in this process
    hidden_R = 100
    model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask_w)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    env = gym.make('f110_gym:f110-v0', map=map_path_w, map_ext='.png', num_agents=1)
    
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    obs_window = []
    dt_window = []
    
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    # Run simulation until it crashes (Infinite limit)
    while not done:
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state = obs['linear_vels_x'][0]
        
        x = np.hstack([[state], downsampled_lidar])
        x_norm = (x - mean_X_w) / std_X_w
        
        obs_window.append(x_norm)
        dt_window.append([0.05])
        
        if len(obs_window) > 100:
            obs_window.pop(0)
            dt_window.pop(0)
            
        x_tensor = torch.tensor(np.array(obs_window), dtype=torch.float32).unsqueeze(0)
        dt_tensor = torch.tensor(np.array(dt_window), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            out = model(x_tensor, dt_tensor)
        
        action = out[0, -1]
        steer = action[0].item()
        speed = action[1].item()
        
        # We cap speed to prevent cheating (just spinning wheels or crawling)
        speed = max(1.5, min( speed, 5.0 ))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * 0.05
        
        if env.lap_counts[0] >= 1.0:
            # Successfully finished a lap!
            break
        
        if done:
            break
            
    env.close()
    
    # Fitness Function = Base distance - Steering Penalty + Massive Lap Completion Bonus + Speed Bonus
    laps = env.lap_counts[0]
    fitness = (distance_traveled * 10.0) - steering_penalty
    
    if laps >= 1.0:
        lap_time = env.lap_times[0]
        # Huge bonus for completing the lap, plus inversely proportional to lap_time
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
        
    return fitness

def run_evolution():
    print("--- STARTING EVOLUTIONARY STRATEGY RL PIPELINE (10,000 GENS) ---")
    
    # 1. Load the "Adam" model (resume from previous champion if exists)
    hidden_R = 100
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    
    adam = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    if os.path.exists("../data/f110_reflex_rl_champion.pt"):
        print("Resuming evolution from previous RL Champion!")
        adam.load_state_dict(torch.load("../data/f110_reflex_rl_champion.pt", map_location='cpu'))
    elif os.path.exists("../data/f110_reflex_100.pt"):
        print("Loaded Adam from pre-trained Imitation Reflex Arc model.")
        adam.load_state_dict(torch.load("../data/f110_reflex_100.pt", map_location='cpu'))
    else:
        print("Pre-trained model not found! Evolving from purely random weights.")
        
    population = [adam]
    # Initialize the rest of generation 0 with mutations of Adam
    for _ in range(POPULATION_SIZE - 1):
        population.append(mutate_model(adam))
        
    mp.set_start_method('spawn', force=True)
    num_processes = 4 # Use 4 workers to simulate 4 cars at once
    
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
        start_t = time.time()
        
        # Extract state dicts for safe multiprocessing
        state_dicts = [m.state_dict() for m in population]
        
        # Evaluate all cars in parallel
        with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
            fitnesses = pool.map(evaluate_agent, state_dicts)
            
        # Sort population by fitness
        scored_population = list(zip(fitnesses, population))
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = scored_population[0][0]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Top Fitness: {best_fitness:.1f} | Avg Fitness: {avg_fitness:.1f} | Time: {time.time() - start_t:.1f}s")
        
        # Selection: Keep the top 4 "Alpha" models
        survivors = [agent for fit, agent in scored_population[:4]]
        
        # Reproduction: Clone and mutate Alphas to fill the new population
        new_population = []
        for i in range(POPULATION_SIZE):
            # Select a random Alpha parent
            parent = survivors[i % len(survivors)]
            if i == 0:
                # Keep the absolute best unmutated (Elitism)
                new_population.append(copy.deepcopy(parent))
            else:
                new_population.append(mutate_model(parent))
                
        population = new_population
        
        # Save the champion of this generation
        torch.save(population[0].state_dict(), f"../data/f110_reflex_rl_champion.pt")
        
    print("\n🏁 EVOLUTION COMPLETE 🏁")
    print("Exporting RL Champion...")
    from omnitrain.esp32_exporter import ESP32Exporter
    exporter = ESP32Exporter(output_dir="data")
    exporter.export(population[0], input_dim=d_in, d_model=hidden_R, output_dim=d_out, filename="f110_reflex_rl_champion.omnibit")
    print("✅ RL Champion exported to data/f110_reflex_rl_champion.omnibit")

if __name__ == "__main__":
    run_evolution()
