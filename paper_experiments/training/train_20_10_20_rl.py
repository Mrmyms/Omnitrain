import gym
import numpy as np
import yaml
import sys
import torch
import os
import copy
import multiprocessing as mp
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC

# Hyperparameters for RL
POPULATION_SIZE = 16 
GENERATIONS = 130
MUTATION_RATE = 0.05
NOISE_STD = 0.02
MAX_STEPS = 10000 
d_in = 25
d_out = 2

# Global vars for workers
mean_X_w = None
std_X_w = None
map_path_w = "../data/maps/maps/vegas"
base_mask_w = None

def init_worker():
    global mean_X_w, std_X_w, base_mask_w
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    n_sen_R = 20
    n_pro_R = 10
    n_hdr_R = 20
    
    # We load the exact mask from the BC model to ensure exact topology match
    bc_model = SparseCfC(input_dim=d_in, hidden_dim=50, output_dim=d_out, adjacency_matrix=torch.ones(50, 75))
    bc_model.load_state_dict(torch.load("../data/f110_20_10_20_bc.pt", map_location='cpu'))
    base_mask_w = bc_model.mask.clone().detach()

def mutate_model(model):
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for param in mutated.parameters():
            mask = (torch.rand_like(param) < MUTATION_RATE).float()
            noise = torch.randn_like(param) * NOISE_STD
            param.add_(mask * noise)
            
        mutated.fc.weight[:, :30] = 0.0
    return mutated

def evaluate_agent(model_state_dict):
    hidden_R = 50
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
        speed = max(1.5, min( action[1].item(), 5.0 ))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * 0.05
        
        if env.lap_counts[0] >= 1.0:
            break
        if done:
            break
            
    env.close()
    
    laps = env.lap_counts[0]
    fitness = (distance_traveled * 10.0) - steering_penalty
    
    if laps >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
        
    return fitness

def run_evolution():
    print("--- STARTING EVOLUTIONARY STRATEGY RL PIPELINE (10,000 GENS) ---")
    print("Architecture: NCP 20-10-20")
    
    hidden_R = 50
    # Create dummy mask, it will be overridden by load_state_dict
    adam = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=torch.ones(50, 75))
    
    if os.path.exists("../data/f110_20_10_20_rl_champion.pt"):
        print("Resuming evolution from previous RL Champion!")
        adam.load_state_dict(torch.load("../data/f110_20_10_20_rl_champion.pt", map_location='cpu'))
    elif os.path.exists("../data/f110_20_10_20_bc.pt"):
        print("Loaded Adam from pre-trained Imitation (BC) model.")
        adam.load_state_dict(torch.load("../data/f110_20_10_20_bc.pt", map_location='cpu'))
    else:
        print("Pre-trained model not found! Evolving from purely random weights.")
        
    population = [adam]
    for _ in range(POPULATION_SIZE - 1):
        population.append(mutate_model(adam))
        
    mp.set_start_method('spawn', force=True)
    num_processes = 4 
    
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
        start_t = time.time()
        
        state_dicts = [m.state_dict() for m in population]
        
        with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
            fitnesses = pool.map(evaluate_agent, state_dicts)
            
        scored_population = list(zip(fitnesses, population))
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = scored_population[0][0]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Top Fitness: {best_fitness:.1f} | Avg Fitness: {avg_fitness:.1f} | Time: {time.time() - start_t:.1f}s")
        
        survivors = [agent for fit, agent in scored_population[:4]]
        
        new_population = []
        for i in range(POPULATION_SIZE):
            parent = survivors[i % len(survivors)]
            if i == 0:
                new_population.append(copy.deepcopy(parent))
            else:
                new_population.append(mutate_model(parent))
                
        population = new_population
        torch.save(population[0].state_dict(), f"../data/f110_20_10_20_rl_champion.pt")
        
    print("\n🏁 EVOLUTION COMPLETE 🏁")

if __name__ == "__main__":
    run_evolution()
