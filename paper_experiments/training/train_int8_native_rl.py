import gym
import numpy as np
import sys
import torch
import os
import copy
import multiprocessing as mp
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc_int8 import SparseCfCInt8
from topology_search_ncp import create_reflex_arc_mask

# Hyperparameters for RL
POPULATION_SIZE = 16 
GENERATIONS = 100
MUTATION_RATE = 0.05
NOISE_STD = 0.02
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
    
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    base_mask_w = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)

def mutate_model(model):
    """Mutates latent float weights. The Native INT8 forward pass handles strict quantization."""
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in mutated.named_parameters():
            if 'mask' in name: continue
            mask = (torch.rand_like(param) < MUTATION_RATE).float()
            noise = torch.randn_like(param) * NOISE_STD
            param.add_(mask * noise)
            
        # Re-enforce zero weights for non-header output paths
        n_sen_R = 50
        n_pro_R = 25
        mutated.fc.weight[:, :n_sen_R + n_pro_R] = 0.0
            
    return mutated

def evaluate_agent(model_state_dict):
    """Worker function to evaluate a single agent in the F1TENTH Gym using Native INT8 Math"""
    hidden_R = 100
    model = SparseCfCInt8(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask_w)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    env = gym.make('f110_gym:f110-v0', map=map_path_w, map_ext='.png', num_agents=1)
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    h_state = None
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    while not done:
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
        state = obs['linear_vels_x'][0]
        
        x = np.hstack([[state], downsampled_lidar])
        x_norm = (x - mean_X_w) / std_X_w
        
        x_tensor = torch.tensor(np.array([x_norm]), dtype=torch.float32).unsqueeze(0)
        dt_tensor = torch.tensor(np.array([[0.05]]), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            out, h_state = model(x_tensor, dt_tensor, h_init=h_state)
        
        action = out[0, -1]
        steer = action[0].item()
        speed = action[1].item()
        speed = max(1.5, min( speed, 5.0 ))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * 0.05
        steering_penalty += abs(steer) * 2.0
        
        if env.lap_counts[0] >= 1.0:
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
    print("--- STARTING NATIVE INT8 QAT EVOLUTION ---")
    
    hidden_R = 100
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    
    adam = SparseCfCInt8(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    if os.path.exists("../data/f110_reflex_qat_champion.pt"):
        print("Resuming evolution from previous Simulated QAT Champion to bootstrap Native INT8!")
        # We load strictly by name because SparseCfC and SparseCfCInt8 have the same parameter names
        state_dict = torch.load("../data/f110_reflex_qat_champion.pt", map_location='cpu')
        # Filter out anything not in adam
        adam_state = adam.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in adam_state}
        adam.load_state_dict(filtered, strict=False)
            
    population = [adam]
    for _ in range(POPULATION_SIZE - 1):
        population.append(mutate_model(adam))
        
    mp.set_start_method('spawn', force=True)
    num_processes = 4
    
    for gen in range(GENERATIONS):
        print(f"\n--- NATIVE INT8 Gen {gen+1}/{GENERATIONS} ---")
        start_t = time.time()
        
        state_dicts = [m.state_dict() for m in population]
        
        with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
            fitnesses = pool.map(evaluate_agent, state_dicts)
            
        scored_population = list(zip(fitnesses, population))
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = scored_population[0][0]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Top Native INT8 Fitness: {best_fitness:.1f} | Avg Fitness: {avg_fitness:.1f} | Time: {time.time() - start_t:.1f}s")
        
        survivors = [agent for fit, agent in scored_population[:4]]
        
        new_population = []
        for i in range(POPULATION_SIZE):
            parent = survivors[i % len(survivors)]
            if i == 0:
                new_population.append(copy.deepcopy(parent))
            else:
                new_population.append(mutate_model(parent))
                
        population = new_population
        torch.save(population[0].state_dict(), f"../data/f110_reflex_native_int8_champion.pt")

if __name__ == "__main__":
    run_evolution()
