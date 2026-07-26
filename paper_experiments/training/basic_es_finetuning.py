import torch
import torch.nn as nn
import numpy as np
import gym
import concurrent.futures
import time
import os
import sys

sys.path.append(os.path.abspath('../../src'))
from omnitrain.sparse_cfc import SparseCfC
from omnitrain.esp32_exporter import ESP32Exporter
from export_rnn_omnibit import DiscreteRNN
from train_f110_ncp import create_advanced_layered_mask

# ES Hyperparameters
POPULATION_SIZE = 16  # Must be even for antithetic sampling
SIGMA = 0.05
LEARNING_RATE = 0.01
GENERATIONS = 300

def get_flat_weights(model):
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])

def set_flat_weights(model, flat_weights):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_weights[idx:idx+numel].view_as(p))
            idx += numel

def evaluate_weights(args):
    # args: (model_name, hidden_dim, is_sparse, rnn_type, flat_weights)
    model_name, hidden_dim, is_sparse, rnn_type, flat_weights = args
    
    # Re-instantiate model in worker to avoid serialization issues
    d_in = 25
    d_out = 2
    
    if 'CfC' in model_name:
        if 'Sparse' in model_name:
            adj = create_advanced_layered_mask(d_in, 50, 25, 25, density=0.25)
            model = SparseCfC(d_in, hidden_dim, d_out, adj)
        else:
            adj = torch.ones(hidden_dim, d_in + hidden_dim)
            model = SparseCfC(d_in, hidden_dim, d_out, adj)
    else:
        model = DiscreteRNN(d_in, hidden_dim, d_out, rnn_type=rnn_type)
        if is_sparse:
            model.apply_sparsity(0.75)
            
    set_flat_weights(model, flat_weights)
    model.eval()
    
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    distance = 0.0
    steps = 0
    prev_time = torch.zeros(1, 1, 1)
    action = np.array([[0.0, 0.0]])
    
    with torch.no_grad():
        while not done and steps < 1000:  # Cap at 1000 steps to prevent infinite loops
            raw_lidar = obs['scans'][0]
            downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24]
            state = obs['linear_vels_x'][0]
            
            x = np.hstack([[state], downsampled_lidar])
            if len(x) == len(mean_X_w) + 1:
                mean = np.append(mean_X_w, 0.0)
                std = np.append(std_X_w, 1.0)
                x_norm = (x - mean) / np.maximum(std, 1e-8)
            else:
                x_norm = (x - mean_X_w[:25]) / np.maximum(std_X_w[:25], 1e-8)
                
            # ensure x_norm is exactly 25
            x_norm = x_norm[:25]
            x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            curr_time = prev_time + 0.05
            if 'CfC' in model_name:
                times = torch.cat([prev_time, curr_time], dim=1)
                x_t_seq = x_t.expand(-1, 2, -1)
                a = model(x_t_seq, times)[:, -1, :].squeeze(0).numpy()
            else:
                a = model(x_t).squeeze(0).squeeze(0).numpy()
                
            steer = float(a[0])
            speed = float(a[1])
            speed = max(1.5, min(speed, 5.0))
            action = np.array([[steer, speed]])
            
            obs, reward, done, info = env.step(action)
            distance += speed * 0.05
            steps += 1
            prev_time = curr_time
            
            if env.lap_counts[0] >= 1.0:
                break
                
    env.close()
    
    # Fitness is simply the distance driven
    return distance

def main():
    baselines_dir = "data/paper_baselines"
    os.makedirs(f"{baselines_dir}/es_finetuned", exist_ok=True)
    
    models = [
        ("CfC_Sparse", 100, True, None),
        ("LSTM_Sparse", 45, True, 'lstm'),
        ("LSTM_Dense", 22, False, 'lstm'),
        ("GRU_Dense", 25, False, 'gru'),
        ("CfC_Dense", 25, False, None),
        ("GRU_Sparse", 50, True, 'gru')
    ]
    
    d_in = 25
    d_out = 2
    exporter = ESP32Exporter(output_dir=f"{baselines_dir}/es_finetuned")
    
    print(f"Starting 300 epochs of Basic ES fine-tuning on {len(models)} models in parallel...")
    
    for name, hidden_dim, is_sparse, rnn_type in models:
        out_pt = f"{baselines_dir}/es_finetuned/{name}.pt"
        if os.path.exists(out_pt):
            print(f"Skipping {name}, already finetuned.")
            continue
            
        print(f"\n{'='*50}\nEvolving {name} for 300 Generations\n{'='*50}")
        
        # Instantiate and load pre-trained weights
        if 'CfC' in name:
            if is_sparse:
                adj = create_advanced_layered_mask(d_in, 50, 25, 25, density=0.25)
                model = SparseCfC(d_in, hidden_dim, d_out, adj)
            else:
                adj = torch.ones(hidden_dim, d_in + hidden_dim)
                model = SparseCfC(d_in, hidden_dim, d_out, adj)
        else:
            model = DiscreteRNN(d_in, hidden_dim, d_out, rnn_type=rnn_type)
            if is_sparse:
                model.apply_sparsity(0.75)
                
        pt_path = f"{baselines_dir}/{name}.pt"
        if os.path.exists(pt_path):
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        else:
            print(f"Skipping {name}, base checkpoint not found.")
            continue
            
        w = get_flat_weights(model)
        n_params = len(w)
        
        # Evaluate baseline performance
        baseline_score = evaluate_weights((name, hidden_dim, is_sparse, rnn_type, w))
        print(f"Baseline F1TENTH Fitness (Distance): {baseline_score:.2f}m")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=POPULATION_SIZE) as executor:
            for gen in range(GENERATIONS):
                start_t = time.time()
                
                # Generate antithetic noise
                noise = torch.randn(POPULATION_SIZE // 2, n_params)
                noise = torch.cat([noise, -noise], dim=0) # Shape: (POPULATION_SIZE, n_params)
                
                jobs = []
                for i in range(POPULATION_SIZE):
                    w_mutated = w + SIGMA * noise[i]
                    jobs.append((name, hidden_dim, is_sparse, rnn_type, w_mutated))
                    
                fitnesses = list(executor.map(evaluate_weights, jobs))
                fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
                
                # Standardize fitnesses
                fit_mean = fitnesses.mean()
                fit_std = fitnesses.std() + 1e-8
                normalized_fitnesses = (fitnesses - fit_mean) / fit_std
                
                # Weight update
                w = w + LEARNING_RATE / (POPULATION_SIZE * SIGMA) * torch.matmul(noise.t(), normalized_fitnesses)
                
                elapsed = time.time() - start_t
                if (gen + 1) % 10 == 0 or gen == 0:
                    print(f"Gen {gen+1:03d} | Mean Fitness: {fit_mean:.2f}m | Max: {fitnesses.max():.2f}m | Time: {elapsed:.1f}s")
                    
        # Save and export fine-tuned model
        set_flat_weights(model, w)
        out_pt = f"{baselines_dir}/es_finetuned/{name}.pt"
        torch.save(model.state_dict(), out_pt)
        exporter.export(model, input_dim=d_in, d_model=hidden_dim, output_dim=d_out, filename=f"{name}.omnibit")
        print(f"Finished {name}. Exported to {out_pt} and .omnibit")

if __name__ == "__main__":
    main()
