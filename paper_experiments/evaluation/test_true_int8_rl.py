import gym
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

sys.path.append(os.path.abspath('../f1tenth_gym_repo/examples'))
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../training'))
from omnitrain.sparse_cfc import SparseCfC
from topology_search_ncp import create_reflex_arc_mask

def quantize_tensor_to_int8(tensor):
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0
    scale = max_val / 127.0
    q_tensor = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
    return q_tensor, scale

class TrueInt8SparseCfC(nn.Module):
    def __init__(self, float_model):
        super().__init__()
        self.input_dim = float_model.input_dim
        self.hidden_dim = float_model.hidden_dim
        self.output_dim = float_model.output_dim
        self._hsize = float_model._hsize
        
        # We pre-quantize the weights into INT8
        masked_weight = float_model.backbone_weight * float_model.mask
        self.q_bb_weight, self.bb_w_scale = quantize_tensor_to_int8(masked_weight)
        
        self.q_f_weight, self.f_w_scale = quantize_tensor_to_int8(float_model.f_weight)
        self.q_g_weight, self.g_w_scale = quantize_tensor_to_int8(float_model.g_weight)
        self.q_h_weight, self.h_w_scale = quantize_tensor_to_int8(float_model.h_weight)
        self.q_fc_weight, self.fc_w_scale = quantize_tensor_to_int8(float_model.fc.weight)
        
        # Biases are kept in float for accumulation step (standard practice)
        self.bb_bias = float_model.backbone_bias.clone()
        self.f_bias = float_model.f_bias.clone()
        self.g_bias = float_model.g_bias.clone()
        self.h_bias = float_model.h_bias.clone()
        self.fc_bias = float_model.fc.bias.clone()

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self._hsize, device=x.device)
        out = []
        
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            
            x_in = torch.cat([x[:, t, :], h], dim=-1)
            
            # Dynamic Quantization of the input activation
            q_x_in, x_in_scale = quantize_tensor_to_int8(x_in)
            
            # TRUE INT8 MATMUL -> INT32 ACCUMULATOR
            # We cast to int32 before multiplication to prevent overflow, simulating an int8 MAC unit
            acc_bb = torch.matmul(q_x_in.to(torch.int32), self.q_bb_weight.t().to(torch.int32))
            
            # Dequantize back to float
            bb = acc_bb.to(torch.float32) * (x_in_scale * self.bb_w_scale) + self.bb_bias
            bb = torch.tanh(bb)
            
            # Element-wise operations in INT8
            q_bb, bb_scale = quantize_tensor_to_int8(bb)
            
            acc_f = (q_bb.to(torch.int32) * self.q_f_weight.to(torch.int32))
            f_val = acc_f.to(torch.float32) * (bb_scale * self.f_w_scale) + self.f_bias
            
            acc_g = (q_bb.to(torch.int32) * self.q_g_weight.to(torch.int32))
            g_val = acc_g.to(torch.float32) * (bb_scale * self.g_w_scale) + self.g_bias
            
            acc_h = (q_bb.to(torch.int32) * self.q_h_weight.to(torch.int32))
            h_val = acc_h.to(torch.float32) * (bb_scale * self.h_w_scale) + self.h_bias
            
            # Activations in float (mimicking a TPU LUT conversion step)
            t_gate = torch.sigmoid(-f_val * dt)
            h = t_gate * torch.tanh(g_val) + (1.0 - t_gate) * torch.tanh(h_val)
            out.append(h.unsqueeze(1))
            
        h_seq = torch.cat(out, dim=1)
        
        # Final FC layer in INT8
        q_h_seq, h_seq_scale = quantize_tensor_to_int8(h_seq)
        acc_fc = torch.matmul(q_h_seq.to(torch.int32), self.q_fc_weight.t().to(torch.int32))
        fc_out = acc_fc.to(torch.float32) * (h_seq_scale * self.fc_w_scale) + self.fc_bias
        
        return fc_out

def main():
    d_in = 25
    d_out = 2
    hidden_R = 100
    n_sen_R = 50
    n_pro_R = 25
    n_hdr_R = 25
    
    # Load Stats
    stats = np.load("../data/f110_real_stats.npz")
    mean_X_w = stats["mean"]
    std_X_w = stats["std"]
    
    # Create Model
    base_mask = create_reflex_arc_mask(d_in, n_sen_R, n_pro_R, n_hdr_R, density=0.25)
    float_model = SparseCfC(input_dim=d_in, hidden_dim=hidden_R, output_dim=d_out, adjacency_matrix=base_mask)
    
    model_path = "../data/f110_reflex_qat_champion.pt"
    float_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Convert to True INT8 Execution Engine
    print("Convertiendo modelo a True INT8 (Aritmetica entera pura)...")
    int8_model = TrueInt8SparseCfC(float_model)
    int8_model.eval()
    print("✅ Motor INT8 Listo!")
    
    # Run Simulation
    map_path = "../data/maps/maps/vegas"
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    
    obs_window = []
    dt_window = []
    
    steps = 0
    distance_traveled = 0.0
    steering_penalty = 0.0
    
    print("🏁 Iniciando Evaluacion en INT8 Real...")
    start_time = time.time()
    
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
            out = int8_model(x_tensor, dt_tensor)
        
        action = out[0, -1]
        steer = action[0].item()
        speed = max(1.5, min(action[1].item(), 5.0))
        
        obs, reward, done, info = env.step(np.array([[steer, speed]]))
        
        steps += 1
        distance_traveled += speed * 0.05
        steering_penalty += abs(steer) * 2.0
        
        if steps % 1000 == 0:
            print(f"Step {steps}: Distancia = {distance_traveled:.1f}m, Vueltas = {env.lap_counts[0]:.2f}")
            
        if env.lap_counts[0] >= 1.0:
            print("🎉 VUELTA COMPLETADA EN INT8 REAL!")
            break
            
        if done:
            print("💥 CHOCÓ!")
            break
            
    env.close()
    
    # Calculate Fitness
    fitness = (distance_traveled * 10.0) - steering_penalty
    if env.lap_counts[0] >= 1.0:
        lap_time = env.lap_times[0]
        fitness += 10000.0
        fitness += (10000.0 / max(1.0, lap_time))
    
    elapsed = time.time() - start_time
    print("\n--- RESULTADOS INT8 PURO ---")
    print(f"Pasos: {steps}")
    print(f"Distancia: {distance_traveled:.1f} metros")
    print(f"Puntos Fitness: {fitness:.1f}")
    print(f"Tiempo de Computo: {elapsed:.1f} segundos")

if __name__ == "__main__":
    main()
