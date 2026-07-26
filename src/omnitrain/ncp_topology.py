import torch

def create_ncp_mask_211(input_dim: int) -> torch.Tensor:
    """
    Creates the adjacency matrix for the 2-1-1 NCP topology.
    Total Hidden Neurons = 4.
    
    Sensory (0, 1) -> Inter (2) -> Command (3)
    
    Returns mask of shape (4, input_dim + 4)
    """
    hidden_dim = 4
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim)
    
    # ── 1. Sensory Neurons (indices 0, 1) ──
    # Connect to all LiDAR inputs
    mask[0:2, :input_dim] = 1.0
    
    # Sensory to Inter
    mask[2, input_dim + 0] = 1.0
    mask[2, input_dim + 1] = 1.0
    
    # ── 2. Inter Neuron (index 2) ──
    # Inter to Command
    mask[3, input_dim + 2] = 1.0
    
    # Recurrent connections (Inter to itself)
    mask[2, input_dim + 2] = 1.0
    
    # Optional: Command recurrent to itself for temporal smoothing
    mask[3, input_dim + 3] = 1.0
    
    return mask

def create_ncp_mask(input_dim: int, n_sensory: int, n_inter: int, n_command: int) -> torch.Tensor:
    """
    Generic NCP wiring mask generator.
    """
    hidden_dim = n_sensory + n_inter + n_command
    mask = torch.zeros(hidden_dim, input_dim + hidden_dim)
    
    idx_sensory = list(range(0, n_sensory))
    idx_inter = list(range(n_sensory, n_sensory + n_inter))
    idx_cmd = list(range(n_sensory + n_inter, hidden_dim))
    
    # 1. Sensory sees inputs
    for i in idx_sensory:
        mask[i, :input_dim] = 1.0
        
    # 2. Sensory connects to Inter
    for s in idx_sensory:
        for i in idx_inter:
            mask[i, input_dim + s] = 1.0
            
    # 3. Inter highly recurrent + connects to Command
    for i in idx_inter:
        for j in idx_inter:
            mask[j, input_dim + i] = 1.0
        for c in idx_cmd:
            mask[c, input_dim + i] = 1.0
            
    # 4. Command recurrent
    for c in idx_cmd:
        mask[c, input_dim + c] = 1.0
        
    return mask
