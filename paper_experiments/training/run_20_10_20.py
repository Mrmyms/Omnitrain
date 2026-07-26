import sys
import os
import torch
from architecture_search_ncp import init_worker, evaluate_architecture_worker

if __name__ == '__main__':
    # Initialize data
    init_worker()
    
    # Run evaluation
    config = {
        'n_sensory': 20,
        'n_process': 10,
        'n_header': 20,
        'density': 0.5
    }
    evaluate_architecture_worker(config, 'cpu', 'temp_results.csv')
