import torch, torch.nn as nn, numpy as np
import train_and_compare as tc

def test():
    tc.EPOCHS = 200
    np.random.seed(42); torch.manual_seed(42)
    X, Y, T = np.load("../data/pendulum_X_0loss.npy"), np.load("../data/pendulum_Y.npy"), np.load("../data/pendulum_T.npy")
    
    print("Training LSTM 200 epochs...")
    lstm = tc.LSTMBaseline(4, 16, 1)
    tc.train_model(lstm, X, Y, epochs=200)
    print("LSTM 0% TTF:", tc.evaluate_closed_loop(lstm, 0.0))
    print("LSTM 60% TTF:", tc.evaluate_closed_loop(lstm, 0.60))

    print("Training CfCFull 200 epochs...")
    cfc = tc.ContinuousCfCFull(4, 16, 1)
    tc.train_model(cfc, X, Y, T=T, epochs=200)
    print("CfC 0% TTF:", tc.evaluate_closed_loop(cfc, 0.0))
    print("CfC 60% TTF:", tc.evaluate_closed_loop(cfc, 0.60))

if __name__ == "__main__":
    test()
