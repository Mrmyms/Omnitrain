import sys
sys.path.append("../src")
from train_and_compare import DiscreteRNN
gru = DiscreteRNN(4, 16, 1, 'gru')
print("GRU params:", sum(p.numel() for p in gru.parameters()))
