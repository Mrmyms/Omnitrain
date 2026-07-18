import gym
import numpy as np
import yaml
from argparse import Namespace

with open('f1tenth_gym_repo/examples/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

env = gym.make('f110_gym:f110-v0', map="../data/maps/maps/vegas", map_ext='.png', num_agents=1)
env = gym.make('f110_gym:f110-v0', map="../data/maps/maps/vegas", map_ext='.png', num_agents=1)
obs, _, done, info = env.reset(np.array([[0.0, 0.0, 0.0]]))
obs, reward, done, info = env.step(np.array([[0.0, 0.0]]))
print("Vegas (0,0,0) valid? Done =", done)

env = gym.make('f110_gym:f110-v0', map="../data/maps/maps/berlin", map_ext='.png', num_agents=1)
obs, _, done, info = env.reset(np.array([[0.0, 0.0, 0.0]]))
obs, reward, done, info = env.step(np.array([[0.0, 0.0]]))
print("Berlin (0,0,0) valid? Done =", done)

env.close()
