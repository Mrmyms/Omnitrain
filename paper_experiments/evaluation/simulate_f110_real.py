import gym
import numpy as np
import yaml
import sys
import argparse
from argparse import Namespace
from numba import njit
import math
import os

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[:,:2] - point
    dists = np.empty((diffs.shape[0],))
    for i in range(dists.shape[0]):
        dists[i] = diffs[i,0]*diffs[i,0] + diffs[i,1]*diffs[i,1]
    min_dist_idx = np.argmin(dists)
    return min_dist_idx

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    start_i = nearest_point_on_trajectory(point, trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:2]
        end = trajectory[i+1,:2]
        V = end - start
        
        a = V[0]*V[0] + V[1]*V[1]
        b = 2.0 * (V[0]*(start[0]-point[0]) + V[1]*(start[1]-point[1]))
        c = (start[0]*start[0] + start[1]*start[1]) + (point[0]*point[0] + point[1]*point[1]) - 2.0*(start[0]*point[0] + start[1]*point[1]) - radius*radius
        discriminant = b*b - 4*a*c
        
        if discriminant >= 0:
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                lookahead_pt = np.zeros(3)
                lookahead_pt[:2] = start + t1 * V
                lookahead_pt[2] = trajectory[i, 2] # speed
                return lookahead_pt, i, t1
            elif t2 >= 0.0 and t2 <= 1.0:
                lookahead_pt = np.zeros(3)
                lookahead_pt[:2] = start + t2 * V
                lookahead_pt[2] = trajectory[i, 2]
                return lookahead_pt, i, t2
                
    # If no intersection found, return None equivalent
    return np.zeros(3), 0, 0.0

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        
    def load_waypoints(self, conf):
        wpts = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.waypoints = np.vstack((wpts[:, conf.wpt_xind], wpts[:, conf.wpt_yind], wpts[:, conf.wpt_vind])).T

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point, i, t = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, self.waypoints)
        if np.sum(np.abs(lookahead_point)) < 1e-6:
            return 0.0, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        return speed, steering_angle


def generate_dataset(samples=10000):
    map_path = "../data/maps/example_map"
    wpt_path = "f1tenth_gym_repo/examples/example_waypoints.csv"
    
    with open('f1tenth_gym_repo/examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    conf.wpt_path = wpt_path
    conf.wpt_delim = ';'
    conf.wpt_rowskip = 3
    conf.tlad = 0.8246
    conf.vgain = 1.0
    
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
    planner = PurePursuitPlanner(conf, 0.33)
    
    states = []
    actions = []
    lidar_scans = []
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    
    steps = 0
    while steps < samples:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], conf.tlad, conf.vgain)
        
        raw_lidar = obs['scans'][0]
        downsampled_lidar = raw_lidar[::len(raw_lidar)//24][:24] 
        
        states.append(np.array([obs['linear_vels_x'][0]]))
        lidar_scans.append(downsampled_lidar)
        actions.append(np.array([steer, speed]))
        
        # Robust DAgger-lite noise injection
        # 5% chance of a severe disruption to force recovery behaviors
        if np.random.rand() < 0.05:
            steer += np.random.normal(0, 0.4)
        
        # Continuous minor noise
        steer += np.random.normal(0, 0.05)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        steps += 1
        
        if done:
            obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            
        if steps % 1000 == 0:
            print(f"Generated {steps}/{samples} samples...")
            
    return np.array(states), np.array(lidar_scans), np.array(actions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    args = parser.parse_args()
    
    print("Generating REAL F1TENTH imitation learning dataset...")
    states, lidar, actions = generate_dataset(args.samples)
    
    os.makedirs("data", exist_ok=True)
    np.savez("../data/f110_real_dataset.npz", 
             states=states, 
             lidar=lidar, 
             actions=actions, 
             dt=np.full((args.samples, 1), 0.05))
             
    print(f"Dataset generated! LiDAR shape: {lidar.shape}, Actions shape: {actions.shape}")
