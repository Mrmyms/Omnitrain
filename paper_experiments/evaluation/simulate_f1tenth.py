import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

class Track:
    def __init__(self, radius=10.0, track_width=3.0):
        self.track_width = track_width
        self.radius = radius
        self.straight_len = 15.0
        self.center_line = self._generate_center_line()
        self.inner_wall, self.outer_wall = self._generate_walls()
        
    def _generate_center_line(self):
        pts = []
        for x in np.linspace(-self.straight_len/2, self.straight_len/2, 50):
            pts.append([x, -self.radius])
        for t in np.linspace(-np.pi/2, np.pi/2, 50):
            pts.append([self.straight_len/2 + self.radius*np.cos(t), self.radius*np.sin(t)])
        for x in np.linspace(self.straight_len/2, -self.straight_len/2, 50):
            pts.append([x, self.radius])
        for t in np.linspace(np.pi/2, 3*np.pi/2, 50):
            pts.append([-self.straight_len/2 + self.radius*np.cos(t), self.radius*np.sin(t)])
        return np.array(pts)
        
    def _generate_walls(self):
        inner = []
        outer = []
        c = self.center_line
        for i in range(len(c)):
            p1 = c[i]
            p2 = c[(i+1)%len(c)]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            n_x, n_y = -dy, dx
            norm = math.hypot(n_x, n_y)
            if norm == 0: continue
            n_x, n_y = n_x/norm, n_y/norm
            
            inner.append([p1[0] - n_x*self.track_width/2, p1[1] - n_y*self.track_width/2])
            outer.append([p1[0] + n_x*self.track_width/2, p1[1] + n_y*self.track_width/2])
        return np.array(inner), np.array(outer)
        
    def check_collision(self, x, y):
        dists = np.linalg.norm(self.center_line - np.array([x, y]), axis=1)
        if np.min(dists) > self.track_width/2 - 0.2:
            return True
        return False

class KinematicBicycle:
    def __init__(self, L=0.33):
        self.L = L 
        self.x = 0.0
        self.y = -10.0 
        self.theta = 0.0
        self.v = 0.0
        
    def reset(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        
    def step(self, delta, a, dt):
        max_steer = 0.4
        delta = np.clip(delta, -max_steer, max_steer)
        a = np.clip(a, -5.0, 5.0)
        
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += (self.v / self.L) * math.tan(delta) * dt
        self.v += a * dt
        self.v = np.clip(self.v, 0.0, 8.0) 

def raycast(x, y, theta, track, num_rays=21, fov=np.pi*1.5, max_range=10.0):
    angles = np.linspace(-fov/2, fov/2, num_rays) + theta
    rays = np.full(num_rays, max_range)
    
    walls = [track.inner_wall, track.outer_wall]
    
    for w in walls:
        for i in range(len(w)):
            p1 = w[i]
            p2 = w[(i+1)%len(w)]
            v1 = p2[0]-p1[0], p2[1]-p1[1]
            
            for j, angle in enumerate(angles):
                v2 = math.cos(angle), math.sin(angle)
                det = v1[0]*v2[1] - v1[1]*v2[0]
                if abs(det) < 1e-6: continue
                
                dx = x - p1[0]
                dy = y - p1[1]
                t1 = (dx*v2[1] - dy*v2[0]) / det
                t2 = (dx*v1[1] - dy*v1[0]) / det
                
                if 0 <= t1 <= 1 and t2 > 0:
                    if t2 < rays[j]:
                        rays[j] = t2
    return rays

def pure_pursuit(car, track, lookahead=2.0):
    dists = np.linalg.norm(track.center_line - np.array([car.x, car.y]), axis=1)
    nearest_idx = np.argmin(dists)
    
    target_idx = nearest_idx
    for i in range(1, len(track.center_line)):
        idx = (nearest_idx + i) % len(track.center_line)
        d = np.linalg.norm(track.center_line[idx] - np.array([car.x, car.y]))
        if d >= lookahead:
            target_idx = idx
            break
            
    target = track.center_line[target_idx]
    alpha = math.atan2(target[1]-car.y, target[0]-car.x) - car.theta
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    delta = math.atan2(2.0 * car.L * math.sin(alpha), lookahead)
    
    target_v = 6.0 if abs(delta) < 0.1 else 3.0
    a = (target_v - car.v) * 2.0
    return delta, a

def generate_f1tenth_dataset(samples=50000, dt=0.05):
    track = Track()
    car = KinematicBicycle()
    
    states = []
    actions = []
    lidar_scans = []
    
    for step in range(samples):
        # Pseudo-DAgger: Randomly perturb the car's state so it learns to recover
        if np.random.rand() < 0.05:  # 5% chance per step to get bumped
            car.theta += np.random.normal(0, 0.2)
            # Add lateral drift (move car perpendicular to its heading)
            drift = np.random.normal(0, 0.5)
            car.x += -math.sin(car.theta) * drift
            car.y += math.cos(car.theta) * drift
            
        delta, a = pure_pursuit(car, track)
        scan = raycast(car.x, car.y, car.theta, track)
        
        states.append(np.array([car.v]))
        lidar_scans.append(scan)
        actions.append(np.array([delta, a]))
        
        car.step(delta, a, dt)
        if track.check_collision(car.x, car.y):
            # print(f"Collision at step {step}! Resetting.")
            # Instead of a fixed reset, reset to a safe point slightly ahead
            car.reset(car.x, car.y, car.theta) # Wait, if it collided it will immediately collide again if we just leave it.
            # Find nearest track center point and reset there
            dists = np.linalg.norm(track.center_line - np.array([car.x, car.y]), axis=1)
            safe_idx = (np.argmin(dists) + 5) % len(track.center_line)
            safe_pt = track.center_line[safe_idx]
            
            # Compute heading matching the track direction
            next_idx = (safe_idx + 1) % len(track.center_line)
            next_pt = track.center_line[next_idx]
            safe_theta = math.atan2(next_pt[1]-safe_pt[1], next_pt[0]-safe_pt[0])
            
            car.reset(safe_pt[0], safe_pt[1], safe_theta)
            
    return np.array(states), np.array(lidar_scans), np.array(actions), track

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    print("Generating F1TENTH imitation learning dataset...")
    states, lidar, actions, track = generate_f1tenth_dataset(args.samples, args.dt)
    
    os.makedirs("data", exist_ok=True)
    np.savez("../data/f1tenth_dataset.npz", 
             states=states, 
             lidar=lidar, 
             actions=actions, 
             dt=np.full((args.samples, 1), args.dt))
             
    print(f"Dataset generated! LiDAR shape: {lidar.shape}, Actions shape: {actions.shape}")
    
    if args.visualize:
        plt.figure(figsize=(10,6))
        plt.plot(track.center_line[:,0], track.center_line[:,1], 'k--')
        plt.plot(track.inner_wall[:,0], track.inner_wall[:,1], 'r')
        plt.plot(track.outer_wall[:,0], track.outer_wall[:,1], 'r')
        plt.title("F1TENTH Synthethic Track")
        plt.axis('equal')
        plt.savefig("../data/f1tenth_track.png")
        print("Track visualization saved to data/f1tenth_track.png")
