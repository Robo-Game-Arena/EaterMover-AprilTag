import numpy as np
import math
from global_planner import Planner
from AprilTag_sensor import AprilTagDetector
from tag_ID import TagID as id

robot_AprilTag_width = 10 # cm

class Parameters:
    def __init__(self, size_px, bounds):
        ratio = size_px / robot_AprilTag_width
        
        self.bounds = bounds
        self.robot_radius = 5 # centimeters
        self.dt = 0.1 # sec
        self.v_mm = 348/2 # mm/s
        self.r = 6 # centimeters
        self.omega_min = -np.pi/2 # rad/sec
        self.omega_max =  np.pi/2 # rad/sec
        self.n_omega = 15
        self.prediction_horizon = 10
        self.pause = 0.001

        self.omega = None
        self.traj = None
        self.ratio = ratio

        # convert to pixels
        self.robot_radius *= ratio
        self.v = ratio * self.v_mm / 10
        self.r0 *= ratio

class LocalPlanner:
    def __init__(self, sensor:AprilTagDetector):
        self.sensor = sensor
        self.gpath = None
        self.reached_goal = False
        
        self.n_pos = 0
        self.sensor.points_traveled = np.array([])
        return

    def add_plan(self):
        self.parms = Parameters(self.sensor.curr_planner.robot_tag_width, 
                                [self.sensor.raw_bounds[0][0], self.sensor.raw_bounds[2][0],
                                 self.sensor.raw_bounds[0][1], self.sensor.raw_bounds[2][1]])
        self.gpath = self.sensor.raw_path # easier to use path without conversions 
        self.goal_idx = 1 # want to seek towards first path point
        self.z = [self.gpath[0][0], self.gpath[0][1], 
                  np.atan2(self.gpath[1][1] - self.gpath[0][1],
                           self.gpath[1][0] - self.gpath[0][0])]
        self.reached_goal = False
        
        self.n_pos = 0
        self.sensor.points_traveled = np.array([])
        
    
    def updatePosition(self):
        x_px, y_px, theta = None, None, None
        for tag in self.sensor.tags:
            if tag.tag_id == id.ROBOT_ID:
                x_px, y_px = tag.center
                R = tag.pose_R
                theta = np.atan2(-R[0][1], R[0][0]) - np.pi/2
        self.z = [x_px, y_px, theta]
        goal = self.gpath[self.goal_idx]
        if x_px is not None:
            self.n_pos += 1
            if self.n_pos % 3 == 0:
                self.sensor.points_traveled = np.reshape(np.append(self.sensor.points_traveled, [int(x_px), int(y_px)]), [-1, 2])
            # move goal until out of radius
            while True:
                if np.hypot(goal[0] - x_px, goal[1] - y_px) > self.parms.r0 + self.parms.robot_radius:
                    break
                
                if self.goal_idx >= len(self.gpath) - 1: # reached end of global path
                    print('navigation finished')
                    self.reached_goal = True
                    self.sensor.dwa_path = None
                    break
                else:
                    self.goal_idx += 1
                    goal = self.gpath[self.goal_idx]
                
            self.sensor.curr_goal = goal
        

    def replan(self):
        x, y, theta = self.z
        if x is None:
            self.parms.v_mm = 0
            return 0, 0
        self.parms.v_mm = 100
        goal = self.gpath[self.goal_idx]
        
        return self.dwa(x, y, theta, self.parms.v, goal, self.parms)


    def dwa(self, x0, y0, theta0, v, goal, parms: Parameters):
        omega_all = np.linspace(parms.omega_min, parms.omega_max, parms.n_omega)
        cost_all = np.zeros(parms.n_omega)
        trajectories = []
        for i, omega in enumerate(omega_all):
            z0 = [x0, y0, theta0]
            traj = [z0]
            for _ in range(parms.prediction_horizon):
                # use euler integration to get z0
                z0 = self.euler_integration([0, parms.dt], z0, [v, omega], parms)
                traj.append(z0)
            traj = np.array(traj)
            trajectories.append(traj)

            valid = True
            goalx, goaly = goal
            for x, y, _ in traj:

                #update costs, use euclidian distances to the goal
                cost_all[i] += ((x - goalx)**2 + (y - goaly)**2)**0.5 / self.parms.ratio 
                if not (parms.bounds[0] <= x <= parms.bounds[1] and parms.bounds[3] <= y <= parms.bounds[2]):
                    cost_all[i] += 1e6
                    valid = False
                    break
                for ob in self.sensor.obstacle_bounds: # ob = [xmin xmax ymin ymax]
                    xmin, xmax, ymin, ymax = ob
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        cost_all[i] += 1e6
                        valid = False
                        break
                if not valid:
                    break
            # add angle cost 
            a = self.goal_idx
            b = None
            if a == 0:
                b = 0
                a = 1
            else: 
                b = a - 1
            goaltheta = np.atan2(self.gpath[a, 1] - self.gpath[b, 1], self.gpath[a, 0] - self.gpath[b, 0])
            cost_all[i] += abs(goaltheta - traj[-1, 2]) * 2

        best_index = np.argmin(cost_all)

        dwa_path = np.array(trajectories[best_index])
        self.sensor.curr_goal = self.gpath[self.goal_idx]
        self.sensor.dwa_path = dwa_path[:, 0:2]

        return omega_all[best_index], trajectories[best_index]

    def euler_integration(self, tspan, z0, u, parms):
        v = min(u[0], parms.v)
        omega = u[1]
        h = tspan[1] - tspan[0]
        x0, y0, theta0 = z0
        x1 = x0 + v * math.cos(theta0) * h
        y1 = y0 + v * math.sin(theta0) * h
        theta1 = theta0 + omega * h
        return [x1, y1, theta1]
