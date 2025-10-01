from safety_gym.envs.engine import Engine
from gym.envs.registration import register
from scipy.spatial.transform import Rotation
import numpy as np
import safety_gym
import pickle
import gym
import cv2
import re

class GymEnv():
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        _, self.robot_name, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16 * 2 + 1
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = self._env.action_space
        self.goal_threshold = np.inf
        self.hazard_size = 0.2
        self.vase_size = 0.1
        self.gremlin_size = 0.1*np.sqrt(2.0)
        self.button_size = 0.1
        self.safety_confidence = 0.0

    def _get_original_state(self):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = np.array([self._env.dist_goal()])
        goal_dist = np.clip(goal_dist, 0.0, self.goal_threshold)
        acc = self._env.world.get_sensor('accelerometer')[:2]
        vel = self._env.world.get_sensor('velocimeter')[:2]
        rot_vel = self._env.world.get_sensor('gyro')[2:]
        hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
        vases_lidar = self._env.obs_lidar(self._env.vases_pos, 3)
        lidar = np.concatenate([hazards_lidar, vases_lidar])
        vase_dist = [self._get_vase_dist()]

        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (lidar - 0.3)/0.3, vase_dist], axis=0)
        return state

    def _get_vase_penalty(self):
        v_dist = self._get_vase_dist()
        v_coeff = 10.0
        safety_threshold = 0.3
        penalty = 1.0/(1.0 + np.exp((v_dist)*v_coeff)) if v_dist < safety_threshold else 0.0
        return -1 * penalty

    def _get_cost(self, h_dist):
        h_coeff = 10.0
        cost = 1.0/(1.0 + np.exp((h_dist - self.safety_confidence)*h_coeff))
        return cost

    def _get_min_dist(self, hazard_pos_list, pos):
        pos = np.array(pos)
        min_dist = np.inf
        for hazard_pos in hazard_pos_list:
            dist = np.linalg.norm(hazard_pos[:2] - pos[:2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_hazard_dist(self):
        # hazard
        h_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
        # # vase
        # v_dist = self._get_min_dist(self._env.vases_pos, self._env.world.robot_pos()) - self.vase_size
        # return min(h_dist, v_dist)
        return h_dist

    def _get_vase_dist(self):
        # vaselist = []
        v_dist = self._get_min_dist(self._env.vases_pos, self._env.world.robot_pos()) - self.vase_size
        return v_dist

    def get_step_wise_cost(self):
        h_dist = self._get_hazard_dist()
        step_wise_cost =  self.safety_confidence - h_dist
        return step_wise_cost
        
    def reset(self):
        self.t = 0
        # seed = np.random.default_rng().integers(0, 100)
        # print("Env Seed is ", seed)
        # self._env.seed(seed)
        self._env.reset()
        state = self._get_original_state()
        return state

    def step(self, state, action):
        reward = 0
        goal_reward = 0
        is_goal_met = 0
        num_cv = 0

        for _ in range(self.action_repeat):
            s_t, r, d_t, info = self._env.step(action)
            self.render()

            goal_reward += r 
            if info['cost'] > 0:
                num_cv += 1
            try:
                if info['goal_met']:
                    is_goal_met = 1
                    goal_reward = 10
                    d_t = True

            except:
                pass                
            
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break

        next_state = self._get_original_state()
        h_dist = self._get_hazard_dist()
        vase_penalty = self._get_vase_penalty()

        surrogate_reward = goal_reward + vase_penalty
        reward = r * 10 if r < 0 else r

        info['goal_met'] = is_goal_met
        info['cost_1'] = reward
        info['cost_2'] = self._get_cost(h_dist)
        info['num_cv'] = num_cv
        return next_state, surrogate_reward, done, info

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()


def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    register_env()
    env_name = "Safexp-PointGoal7-v0"
    return GymEnv(env_name, seed, max_episode_length, action_repeat)

def register_env():
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        'constrain_hazards': True,
        'constrain_vases': True,
        'vases_velocity_cost': 1.0,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'robot_locations':[(-2,2)],
        'vases_locations': [(-1,1), (-1,0)],
        'goal_locations':[(0,0)],
        'hazards_num': 5,
        'vases_num': 2
    }

    register(id='Safexp-PointGoal7-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config})
