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
        _, self.robot_name, _, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16 * 2 + 2 + 2
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = self._env.action_space
        self.goal_threshold = np.inf
        self.hazard_size = 0.2
        self.vase_size = 0.1
        self.gremlin_size = 0.1*np.sqrt(2.0)
        self.button_size = 0.1
        self.safety_confidence = 0.0


    def _get_original_state(self, prev_state):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = np.array([self._env.dist_goal()])
        goal_dist = np.clip(goal_dist, 0.0, self.goal_threshold)
        acc = self._env.world.get_sensor('accelerometer')[:2]
        vel = self._env.world.get_sensor('velocimeter')[:2]
        rot_vel = self._env.world.get_sensor('gyro')[2:]
        if self.task_name == 'goal':
            gremlins_lidar = self._env.obs_lidar(self._env.gremlins_obj_pos, 3)
            buttons_lidar = self._env.obs_lidar(self._env.buttons_pos, 3)
            lidar = np.concatenate([gremlins_lidar, buttons_lidar])
        elif self.task_name == 'button':
            hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
            gremlins_lidar = self._env.obs_lidar(self._env.gremlins_obj_pos, 3)
            buttons_lidar = self._env.obs_lidar(self._env.buttons_pos, 3)
            lidar = np.concatenate([hazards_lidar, gremlins_lidar, buttons_lidar])
        button_positions = self._env.buttons_pos
        button_dist = [self._get_button_dist(button_positions[0]), self._get_button_dist(button_positions[1])] # size=2
        try:
            if prev_state == None:
                button1_pushed = [0]
                button2_pushed = [0]
        except:
            button1_pushed = [1] if prev_state[-2] == 1 or self._get_button_dist(button_positions[0]) <= 0 + self.button_size else [0] # size=1
            button2_pushed = [1] if prev_state[-1] == 1 or self._get_button_dist(button_positions[1]) <= 0 + self.button_size else [0] # size=1
        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (lidar - 0.3)/0.3, button_dist, button1_pushed, button2_pushed], axis=0)
        return state

    def _get_button_reward(self, state, next_state):
        #reward for button 1 
        if state[-2] == 1: 
            # button1 has already been pressed, agent can no longer get a reward for the button
            reward1 = 0.0
        else:
            displacement = state[-4] - next_state[-4]
            reward1 = 10.0 if state[-2] == 0 and next_state[-2] == 1 else displacement
        
        #reward for button 2
        if state[-1] == 1: 
            # button2 has already been pressed, agent can no longer get a reward for the button
            reward2 = 0.0
        else:
            displacement = state[-3] - next_state[-3]
            reward2 = 10.0 if state[-1] == 0 and next_state[-1] == 1 else displacement
        return reward1+reward2

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

    def _get_gremlin_dist(self):
        # gremlin
        h_dist = self._get_min_dist(self._env.gremlins_obj_pos, self._env.world.robot_pos()) - self.gremlin_size
        # # hazard
        # h_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
        # # vase
        # v_dist = self._get_min_dist(self._env.vases_pos, self._env.world.robot_pos()) - self.vase_size
        return h_dist

    def _get_button_dist(self, button_pos):
        # buttonlist = []
        pos = self._env.world.robot_pos()
        b_dist = np.linalg.norm(button_pos[:2] - pos[:2]) - self.button_size
        return b_dist

    def get_step_wise_cost(self):
        h_dist = self._get_hazard_dist()
        step_wise_cost =  self.safety_confidence - h_dist
        return step_wise_cost
        
    def reset(self):
        self.t = 0
        self._env.reset()
        # self._env.seed(6)
        state = self._get_original_state(None)
        return state

    def step(self, state, action):
        button = 0
        reward = 0
        goal_reward = 0
        is_goal_met = 0
        num_cv = 0

        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)
            self.render()
            reward += r_t
            goal_reward += r_t 
            if info['cost'] > 0:
                num_cv += 1
            try:
                if info['goal_met']:
                    is_goal_met = 1
                    # goal_reward = 10.0
                    d_t = True
            except:
                pass                
            
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break

        next_state = self._get_original_state(state)
        g_dist = self._get_gremlin_dist()
        button_reward = self._get_button_reward(state, next_state)
        
        if state[-2] == 0 and next_state[-2] == 1:
            button += 1
        if state[-1] == 0 and next_state[-1] == 1:
            button += 1

        surrogate_reward = button_reward + goal_reward
        reward = reward * 10 if reward < 0 else reward

        info['goal_met'] = is_goal_met
        info['cost_1'] = reward
        info['cost_2'] = self._get_cost(g_dist)
        info['num_cv'] = num_cv
        info['button'] = button
        return next_state, surrogate_reward, done, info

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()


def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    register_env()
    env_name = "Safexp-PointGremlinGoal2-v0"
    return GymEnv(env_name, seed, max_episode_length, action_repeat)

def register_env():
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_gremlins': True,
        'observe_buttons': True,
        'constrain_gremlins': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'buttons_keepout': 0.5,
        'gremlins_num': 3,
        'buttons_num': 2
    }

    register(id='Safexp-PointGremlinGoal2-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config})
