# from logger import Logger, ExperimentLogger
from agent import Agent
# from env import Env
import importlib

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import safety_gym
import argparse
import pickle
import random
import torch
import wandb
import yaml
import copy
import time
import gym

def train(main_args):
    env_name = main_args.task

    with open(f"{main_args.task}"+"/"+f"{main_args.config}", "r") as f:
    	config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    # custom import 
    module = importlib.import_module(f"{main_args.task}"+".env")
    Env = getattr(module, "Env")
    env = Env(env_name, seed, config["max_ep_len"])
    agent = Agent(env, device, config, main_args.checkpoint)

    # for wandb
    wandb.init(project=env_name)
    # logger = ExperimentLogger(log_file="experiment1.csv")

    for epoch in range(config["epochs"]):
        trajectories = []
        ep_step = 0
        surrogate_rewards_ = []
        costs1_ = []
        costs2_ = []
        cvs = []
        goals = []
        while ep_step < config["max_steps"]:
            state = env.reset()
            surrogate_rewards = 0
            costs1 = 0
            costs2 = 0
            cv = 0
            goal = 0
            step = 0
            while True:
                ep_step += 1
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                next_state, surrogate_reward, done, info = env.step(state, clipped_action)
                cost1 = info['cost_1']
                cost2 = info['cost_2']

                done = True if step >= config["max_ep_len"] else done
                fail = True if step < config["max_ep_len"] and done else False
                trajectories.append([state, action, surrogate_reward, cost1, cost2, done, fail, next_state])

                state = next_state
                surrogate_rewards += surrogate_reward
                costs1 += cost1
                costs2 += cost2
                cv += info['num_cv']
                goal += info['goal_met']

                if done or step >= config["max_ep_len"]:
                    # input('Episode ended')
                    break

            surrogate_rewards_.append(surrogate_rewards)
            costs1_.append(costs1)
            costs2_.append(costs2)
            cvs.append(cv)
            goals.append(goal)
            print("Return: ", surrogate_rewards, costs1, costs2)
        v_loss, cost1_v_loss, cost2_v_loss, objective, cost1_surrogate, cost2_surrogate, kl, entropy = agent.train(trajs=trajectories)
        surrogate_rewards = np.mean(surrogate_rewards_)
        costs1 = np.mean(costs1_)
        costs2 = np.mean(costs2_)
        cv = np.mean(cvs)
        goal_rate = np.mean(goals)
        log_data = {"returns: U_H":surrogate_rewards, 'constraint violations':cv, 'goal rate':goal_rate, \
                    "returns: R_A":costs1, "costs: C_A":costs2}
        # logger.log_epoch(epoch, surrogate_rewards, costs1, costs2, cv, goal_rate)
        wandb.log(log_data)
        if (epoch + 1)%config["save_freq"] == 0:
            agent.save()


def test(main_args):
    env_name = main_args.task

    with open(f"{main_args.task}"+"/"+f"{main_args.config}", "r") as f:
    	config = yaml.safe_load(f)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    # custom import 
    module = importlib.import_module(f"{main_args.task}"+".env")
    Env = getattr(module, "Env")
    env = Env(env_name, seed, config["max_ep_len"])
    agent = Agent(env, device, config, main_args.checkpoint)

    test_running_reward = 0
    print("--------------------------------------------------------------------------------------------")

    for ep in range(1, config["test_episodes"]+1):
        state = env.reset()
        R_A = 0
        C_A = 0
        U_H = 0
        cv = 0
        while True:
            state_tensor = torch.tensor(state, device=device, dtype=torch.float)
            action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=False)
            action = action_tensor.detach().cpu().numpy()
            clipped_action = clipped_action_tensor.detach().cpu().numpy()
            next_state, reward, done, info = env.step(state, clipped_action)
            cost1 = info['cost_1']
            cost2 = info['cost_2']
            
            R_A += cost1
            C_A += cost2
            U_H += reward
            cv += info['num_cv']

            state = next_state

            if done:
                break
        print('Episode: {} \t\t R_A: {} \t\t C_A: {} \t\t U_H: {} \t\t Goal: {} \t\t CV: {}'.format(ep, round(R_A, 2), round(C_A, 2), round(U_H, 2), info['goal_met'], cv))
        env.close()

    print("============================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SEPS')
    parser.add_argument('--test', action='store_true', help='For test.')
    # parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--checkpoint', default='checkpoint', help='To use a specific checkpoint file')
    parser.add_argument('--config', default='config.yaml', help='To use a specific config file')
    parser.add_argument('--task', required=True, help='specify the task name, ex: pointGoal or pointButton')

    args = parser.parse_args()
    dict_args = vars(args)

    if args.test:
        test(args)
    else:
        train(args)
