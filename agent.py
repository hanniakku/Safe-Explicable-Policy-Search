from typing import Optional, List

from models import Policy
from models import Value

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

class Agent:
    def __init__(self, env, device, args, checkpoint_file):
        # default
        self.device = device

        # member variable
        self.checkpoint_dir='{}/checkpoint'.format(args['task'])
        self.checkpoint_file = checkpoint_file
        self.discount_factor = args['discount_factor']
        self.gae_coeff = args['gae_coeff']
        self.damping_coeff = args['damping_coeff']
        self.num_conjugate = args['num_conjugate']
        self.max_decay_num = args['max_decay_num']
        self.line_decay = args['line_decay']
        self.max_kl = args['max_kl']
        self.v_lr = args['v_lr']
        self.cost_v_lr = args['cost_v_lr']
        self.value_epochs = args['value_epochs']
        self.batch_size = args['batch_size']
        self.cost_d1 = args['cost_d1']
        self.cost_d2 = args['cost_d2']

        # constant about env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound_min = torch.tensor(env.action_space.low, device=device)
        self.action_bound_max = torch.tensor(env.action_space.high, device=device)

        # declare value and policy
        args['state_dim'] = self.state_dim
        args['action_dim'] = self.action_dim
        self.policy = Policy(args).to(device)
        self.value = Value(args).to(device)
        self.cost1_value = Value(args).to(device)
        self.cost2_value = Value(args).to(device)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.v_lr)
        self.cost1_v_optimizer = torch.optim.Adam(self.cost1_value.parameters(), lr=self.cost_v_lr)
        self.cost2_v_optimizer = torch.optim.Adam(self.cost2_value.parameters(), lr=self.cost_v_lr)
        self.load()


    def normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getAction(self, state:torch.Tensor, is_train:bool) -> List[torch.Tensor]:
        '''
        input:
            states:     Tensor(state_dim,)
            is_train:   boolean
        output:
            action:         Tensor(action_dim,)
            cliped_action:  Tensor(action_dim,)
        '''
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise*std)
        else:
            action = self.unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action

    def getGaesTargets(self, rewards:np.ndarray, values:np.ndarray, dones:np.ndarray, fails:np.ndarray, next_values:np.ndarray) -> List[np.ndarray]:
        '''
        input:
            rewards:        np.array(n_steps,)
            values:         np.array(n_steps,)
            dones:          np.array(n_steps,)
            fails:          np.array(n_steps,)
            next_values:    np.array(n_steps,)
        output:
            gaes:       np.array(n_steps,)
            targets:    np.array(n_steps,)
        '''
        deltas = rewards + (1.0 - fails)*self.discount_factor*next_values - values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*self.discount_factor*self.gae_coeff*gaes[t + 1]
        targets = values + gaes
        return gaes, targets

    def getEntropy(self, states:torch.Tensor) -> torch.Tensor:
        '''
        return scalar tensor for entropy value.
        input:
            states:     Tensor(n_steps, state_dim)
        output:
            entropy:    Tensor(,)
        '''
        means, log_stds, stds = self.policy(states)
        normal = torch.distributions.Normal(means, stds)
        entropy = torch.mean(torch.sum(normal.entropy(), dim=1))
        return entropy

    def train(self, trajs):
        # convert to numpy array
        states = np.array([traj[0] for traj in trajs])
        actions = np.array([traj[1] for traj in trajs])
        rewards = np.array([traj[2] for traj in trajs])
        costs1 = np.array([traj[3] for traj in trajs])
        costs2 = np.array([traj[4] for traj in trajs])
        dones = np.array([traj[5] for traj in trajs])
        fails = np.array([traj[6] for traj in trajs])
        next_states = np.array([traj[7] for traj in trajs])
        
        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float)
        norm_actions_tensor = self.normalizeAction(actions_tensor)
        next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float)

        # get GAEs and Tagets
        # for reward
        values_tensor = self.value(states_tensor)
        next_values_tensor = self.value(next_states_tensor)
        values = values_tensor.detach().cpu().numpy()
        next_values = next_values_tensor.detach().cpu().numpy()
        gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float)
        # for cost1
        cost1_values_tensor = self.cost1_value(states_tensor)
        next_cost1_values_tensor = self.cost1_value(next_states_tensor)
        cost1_values = cost1_values_tensor.detach().cpu().numpy()
        next_cost1_values = next_cost1_values_tensor.detach().cpu().numpy()
        cost1_gaes, cost1_targets = self.getGaesTargets(costs1, cost1_values, dones, fails, next_cost1_values)
        cost1_gaes_tensor = torch.tensor(cost1_gaes, device=self.device, dtype=torch.float)
        cost1_targets_tensor = torch.tensor(cost1_targets, device=self.device, dtype=torch.float)
        # for cost2
        cost2_values_tensor = self.cost2_value(states_tensor)
        next_cost2_values_tensor = self.cost2_value(next_states_tensor)
        cost2_values = cost2_values_tensor.detach().cpu().numpy()
        next_cost2_values = next_cost2_values_tensor.detach().cpu().numpy()
        cost2_gaes, cost2_targets = self.getGaesTargets(costs2, cost2_values, dones, fails, next_cost2_values)
        cost2_gaes_tensor = torch.tensor(cost2_gaes, device=self.device, dtype=torch.float)
        cost2_targets_tensor = torch.tensor(cost2_targets, device=self.device, dtype=torch.float)

        # get cost mean
        cost1_mean = np.mean(costs1)/(1 - self.discount_factor)
        cost2_mean = np.mean(costs2)/(1 - self.discount_factor)

        # get entropy
        entropy = self.getEntropy(states_tensor)

        # ======================================= #
        # ========== for policy update ========== #
        # backup old policy
        means, log_stds, stds = self.policy(states_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()

        # get objective & KL & cost surrogate
        objective = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
        cost1_surrogate_ = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, old_stds, cost1_gaes_tensor, cost1_mean)
        cost1_surrogate = torch.negative(cost1_surrogate_)
        cost2_surrogate = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, old_stds, cost1_gaes_tensor, cost2_mean)
        kl = self.getKL(states_tensor, old_means, old_stds)

        # get gradient
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        grad_b0 = flatGrad(-cost1_surrogate, self.policy.parameters(), retain_graph=True)
        grad_b1 = flatGrad(-cost2_surrogate, self.policy.parameters(), retain_graph=True)
        x_value = self.conjugateGradient(kl, grad_g)
        approx_g = self.Hx(kl, x_value)
        cost_d1 = self.cost_d1/(1.0 - self.discount_factor)
        cost_d2 = self.cost_d2/(1.0 - self.discount_factor)
        c0 = cost1_surrogate + cost_d1
        c1 = cost2_surrogate - cost_d2

        # solve Lagrangian problem
        if torch.dot(grad_b1, grad_b1) <= 1e-8 and c1 < 0 and torch.dot(grad_b0, grad_b0) <= 1e-8 and c0 < 0:
            # Gradients are close to zero. 
            # Indicates constraint is effectively not active.
            # Policy update may be considered as unconstrained case
            H_inv_b1, H_inv_b0, r1, r0, s1, s0, t = 0, 0, 0, 0, 0, 0, 0
            q = torch.dot(approx_g, x_value)
            optim_case = 10
        else:
            H_inv_b1 = self.conjugateGradient(kl, grad_b1)
            H_inv_b0 = self.conjugateGradient(kl, grad_b0)
            approx_b1 = self.Hx(kl, H_inv_b1)
            approx_b0 = self.Hx(kl, H_inv_b0)
            q = torch.dot(approx_g, x_value)
            r1 = torch.dot(approx_g, H_inv_b1)
            r0 = torch.dot(approx_g, H_inv_b0)
            s1 = torch.dot(approx_b1, H_inv_b1)
            s0 = torch.dot(approx_b0, H_inv_b0)
            t = torch.dot(approx_b0, H_inv_b1)
            denom = s0 * s1 - t**2
            A1 = q - (s1*r0**2 + s0*r1**2 - 2*t*r0*r1)/(denom+EPS)
            B1 = 2*self.max_kl - (s1*c0**2 + s0*c1**2 - 2*t*c0*c1)/(denom+EPS)
            C1 = (t*r1*c0 + t*r0*c1 - s1*r0*c0 - s0*r1*c1)/(denom+EPS)
            A2 = q - (r0**2)/(s0+EPS) #fa2
            B2 = 2*self.max_kl - (c0**2)/(s0+EPS) #fa2
            C2 = - (r0*c0)/(s0+EPS) #fa2 coefficient without lam
            A3 = q - (r1**2)/(s1+EPS) #fb1
            B3 = 2*self.max_kl - (c1**2)/(s1+EPS) #fb1
            C3 = - (r1*c1)/(s1+EPS) #fb1 coefficient without lam
            A4 = q #fb2
            B4 = 2*self.max_kl #fb2
            C4 = 0 #fb2 coefficient without lam
            # print(c0, c1, B1, B2, B3)
            # print("c0 < 0: ", c0 < 0, "c1 < 0: ", c1 < 0, "B1 < 0: ", B1 < 0, "B2 < 0: ", B2 < 0, "B3 < 0: ", B3 < 0)
            if c1 < 0 and c0 < 0 and B1 < 0 and B2 < 0 and B3 < 0:
                # Indicates both constraints are not violated and safety boundary does not intersect the trust region
                optim_case = 9
            elif c1 < 0 and c0 < 0 and B1 >= 0:
                # Indicates constraints are not violated and safety boundary of both constraints intersects the trust region
                optim_case = 8
            elif c1 >= 0 and c0 >= 0 and B1 >= 0:
                # indicates both of the constraints are violated and safety boundary intersects the trust region
                optim_case = 5
            elif c1 < 0 and c0 < 0 and B2 >= 0:
                # Indicates constraints are not violated and safety boundary of 1st constraint intersects the trust region
                optim_case = 7
            elif c0 >= 0 and B2 >= 0:
                # indicates the 1st constraint is violated and safety boundary intersects the trust region
                optim_case = 4
            elif c1 < 0 and c0 < 0 and B3 >= 0:
                # Indicates constraints are not violated and safety boundary of 2nd constraint intersects the trust region
                optim_case = 6
            elif c1 >= 0 and B3 >= 0:
                # indicates the 2nd constraint is violated and safety boundary intersects the trust region
                optim_case = 3
            elif (c0 >= 0 or c1 >= 0): #and B1 < 0 and B2 < 0 and B3 < 0:
                # indicates one or both constraint are violated and safety boundary does not intersect the trust region
                if c0 >= 0 and c1 >= 0:
                    print("both constraints violated")
                    optim_case = 2
                if c0 >= 0:
                    optim_case = 1
                else:
                    optim_case = 0
            else:
                optim_case = -1
            # else:
            #   # c1 >= 0 or c0 >= 0 and B_value < 0:
            #   # Indicates both constraints are violated and safety boundary does not intersect the trust region
            #   optim_case = 0
        print("optimizing case :", optim_case)
        if optim_case in [10,9]:
            lam = torch.sqrt(A4/B4)
            nu1 = 0
            nu2 = 0
        elif optim_case in [8,5]:
            if (r1*s0 - t*r0)/(s0*c1 - t*c0) > 0 and (r0*s1 - t*r1)/(s1*c0 - t*c1) > 0:
                LA, LB = [max((r1*s0 - t*r0)/(s0*c1 - t*c0), (r0*s1 - t*r1)/(s1*c0 - t*c1)), np.inf], [0, max((r1*s0 - t*r0)/(s0*c1 - t*c0), (r0*s1 - t*r1)/(s1*c0 - t*c1))]
            else:
                LA, LB = [0, max((r1*s0 - t*r0)/(s0*c1 - t*c0), (r0*s1 - t*r1)/(s1*c0 - t*c1))], [max((r1*s0 - t*r0)/(s0*c1 - t*c0), (r0*s1 - t*r1)/(s1*c0 - t*c1)), np.inf]
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A1/B1), LA) 
            lam_b = proj(torch.sqrt(A4/B4), LB)
            f_a = lambda lam : -0.5 * (A1 / (lam + EPS) + B1 * lam) + C1
            f_b = lambda lam : -0.5 * (A4 / (lam + EPS) + B4 * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu1 = max(0, (lam * (s0*c1 - t*c0) - r1*s0 + t*r0)/(denom+EPS))
            nu2 = max(0, (lam * (s1*c0 - t*c1) - r0*s1 + t*r1)/(denom+EPS))
        elif optim_case in [7,4]:
            if c0 >= 0:
                LA, LB = [r0/c0, np.inf], [0, r0/c0]
            else:
                LA, LB = [0, r0/c0], [r0/c0, np.inf]
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A2/B2), LA) 
            lam_b = proj(torch.sqrt(A4/B4), LB)
            f_a = lambda lam : -0.5 * (A2 / (lam + EPS) + B2 * lam) + C2
            f_b = lambda lam : -0.5 * (A4 / (lam + EPS) + B4 * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu1 = max(0, (lam*c0 - r0)/(s0 + EPS))
            nu2 = 0
        elif optim_case in [6,3]:
            if c1 >= 0:
                LA, LB = [r1/c1, np.inf], [0, r1/c1]
            else:
                LA, LB = [0, r1/c1], [r1/c1, np.inf]
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A3/B3), LA) 
            lam_b = proj(torch.sqrt(A4/B4), LB)
            f_a = lambda lam : -0.5 * (A3 / (lam + EPS) + B3 * lam) + C3
            f_b = lambda lam : -0.5 * (A4 / (lam + EPS) + B4 * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu1 = 0
            nu2 = max(0, (lam*c1 - r1)/(s1 + EPS))
        else:
            if c0 >= 0 and c1 >= 0: 
                lam = 0
                nu1 = torch.sqrt(2*self.max_kl / (s0+EPS))
                nu2 = torch.sqrt(2*self.max_kl / (s1+EPS))
            elif c0 >= 0: 
                lam = 0
                nu1 = torch.sqrt(2*self.max_kl / (s0+EPS))
                nu2 = 0
            else:
                lam = 0
                nu1 = 0
                nu2 = torch.sqrt(2*self.max_kl / (s1+EPS))

        ## line search
        if optim_case > 2:
            delta_theta = (1./(lam + EPS)) * (x_value + nu1*H_inv_b0 + nu2*H_inv_b1)
        elif optim_case == 2:
            delta_theta = nu1*H_inv_b0 + nu2*H_inv_b1 
        elif optim_case == 1:
            delta_theta = nu1*H_inv_b0
        elif optim_case == 0:
            delta_theta = nu2*H_inv_b1
        else:
            delta_theta = 0

        beta = 1.0
        init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        init_cost1_surrogate = cost1_surrogate.clone().detach()
        init_cost2_surrogate = cost2_surrogate.clone().detach()
        while True:
            theta = beta*delta_theta + init_theta
            self.applyParams(theta)
            objective = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
            cost1_surrogate_ = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, old_stds, cost1_gaes_tensor, cost1_mean)
            cost1_surrogate = torch.negative(cost1_surrogate_)
            cost2_surrogate = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, old_stds, cost2_gaes_tensor, cost2_mean)
            kl = self.getKL(states_tensor, old_means, old_stds)
            # if kl <= self.max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c1_value, 0):
            #     break

            # Check if the new objective improves or remains within constraints
            objective_improved = objective > init_objective if optim_case > 5 else True
            cost_within_bounds = (cost1_surrogate - init_cost1_surrogate <= max(-c0, 0) and
                                    cost2_surrogate - init_cost2_surrogate <= max(-c1, 0))
            # Check if the KL divergence is within the trust region
            kl_within_TR = True if kl <= self.max_kl else False
                
            # If KL is within trust reagion and the objective improves and constraints are satisfied, break the line search
            if kl_within_TR and objective_improved and cost_within_bounds:
                break
                    
            beta *= self.line_decay
        # ======================================= #

        # ======================================== #
        # =========== for value update =========== #
        for _ in range(self.value_epochs):
            value_loss = torch.mean(0.5*torch.square(self.value(states_tensor) - targets_tensor))
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

            cost1_value_loss = torch.mean(0.5*torch.square(self.cost1_value(states_tensor) - cost1_targets_tensor))
            self.cost1_v_optimizer.zero_grad()
            cost1_value_loss.backward()
            self.cost1_v_optimizer.step()

            cost2_value_loss = torch.mean(0.5*torch.square(self.cost2_value(states_tensor) - cost2_targets_tensor))
            self.cost2_v_optimizer.zero_grad()
            cost2_value_loss.backward()
            self.cost2_v_optimizer.step()
        # ======================================== #

        scalar = lambda x:x.detach().cpu().numpy()
        np_value_loss = scalar(value_loss)
        np_cost1_value_loss = scalar(cost1_value_loss)
        np_cost2_value_loss = scalar(cost2_value_loss)
        np_objective = scalar(objective)
        np_cost1_surrogate = scalar(cost1_surrogate)
        np_cost2_surrogate = scalar(cost2_surrogate)
        np_kl = scalar(kl)
        np_entropy = scalar(entropy)
        return np_value_loss, np_cost1_value_loss, np_cost2_value_loss, np_objective, np_cost1_surrogate, np_cost2_surrogate, np_kl, np_entropy

    def getObjective(self, states, norm_actions, gaes, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)
        objective = torch.mean(torch.exp(log_probs - old_log_probs)*gaes)
        return objective

    def getCostSurrogate(self, states, norm_actions, old_means, old_stds, cost_gaes, cost_mean):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)
        cost_surrogate = cost_mean + (1.0/(1.0 - self.discount_factor))*(torch.mean(torch.exp(log_probs - old_log_probs)*cost_gaes) - torch.mean(cost_gaes))
        return cost_surrogate

    def getKL(self, states, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

    def applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def Hx(self, kl:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def conjugateGradient(self, kl:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self.Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/rs_old)*p
            rs_old = rs_new
        return x

    def save(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'cost1_value': self.cost1_value.state_dict(),
            'cost2_value': self.cost2_value.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'cost1_v_optimizer': self.cost1_v_optimizer.state_dict(),
            'cost2_v_optimizer': self.cost2_v_optimizer.state_dict(),
            }, f"{self.checkpoint_dir}/checkpoint")
        print('[save] success.')

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/{self.checkpoint_file}"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.cost1_value.load_state_dict(checkpoint['cost1_value'])
            self.cost2_value.load_state_dict(checkpoint['cost2_value'])
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
            self.cost1_v_optimizer.load_state_dict(checkpoint['cost1_v_optimizer'])
            self.cost2_v_optimizer.load_state_dict(checkpoint['cost2_v_optimizer'])
            print('[load] success.')
        else:
            self.policy.initialize()
            self.value.initialize()
            self.cost1_value.initialize()
            self.cost2_value.initialize()
            print('[load] fail.')
