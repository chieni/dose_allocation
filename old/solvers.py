import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import NormalDist


np.random.seed(10)

class Solver:
    def __init__(self, bandit, init_proba, group_indices=None):
        """
        bandit (Bandit): the target bandit to solve.
        """
        self.bandit = bandit

        self.counts = [0] * self.bandit.num_arms
        self.cumulative_reward = 0.
        self.regret = 0.  # Cumulative regret.

        if group_indices is not None:
            self.group_rewards = {key: 0 for key in group_indices}
            self.group_regrets = {key: 0 for key in group_indices}

        # self.estimates is the empirical mean of the observed reward samples
        self.estimates = [init_proba] * self.bandit.num_arms # Optimistic initialization

    def update_regret(self, selected_arm, group_idx):
        # i (int): index of the selected machine.
        if group_idx is not None:
            regret = self.bandit.best_reward_prob_dict[group_idx] - self.bandit.reward_dict[group_idx][selected_arm]
            self.regret += regret    
            self.group_regrets[group_idx] += regret
        else:
            self.regret += self.bandit.best_reward_prob - self.bandit.reward_probs[selected_arm]
        return self.regret
    
    def update_reward(self, reward, group_idx):
        self.cumulative_reward += reward
        if group_idx is not None:
            self.group_rewards[group_idx] += reward
        return self.cumulative_reward
    
    def get_groups_reward_regret(self, group_idx):
        if group_idx is None:
            return None, None
        return self.group_rewards[group_idx], self.group_regrets[group_idx]

    def get_uncertainty(self, arm, timestep):
        return np.sqrt(2. * np.log(1 + timestep * np.log(timestep)**2) / (self.counts[arm])) 

    def update_estimates(self, selected_arm, group_idx):
        reward = self.bandit.select_arm(selected_arm, group_idx)
        self.estimates[selected_arm] += 1. / (self.counts[selected_arm]) * (reward - self.estimates[selected_arm])
        return reward

    def select_arm(self):
        """Return the machine index to take action on."""
        raise NotImplementedError
    
    def run_one_step(self, timestep, selected_arm, group_idx):
        self.counts[selected_arm] += 1
        reward = self.update_estimates(selected_arm, group_idx)
        updated_reward = self.update_reward(reward, group_idx)
        updated_regret = self.update_regret(selected_arm, group_idx)
        group_reward, group_regret = self.get_groups_reward_regret(group_idx)
        metric_dict = {
            'timestep': timestep,
            'selected_arm': selected_arm,
            'reward': updated_reward,
            'regret': updated_regret,
            'group_reward': group_reward,
            'group_regret': group_regret,
            'group': group_idx
        }
        return metric_dict
        
    def run(self, num_steps, group_sampler=None):
        metric_dicts = []
        timestep = 0

        # Try sampling each arm once to start off with
        for arm_idx in self.bandit.arm_indices:
            group_sample = None
            if group_sampler is not None:
                group_sample = group_sampler.sample_one()
            metric_dict = self.run_one_step(timestep, arm_idx, group_sample)
            metric_dicts.append(metric_dict)
            timestep += 1

        for _ in range(num_steps):
            group_sample = None
            if group_sampler is not None:
                group_sample = group_sampler.sample_one()
            selected_arm = self.select_arm(timestep)
            metric_dict = self.run_one_step(timestep, selected_arm, group_sample)
            metric_dicts.append(metric_dict)
            timestep += 1

        final_uncertainties = [self.get_uncertainty(arm, timestep) for arm in self.bandit.arm_indices]
        arms_dict = {'arm_idx': self.bandit.arm_indices,
                     'uncertainty_estimate': final_uncertainties,
                     'mean_estimate': self.estimates,
                     'count': self.counts
                     }
        return pd.DataFrame(metric_dicts), pd.DataFrame(arms_dict)

class EpsilonGreedy(Solver):

    def __init__(self, bandit, epsilon, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super().__init__(bandit, init_proba)

        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon

    def select_arm(self, timestep):
        available_arms = self.bandit.arm_indices

        if np.random.random() < self.epsilon:
            # Let's do random exploration!
            selected_arm = np.random.choice(available_arms)
        else:
            # Pick the best one.
            selected_arm = max(available_arms, key=lambda x: self.estimates[x])

        return selected_arm


class UCB(Solver):
    '''
    UCB1
    '''
    def __init__(self, bandit, init_proba=1.0, group_indices=None):
        """
        eps (float): the probability to explore at each time step.
        """
        super().__init__(bandit, init_proba, group_indices)

    def select_arm(self, timestep):
        epsilon = 0.01

        # Pick the best one with consideration of upper confidence bounds.
        available_arms = self.bandit.arm_indices

        bounds = {arm: self.get_ucb(arm, timestep) for arm in available_arms}

        selected_arm = max(bounds, key=bounds.get)
        max_arms = [selected_arm]
        for arm, bound in bounds.items():
            if arm != selected_arm:
                if bound + epsilon > bounds[selected_arm]:
                    max_arms.append(arm)

        selected_arm = np.random.choice(max_arms)
        return selected_arm

    def get_ucb(self, arm, timestep):
        ucb = self.estimates[arm] + self.get_uncertainty(arm, timestep)
        return ucb
        

class LinUCB:
    def __init__(self, bandit, alpha=2.0, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        self.bandit = bandit
        self.alpha = alpha
        assert self.bandit.context is not None
        self.context_dim = self.bandit.context.shape[-1]

    def get_best_reward(self, X_timepoint):
        rewards = [self.bandit.select_arm(arm, X_timepoint[arm]) for arm in range(self.bandit.num_arms)]
        best_reward = np.max(rewards)
        return best_reward
    
    def get_random_reward(self, X_timepoint):
        random_arm = np.random.choice(self.bandit.num_arms)
        random_reward = self.bandit.select_arm(random_arm, X_timepoint[random_arm])
        return random_reward

    def run(self, num_timepoints):
        contexts = self.bandit.context
        
        oracle = np.empty(num_timepoints)
        rewards = np.empty(num_timepoints)
        random_rewards = np.empty(num_timepoints)
        selected_arms = np.empty(num_timepoints)
        theta = np.empty(shape=(num_timepoints, self.bandit.num_arms, self.context_dim))
        # expected reward
        predictions = np.empty(shape=(num_timepoints, self.bandit.num_arms))

        A_matrix = [np.identity(self.context_dim) for arm in range(self.bandit.num_arms)]
        b_vector = [np.zeros(self.context_dim) for arm in range(self.bandit.num_arms)] 

        for timestep in range(num_timepoints):
            # For each arm, calculate theta (regression coefficient) and ucb (confidence bound)
            for arm_idx in range(self.bandit.num_arms):
                A = A_matrix[arm_idx]
                b = b_vector[arm_idx]
                A_inv = np.linalg.inv(A)
                theta[timestep, arm_idx] = np.dot(A_inv, b)

                # Context for timestep, arm_idx
                X_ta = contexts[timestep, arm_idx, :]

                predictions[timestep, arm_idx] = \
                    np.dot(theta[timestep, arm_idx].T, X_ta) + (self.alpha * np.sqrt(np.dot(X_ta.T, np.dot(A_inv, X_ta))))
                
                

            # Select arm with highest confidence bound
            selected_arm = np.argmax(predictions[timestep])
            X_selected_arm = contexts[timestep, selected_arm, :]

            rewards[timestep] = self.bandit.select_arm(selected_arm, X_selected_arm)
            selected_arms[timestep] = selected_arm 

            # Get oracle
            oracle[timestep] = self.get_best_reward(contexts[timestep, :, :])
            random_rewards[timestep] = self.get_random_reward(contexts[timestep, :, :])

            # Update A_matrix and b_matrix
            A_matrix[selected_arm] += np.outer(X_selected_arm, X_selected_arm.T)
            b_vector[selected_arm] += rewards[timestep] * X_selected_arm


        return theta, predictions, selected_arms, rewards, oracle, random_rewards
