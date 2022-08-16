import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bandits import BernoulliBandit, ContextualBandit, GaussianBandit, GroupedBernoulliBandit
from samplers import GroupSampler
from solvers import Solver, EpsilonGreedy, UCB, LinUCB
from plots import *
from utils import NormalDist


def _get_increasing_means_dist(num_arms, variance=1):
    return [NormalDist(arm, variance) for arm in range(num_arms)]

def _get_increasing_variance_dist(num_arms, mean=0):
    return [NormalDist(mean, (0.5 * arm) + 0.5) for arm in range(num_arms)]

def bernoulli_experiment(reward_probs, timesteps, output_directory, num_trials, num_arms):
    solver_names = []
    all_metric_frames = []
    all_arms_frames = []
    for trial in range(num_trials):
        bandit = BernoulliBandit(num_arms, reward_probs)
        test_solvers = {
        r'$\epsilon$' + '-Greedy': EpsilonGreedy(bandit, 0.1),
        "UCB": UCB(bandit)
        }

        metric_frames = []
        arms_frames = []
        for solver_name, solver in test_solvers.items():
            solver_names.append(solver_name)
            metric_frame, arms_frame = solver.run(timesteps)
            arms_frame['true_mean'] = list(reward_probs)
            arms_frame['true_uncertainty'] = [0 for prob in reward_probs]
            metric_frames.append(metric_frame)
            arms_frames.append(arms_frame)

            metric_frame['trial'] = trial
            arms_frame['trial'] = trial
            all_metric_frames.append(metric_frame)
            all_arms_frames.append(arms_frame)


        plot_directory = os.path.join(output_directory, str(trial))
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plot_filename = os.path.join(plot_directory, f"results_K{num_arms}_N{timesteps}_trial{trial}.png")
        plot_results_by_solver(solver_names, metric_frames, arms_frames, plot_filename)

    plot_all_filename = os.path.join(output_directory, f"all_results_K{num_arms}_N{timesteps}.png")
    plot_results_by_solver(solver_names, all_metric_frames, all_arms_frames, plot_all_filename)


def group_bernoulli_experiment(reward_probs, timesteps, output_directory, num_trials, num_arms, group_dist):
    solver_names = []
    all_metric_frames = []
    all_arms_frames = []
    for trial in range(num_trials):
        bandit = BernoulliBandit(num_arms, reward_probs)
        solver = UCB(bandit)
        group_sampler = GroupSampler(group_dist)

        metric_frame, arms_frame = solver.run(timesteps, group_sampler)
        arms_frame['true_mean'] = list(reward_probs)
        arms_frame['true_uncertainty'] = [0 for prob in reward_probs]
        metric_frame['trial'] = trial
        arms_frame['trial'] = trial
        all_metric_frames.append(metric_frame)
        all_arms_frames.append(arms_frame)

        plot_directory = os.path.join(output_directory, str(trial))
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plot_filename = os.path.join(plot_directory, f"results_K{num_arms}_N{timesteps}_trial{trial}.png")
        plot_results_by_group([metric_frame], [arms_frame], plot_filename, group_dist)

    plot_all_filename = os.path.join(output_directory, f"all_results_K{num_arms}_N{timesteps}.png")
    plot_results_by_group(all_metric_frames, all_arms_frames, plot_all_filename, group_dist)


def grouped_bernoulli_experiment(reward_dict, timesteps, output_directory, num_trials, num_arms, group_dist):
    all_metric_frames = []
    all_arms_frames = []
    group_indices = list(range(len(group_dist)))
    for trial in range(num_trials):
        bandit = GroupedBernoulliBandit(num_arms, reward_dict)
        solver = UCB(bandit, 1.0, group_indices)
        group_sampler = GroupSampler(group_dist)

        metric_frame, arms_frame = solver.run(timesteps, group_sampler)
        arms_frame['true_mean'] = list(reward_probs)
        arms_frame['true_uncertainty'] = [0 for prob in reward_probs]
        metric_frame['trial'] = trial
        arms_frame['trial'] = trial
        all_metric_frames.append(metric_frame)
        all_arms_frames.append(arms_frame)

        plot_directory = os.path.join(output_directory, str(trial))
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plot_filename = os.path.join(plot_directory, f"results_K{num_arms}_N{timesteps}_trial{trial}.png")
        plot_results_by_group([metric_frame], [arms_frame], plot_filename, group_dist, reward_dict)

    plot_all_filename = os.path.join(output_directory, f"all_results_K{num_arms}_N{timesteps}.png")
    plot_results_by_group(all_metric_frames, all_arms_frames, plot_all_filename, group_dist, reward_dict)


def gaussian_experiment(reward_dist, timesteps, output_directory, num_trials, num_arms):
    solver_names = []
    all_metric_frames = []
    all_arms_frames = []
    for trial in range(num_trials):
        bandit = GaussianBandit(num_arms, reward_dist)
        test_solvers = {
        r'$\epsilon$' + '-Greedy': EpsilonGreedy(bandit, 0.1),
        "UCB": UCB(bandit)
        }

        metric_frames = []
        arms_frames = []
        for solver_name, solver in test_solvers.items():
            solver_names.append(solver_name)
            metric_frame, arms_frame = solver.run(timesteps)
            arms_frame['true_mean'] = [dist.mean for dist in reward_dist]
            arms_frame['true_uncertainty'] = [dist.std for dist in reward_dist]
            metric_frames.append(metric_frame)
            arms_frames.append(arms_frame)

            metric_frame['trial'] = trial
            arms_frame['trial'] = trial
            all_metric_frames.append(metric_frame)
            all_arms_frames.append(arms_frame)


        plot_directory = os.path.join(output_directory, str(trial))
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plot_filename = os.path.join(plot_directory, f"results_K{num_arms}_N{timesteps}_trial{trial}.png")
        plot_results_by_solver(solver_names, metric_frames, arms_frames, plot_filename)

    plot_all_filename = os.path.join(output_directory, f"all_results_K{num_arms}_N{timesteps}.png")
    plot_results_by_solver(solver_names, all_metric_frames, all_arms_frames, plot_all_filename)


# Basic experiment with arms of increasing means, on a Bernoulli bandit model
folder_name = "1"
num_arms = 5
timesteps = 50
num_trials = 100
output_directory = 'results'
reward_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
reward_dist = _get_increasing_means_dist(num_arms)
group_dist = [0.5, 0.5]
reward_dict = {0: [0.1, 0.15, 0.2, 0.25, 0.3], 1: [0.1, 0.3, 0.5, 0.7, 0.9]}

# bernoulli_experiment(reward_probs, timesteps, f"{output_directory}/{folder_name}", num_trials, num_arms)

# Now we want to introduce groups; each timestep is an arrival from a different group.
# What does distribution of regret look like for groups of different sizes?

# folder_name = "groups3"
# group_bernoulli_experiment(reward_probs, timesteps, f"{output_directory}/{folder_name}", num_trials, num_arms, group_dist)

folder_name = "groups8"
grouped_bernoulli_experiment(reward_dict, timesteps, f"{output_directory}/{folder_name}", num_trials, num_arms, group_dist)