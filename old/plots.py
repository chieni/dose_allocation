import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns 

from solvers import Solver

def regret_subplot(data, ax, hue_name='solver'):
    # Sub.fig. 1: Regrets in time.
    sns.lineplot(x='timestep', y='regret', hue=hue_name, data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative regret')
    ax.grid('k', ls='--', alpha=0.3)

def groups_regret_subplot(data, ax):
    # Sub.fig. 1: Regrets in time.
    sns.lineplot(x='timestep', y='group_regret', hue='group', data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative regret by group')
    ax.grid('k', ls='--', alpha=0.3)    

def reward_subplot(data, ax, hue_name='solver'):
    # Sub.fig. 2: Cumulative rewards over time
    sns.lineplot(x='timestep', y='reward', hue=hue_name, data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative reward')
    ax.grid('k', ls='--', alpha=0.3)

def groups_reward_subplot(data, ax):
    # Sub.fig. 2: Cumulative rewards over time
    sns.lineplot(x='timestep', y='group_reward', hue='group', data=data, ax=ax)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative reward by group')
    ax.grid('k', ls='--', alpha=0.3)    

def mean_estimate_subplot(arm_indices, true_means, data, ax, hue_name='solver'):
    # Sub.fig. 3: Probabilities estimated by solvers.
    ax.plot(arm_indices, [true_means[x] for x in arm_indices], 'k--', markersize=12, label='ground truth')
    if hue_name is not None:
        sns.pointplot(x='arm_idx', y='mean_estimate', hue=hue_name, data=data, ci='sd', legend=True, ax=ax, join=False)
    else:
        sns.pointplot(x='arm_idx', y='mean_estimate', data=data, ci='sd', legend=True, ax=ax, join=False)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Estimated mean')
    ax.grid('k', ls='--', alpha=0.3)

def grouped_mean_estimate_subplot(arm_indices, data, ax, reward_dict):
    # TODO!!
    # Sub.fig. 3: Probabilities estimated by solvers.
    colors = ['blue', 'orange']
    for group_idx, group_probs in reward_dict.items():
        ax.plot(arm_indices, [group_probs[x] for x in arm_indices], 'k--', markersize=12,
                color=colors[group_idx], label=f"{group_idx} ground truth")
    sns.pointplot(x='arm_idx', y='mean_estimate', data=data, ci='sd', legend=True, ax=ax, join=False)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Estimated mean')
    ax.grid('k', ls='--', alpha=0.3)
    ax.legend()

def std_estimate_subplot(arm_indices, true_stds, data, ax, hue_name='solver'):
    # Sub.fig. 4: Estimated uncertainties
    ax.plot(arm_indices, [true_stds[x] for x in arm_indices], 'k--', markersize=12, label='ground truth std')
    if hue_name is not None:
        sns.pointplot(x='arm_idx', y='uncertainty_estimate', hue=hue_name, data=data, ci='sd', legend=True, ax=ax, join=False)
    else:
        sns.pointplot(x='arm_idx', y='uncertainty_estimate', data=data, ci='sd', legend=True, ax=ax, join=False)
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Estimated uncertainty')
    ax.grid('k', ls='--', alpha=0.3)

def action_counts_subplot(num_timesteps, data, ax, hue_name='solver'):
    # Sub.fig. 5: Action counts
    data['fraction'] = data['count'] / float(num_timesteps)
    sns.lineplot(x='arm_idx', y='fraction', hue=hue_name, data=data, legend=True, ax=ax)
    num_arms = data['arm_idx'].nunique()
    ax.set_xticks(np.arange(0, num_arms, 1))
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Fraction of samples')
    ax.grid('k', ls='--', alpha=0.3)

def group_action_counts_subplot(num_timesteps, data, ax):
    # Sub.fig. 5: Action counts
    group_arm_counts = data.value_counts(subset=['group', 'selected_arm', 'trial']).reset_index()
    group_arm_counts.columns = ['group', 'arm_idx', 'trial', 'count']

    total_arm_counts = data.value_counts(subset=['selected_arm', 'trial']).reset_index()
    total_arm_counts.columns = ['arm_idx', 'trial', 'count']
    total_arm_counts['group'] = ['all'] * len(total_arm_counts)

    plot_data = pd.concat([group_arm_counts, total_arm_counts])

    # data['fraction'] = data['count'] / float(num_timesteps)
    sns.pointplot(x='arm_idx', y='count', hue='group', data=plot_data, ci='sd', legend=True, ax=ax, join=False)
    num_arms = plot_data['arm_idx'].nunique()
    ax.set_xticks(np.arange(0, num_arms, 1))
    ax.set_xlabel('Actions by index')
    ax.set_ylabel('Fraction of samples')
    ax.grid('k', ls='--', alpha=0.3)

def plot_results_by_solver(solver_names, metric_frames, arms_frames, plot_filename):
    """
    Plot the results by multi-armed bandit solvers.
    Args:
        solver_names (list<str)
        plot_filename (str)
    """

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4)
    # fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax5 = plt.subplot(gs[4])

    # Concatenate metric frames for all types of solvers
    all_metric_frames = []
    for idx, metric_frame in enumerate(metric_frames):
        metric_frame['solver'] = solver_names[idx]
        all_metric_frames.append(metric_frame)
    all_metric_frame = pd.concat(all_metric_frames)

    all_arms_frames = []
    for idx, arms_frame in enumerate(arms_frames):
        arms_frame['solver'] = solver_names[idx]
        all_arms_frames.append(arms_frame)
    all_arms_frame = pd.concat(all_arms_frames)

    num_arms = all_arms_frame['arm_idx'].unique().shape[0]
    num_timesteps = all_metric_frame['timestep'].max()
    true_means = arms_frames[0]['true_mean'].values
    true_stds = arms_frames[0]['true_uncertainty'].values
    sorted_indices = range(num_arms)

    regret_subplot(all_metric_frame, ax1)
    reward_subplot(all_metric_frame, ax2)
    mean_estimate_subplot(sorted_indices, true_means, all_arms_frame, ax3)
    std_estimate_subplot(sorted_indices, true_stds, all_arms_frame, ax4)
    action_counts_subplot(num_timesteps, all_arms_frame, ax5)

    plt.savefig(plot_filename)
    plt.close()

# Add plot by subgroup
def plot_results_by_group(metric_frames, arms_frames, plot_filename, group_dist, reward_dict):
    """
    Plot the results by multi-armed bandit solvers.
    Args:
        solver_names (list<str)
        plot_filename (str)
    """

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4)
    # fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])

    # Concatenate metric frames for all trials
    all_metric_frame = pd.concat(metric_frames)
    all_arms_frame = pd.concat(arms_frames)

    num_arms = all_arms_frame['arm_idx'].unique().shape[0]
    num_timesteps = all_metric_frame['timestep'].max()
    true_means = arms_frames[0]['true_mean'].values
    true_stds = arms_frames[0]['true_uncertainty'].values
    sorted_indices = range(num_arms)

    # TODO: regret and reward plots by group incorrect. Regret and reward should be kept by group
    # regret_subplot(all_metric_frame, ax1, hue_name='group')
    # reward_subplot(all_metric_frame, ax2, hue_name='group')
    groups_regret_subplot(all_metric_frame, ax1)
    groups_reward_subplot(all_metric_frame, ax2)
    #mean_estimate_subplot(sorted_indices, true_means, all_arms_frame, ax3, hue_name=None)
    grouped_mean_estimate_subplot(sorted_indices, all_arms_frame, ax3, reward_dict)
    std_estimate_subplot(sorted_indices, true_stds, all_arms_frame, ax4, hue_name=None)
    group_action_counts_subplot(num_timesteps, all_metric_frame, ax5)

    plt.savefig(plot_filename)
    plt.close()