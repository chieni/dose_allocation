import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def _plot_gp_helper(ax, x_train, y_train, x_true, y_true, x_test, y_test,
                    y_test_lower, y_test_upper):
    x_test = x_test.cpu().numpy()
    markevery_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[markevery_mask].tolist()

    ax.scatter(x_train, y_train, s=40, c='k', alpha=0.1, label='Training Data')
    ax.plot(x_test, y_test, 'b-', markevery=markevery, marker='o',label='GP Predicted')
    ax.plot(x_true, y_true, 'g-', marker='o', label='True')
    ax.fill_between(x_test, y_test_lower, y_test_upper, alpha=0.5)
    # ax.legend()

def _plot_gp_helper_utility(ax, x_test, x_true, utility):
    x_test = x_test.cpu().numpy()
    markevery_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[markevery_mask].tolist()
    ax.plot(x_test, utility, 'gray', markevery=markevery, marker='o', label='Utility')
    ax.set_ylim([0, 1.1])

def plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices, num_subgroups,
            x_test, y_tox_dist, y_eff_dist, util_func, filename):
    sns.set()
    _, axs = plt.subplots(num_subgroups, 3, figsize=(12, 8))
    for subgroup_idx in range(num_subgroups):
        axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 2].set_title(f"Utility - Subgroup {subgroup_idx}")

        group_x_train = x_train[subgroup_indices == subgroup_idx]
        group_y_tox_train = y_tox_train[subgroup_indices == subgroup_idx]
        group_y_eff_train = y_eff_train[subgroup_indices == subgroup_idx]

        _plot_gp_helper(axs[subgroup_idx, 0], group_x_train, group_y_tox_train,
                        dose_scenario.dose_labels, dose_scenario.toxicity_probs[subgroup_idx, :],
                        x_test, y_tox_dist.mean[subgroup_idx, :], y_tox_dist.lower[subgroup_idx, :],
                        y_tox_dist.upper[subgroup_idx, :])
        _plot_gp_helper(axs[subgroup_idx, 1], group_x_train, group_y_eff_train,
                        dose_scenario.dose_labels, dose_scenario.efficacy_probs[subgroup_idx, :],
                        x_test, y_eff_dist.mean[subgroup_idx, :], y_eff_dist.lower[subgroup_idx, :],
                        y_eff_dist.upper[subgroup_idx, :])
        _plot_gp_helper_utility(axs[subgroup_idx, 2], x_test, dose_scenario.dose_labels,
                                util_func[subgroup_idx, :])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def _plot_gp_timestep_helper(ax, x_train, y_train, x_true, y_true, x_test, y_test,
                             y_test_lower, y_test_upper, y_acqui_func, threshold,
                             selected_dose):
    x_test = x_test.cpu().numpy()
    markevery_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[markevery_mask].tolist()

    ax.scatter(x_train, y_train, s=40, c='k', alpha=0.1, label='Training Data')
    ax.plot(x_test, y_test, 'b-', markevery=markevery, marker='o',label='GP Predicted')
    ax.plot(x_true, y_true, 'g-', marker='o', label='True')
    ax.fill_between(x_test, y_test_lower, y_test_upper, alpha=0.5)
    ax.plot(x_test, y_acqui_func, 'gray', markevery=markevery, marker='o', label='Acquisition Function')
    ax.plot(x_test, np.repeat(threshold, len(x_test)), 'm', label='Threshold')
    ax.plot(x_true[selected_dose], y_acqui_func[markevery_mask][selected_dose], 'r', marker='o')
    ax.set_ylim([0, 1.1])
    # ax.legend()

def _plot_gp_timestep_utility(ax, x_test, x_true, utility, selected_dose):
    x_test = x_test.cpu().numpy()
    markevery_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[markevery_mask].tolist()
    ax.plot(x_test, utility, 'gray', markevery=markevery, marker='o', label='Utility')
    ax.plot(x_true[selected_dose], utility[markevery_mask][selected_dose], 'r', marker='o')
    ax.set_ylim([0, 1.1])

def plot_gp_timestep(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices,
                     num_subgroups, x_test, y_tox_dist, y_eff_dist, 
                     y_tox_acqui_func, y_eff_acqui_func, util_func, selected_doses,
                     filename):
    sns.set()
    _, axs = plt.subplots(num_subgroups, 3, figsize=(12, 8))
    for subgroup_idx in range(num_subgroups):
        axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 2].set_title(f"Utility - Subgroup {subgroup_idx}")

        group_x_train = x_train[subgroup_indices == subgroup_idx]
        group_y_tox_train = y_tox_train[subgroup_indices == subgroup_idx]
        group_y_eff_train = y_eff_train[subgroup_indices == subgroup_idx]

        _plot_gp_timestep_helper(axs[subgroup_idx, 0], group_x_train, group_y_tox_train,
                                 dose_scenario.dose_labels, dose_scenario.toxicity_probs[subgroup_idx, :],
                                 x_test, y_tox_dist.mean[subgroup_idx, :], y_tox_dist.lower[subgroup_idx, :],
                                 y_tox_dist.upper[subgroup_idx, :], y_tox_acqui_func[subgroup_idx, :],
                                 dose_scenario.toxicity_threshold, selected_doses[subgroup_idx])
        _plot_gp_timestep_helper(axs[subgroup_idx, 1], group_x_train, group_y_eff_train,
                                 dose_scenario.dose_labels, dose_scenario.efficacy_probs[subgroup_idx, :],
                                 x_test, y_eff_dist.mean[subgroup_idx, :], y_eff_dist.lower[subgroup_idx, :],
                                 y_eff_dist.upper[subgroup_idx, :], y_eff_acqui_func[subgroup_idx, :],
                                 dose_scenario.efficacy_threshold, selected_doses[subgroup_idx])
        _plot_gp_timestep_utility(axs[subgroup_idx, 2], x_test, dose_scenario.dose_labels, util_func[subgroup_idx, :], 
                                 selected_doses[subgroup_idx])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def _plot_gp_trials(ax, rep_means, test_x, true_x, true_y, label):
    markevery_mask = np.isin(test_x, true_x)
    markevery = np.arange(len(test_x))[markevery_mask].tolist()

    sns.set()
    mean = np.mean(rep_means, axis=0)
    ci = 1.96 * np.std(rep_means, axis=0) / np.sqrt(rep_means.shape[0])
    ax.plot(test_x, mean, 'b-', markevery=markevery, marker='o', label=label)
    if true_x is not None and true_y is not None:
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
    ax.fill_between(test_x, (mean-ci), (mean+ci), alpha=0.5)
    ax.set_ylim([0, 1.1])
    ax.legend()

def plot_gp_trials(tox_means, eff_means, util_func, test_x,
                   dose_labels, tox_probs, eff_probs,
                   num_subgroups, results_dir):
    fig, axs = plt.subplots(num_subgroups, 3, figsize=(12, 8))
    for subgroup_idx in range(num_subgroups):
        axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
        axs[subgroup_idx, 2].set_title(f"Utility - Subgroup {subgroup_idx}")
        _plot_gp_trials(axs[subgroup_idx, 0], tox_means[:, subgroup_idx, :], test_x,
                             dose_labels, tox_probs[subgroup_idx, :],
                             'GP Predicted')
        _plot_gp_trials(axs[subgroup_idx, 1], eff_means[:, subgroup_idx, :], test_x,
                             dose_labels, eff_probs[subgroup_idx, :],
                             'GP Predicted')
        _plot_gp_trials(axs[subgroup_idx, 2], util_func[subgroup_idx, :], test_x,
                        None, None, 'Utility')

    fig.tight_layout()
    plt.savefig(f"{results_dir}/all_trials_plot.png")
    plt.close()

