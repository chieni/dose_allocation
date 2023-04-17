import os
import argparse
import math
import pickle
from statistics import NormalDist

import numpy as np
import scipy.stats
import torch
import gpytorch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from new_gp import MultitaskClassificationRunner, MultitaskGPModel
from plots import plot_gp, plot_gp_timestep, plot_gp_trials
from experiment_utils import PosteriorPrediction
from new_experiments_clean import DoseExperimentMetrics, calculate_utility_thall, get_bernoulli_confidence_region


def get_model_predictions(runner, num_subgroups, x_test, num_confidence_samples, use_gpu):
    y_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    y_latents = PosteriorPrediction(num_subgroups, len(x_test))

    for subgroup_idx in range(num_subgroups):
        test_task_indices = torch.LongTensor(np.repeat(subgroup_idx, len(x_test)))
        post_latents, _ = runner.predict(x_test, test_task_indices, use_gpu)
        mean, lower, upper, variance = get_bernoulli_confidence_region(post_latents, runner.likelihood, num_confidence_samples)
        latent_lower, latent_upper = post_latents.confidence_region()
        y_posteriors.set_variables(subgroup_idx, mean.cpu().numpy(), lower.cpu().numpy(),
                                   upper.cpu().numpy(), variance.cpu().numpy())
        y_latents.set_variables(subgroup_idx, post_latents.mean.cpu().numpy(),
                                latent_lower.cpu().numpy(), latent_upper.cpu().numpy())
    return y_posteriors, y_latents

def select_dose_confidence_and_increasing(dose_labels, max_dose, tox_mean, tox_upper,
                                          tox_lower, tox_thre, beta_param, use_lcb=False):
    ## Select ideal dose for subgroup
    # Available doses are current max dose val + 3. 
    available_doses_mask = (dose_labels <= max_dose + 3.)

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)
    tox_lcb = tox_mean - (beta_param * (tox_mean - tox_lower))

    tox_acqui = tox_ucb
    if use_lcb:
        #tox_acqui = tox_lcb
        tox_acqui = tox_mean
    
    safe_doses_mask = tox_acqui <= tox_thre
    gt_threshold = np.where(tox_acqui > tox_thre)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    
    tox_intervals = tox_upper - tox_lower
    tox_intervals[~dose_set_mask] = -np.inf
    max_tox_interval = tox_intervals.max()

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose_idx = 0
    else:
        # Select largest safe dose
        selected_dose_idx = np.where(dose_set_mask == True)[0][-1]
        # If this dose has already been seen, instead select based on widest confidence interval
        if selected_dose_idx <= max_dose:
            selected_dose_idx = np.where(tox_intervals == max_tox_interval)[0][-1]
    
    return selected_dose_idx, tox_acqui, dose_set_mask


def select_dose(dose_labels, max_dose, tox_mean, tox_upper,
                tox_lower, eff_mean, eff_variance, beta_param,
                tox_thre, eff_thre, p_param,
                use_utility=False, use_lcb=False):
    ## Select ideal dose for subgroup
    available_doses_mask = (dose_labels <= max_dose + 3.)

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)
    tox_lcb = tox_mean - (beta_param * (tox_mean - tox_lower))
    tox_acqui = tox_ucb
    if use_lcb:
        tox_acqui = tox_lcb

    safe_doses_mask = tox_acqui <= tox_thre
    gt_threshold = np.where(tox_acqui > tox_thre)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    ## Select optimal dose using EI of efficacy posteriors
    tradeoff_param = 0.1
    eff_stdev = np.sqrt(eff_variance)
    eff_opt = eff_mean.max()
    improvement = eff_mean - eff_opt - tradeoff_param
    z_val = improvement / eff_stdev
    eff_ei = (improvement * scipy.stats.norm.cdf(z_val)) + (eff_stdev * scipy.stats.norm.pdf(z_val))
    eff_ei[eff_stdev == 0.] = 0.

    if not use_utility:
        dose_eff_ei = eff_ei
        dose_eff_ei[~dose_set_mask] = -np.inf
        max_eff_ei = dose_eff_ei.max()
        selected_dose_idx = np.where(dose_eff_ei == max_eff_ei)[0][-1]
    else:
        ## Select optimal dose using utility
        utilities = calculate_utility_thall(tox_mean, eff_mean, tox_thre, eff_thre,
                                            p_param)

        utilities[~dose_set_mask] = -np.inf
        max_utility = utilities.max()
        selected_dose_idx = np.where(utilities == max_utility)[0][-1]

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose_idx = 0

    return selected_dose_idx, tox_acqui, eff_ei

def select_final_dose(num_subgroups, dose_labels, max_doses,
                      tox_posteriors, eff_posteriors, 
                      tox_thre, eff_thre, final_beta_param,
                      p_param, use_thall=False):

    dose_rec = np.ones(num_subgroups) * len(dose_labels)
    final_utilities = np.empty((num_subgroups, len(dose_labels)))
        
    # Select dose with highest utility that is below toxicity threshold
    for subgroup_idx in range(num_subgroups):
        # Can only select dose that has been seen before
        max_dose = max_doses[subgroup_idx]
        available_doses_mask = (dose_labels <= max_dose + 3.)

        # Determine safe dose set on posterior means
        tox_mean = np.copy(tox_posteriors.mean[subgroup_idx, :])
        tox_upper = np.copy(tox_posteriors.upper[subgroup_idx, :])
        eff_mean = np.copy(eff_posteriors.mean[subgroup_idx, :])

        tox_conf_interval = tox_upper - tox_mean
        tox_ucb = tox_mean + (final_beta_param * tox_conf_interval)
        safe_doses_mask = tox_ucb <= tox_thre

        gt_threshold = np.where(tox_ucb > tox_thre)[0]
        if gt_threshold.size:
            first_idx_above_threshold = gt_threshold[0]
            safe_doses_mask[first_idx_above_threshold:] = False

        dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)

        # Calculate utilities
        thall_utilities = calculate_utility_thall(tox_mean, eff_mean,
                                                  tox_thre,
                                                  eff_thre,
                                                  p_param)

        eff_mean[~dose_set_mask] = -np.inf
        thall_utilities[~dose_set_mask] = -np.inf

        max_util_thall = thall_utilities.max()
        max_util_thall_idx = np.where(thall_utilities == max_util_thall)[0][-1]

        final_utilities[subgroup_idx, :] = thall_utilities
        if eff_mean[max_util_thall_idx] >= eff_thre:
            dose_rec[subgroup_idx] = max_util_thall_idx
        else:
            # Try second highest utility
            util_copy = np.copy(thall_utilities)
            util_copy.sort()
            second_max_util_thall = thall_utilities[-2]
            second_max_util_idx = np.where(thall_utilities == second_max_util_thall)[0][-1]
            if eff_mean[second_max_util_idx] >= eff_thre:
                dose_rec[subgroup_idx] = second_max_util_idx
    
    print(f"Final doses: {dose_rec}")
    return dose_rec, final_utilities
    

def online_dose_finding(filepath, dose_scenario, patient_scenario,
                        num_samples, num_latents, beta_param, learning_rate,
                        tox_lengthscale_init, eff_lengthscale_init,
                        tox_mean_init, eff_mean_init,
                        final_beta_param, sampling_timesteps, increase_beta_param,
                        use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu):
    plots_filepath = f"{filepath}/gp_plots"
    latent_plots_filepath = f"{filepath}/latent_gp_plots"
    if not os.path.exists(filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(plots_filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(latent_plots_filepath):
        os.makedirs(latent_plots_filepath)

    # Hyperparameters
    num_epochs = 300
    num_confidence_samples = 1000
    cohort_size = 3

    dose_labels = dose_scenario.dose_labels
    num_subgroups = patient_scenario.num_subgroups
    num_tasks = patient_scenario.num_subgroups
    patients = patient_scenario.generate_samples(num_samples)
    num_doses = len(dose_scenario.dose_labels)

    timestep = 0
    max_doses = np.repeat(min(dose_scenario.dose_labels), num_subgroups)

    # Initialize arrays
    selected_doses = []
    selected_dose_values = []
    tox_outcomes = []
    eff_outcomes = []

    # Initialize first cohort with lowest dose
    for subgroup_idx in patients[timestep: timestep + cohort_size]:
        selected_dose = 0
        selected_dose_val = dose_labels[selected_dose]
        tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
        eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

        selected_doses.append(selected_dose)
        selected_dose_values.append(selected_dose_val)
        tox_outcomes.append(tox_outcome)
        eff_outcomes.append(eff_outcome)
    
    # Construct test data (works for all models)
    np_x_test = dose_scenario.dose_labels.astype(np.float32)
    x_mask = np.isin(np_x_test, dose_labels)
    markevery = np.arange(len(np_x_test))[x_mask].tolist()
    timestep += cohort_size

    dose_set_mask = np.empty((num_subgroups, num_doses), dtype=np.bool)
    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        current_beta_param = np.float32(beta_param)

        # Construct training data
        task_indices = torch.LongTensor(patients[:timestep])
        x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
        y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
        y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)
        x_test = torch.tensor(np_x_test, dtype=torch.float32)

        # Train model
        tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, tox_lengthscale_init,
                                                   tox_mean_init)
        tox_runner.train(x_train, y_tox_train, task_indices,
                         num_epochs, learning_rate, use_gpu, set_lmc=True)

        eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, eff_lengthscale_init,
                                                   eff_mean_init)
        eff_runner.train(x_train, y_eff_train, task_indices,
                         num_epochs, learning_rate, use_gpu, set_lmc=True)

        # Get model predictions
        y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)
        y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)

        # Get dose selections
        selected_dose_by_subgroup = np.empty(num_subgroups, dtype=np.int32)
        tox_acqui_funcs = np.empty((num_subgroups, len(np_x_test)))
        eff_acqui_funcs = np.empty((num_subgroups, len(np_x_test)))
        util_func = np.empty((num_subgroups, len(np_x_test)))
        cohort_patients = patients[timestep: timestep + cohort_size]

        for subgroup_idx in range(patient_scenario.num_subgroups):
            if timestep <= sampling_timesteps:
                selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :], dose_set_mask[subgroup_idx, :] \
                    = select_dose_confidence_and_increasing(dose_labels, max_doses[subgroup_idx],
                                                            y_tox_posteriors.mean[subgroup_idx, :], 
                                                            y_tox_posteriors.upper[subgroup_idx, :],
                                                            y_tox_posteriors.lower[subgroup_idx, :],
                                                            dose_scenario.toxicity_threshold, current_beta_param,
                                                            use_lcb=use_lcb_init)
                eff_acqui_funcs[subgroup_idx, :] = 0.

            else:
                selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :],\
                eff_acqui_funcs[subgroup_idx, :] = \
                select_dose(dose_labels, max_doses[subgroup_idx],
                            y_tox_posteriors.mean[subgroup_idx, :],
                            y_tox_posteriors.upper[subgroup_idx, :],
                            y_tox_posteriors.lower[subgroup_idx, :],
                            y_eff_posteriors.mean[subgroup_idx, :],
                            y_eff_posteriors.variance[subgroup_idx, :],
                            current_beta_param, dose_scenario.toxicity_threshold,
                            dose_scenario.efficacy_threshold, dose_scenario.p_param,
                            use_utility=use_utility, use_lcb=use_lcb_exp)

            # Calculate utility
            util_func[subgroup_idx, :] = calculate_utility_thall(y_tox_posteriors.mean[subgroup_idx, :],
                                                        y_eff_posteriors.mean[subgroup_idx, :],
                                                        dose_scenario.toxicity_threshold,
                                                        dose_scenario.efficacy_threshold,
                                                        dose_scenario.p_param)

        print(f"Selected dose by subgroup: {selected_dose_by_subgroup}")

        # Assign dose to each patient in cohort, update outcomes.
        for subgroup_idx in cohort_patients:
            selected_dose = selected_dose_by_subgroup[subgroup_idx]
            selected_dose_val = dose_labels[selected_dose]
            tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
            eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

            selected_doses.append(selected_dose)
            selected_dose_values.append(selected_dose_val)
            tox_outcomes.append(tox_outcome)
            eff_outcomes.append(eff_outcome)

            # Update max doses
            if selected_dose_val > max_doses[subgroup_idx]:
                max_doses[subgroup_idx] = dose_labels[selected_dose]
        
        print(f"Selected doses: {selected_doses[timestep:]}")
        print(f"Max doses: {max_doses}")

        # Plot timestep
        plot_gp_timestep(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                         num_subgroups, np_x_test, y_tox_posteriors, y_eff_posteriors, 
                         tox_acqui_funcs, eff_acqui_funcs, util_func, selected_dose_by_subgroup,
                         markevery, x_mask, f"{plots_filepath}/timestep{timestep}", marker_val='')

        plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                patient_scenario.num_subgroups, np_x_test, y_tox_latents, y_eff_latents,
                util_func, selected_dose_by_subgroup, markevery, x_mask,
                f"{latent_plots_filepath}/timestep{timestep}", dose_scenario.optimal_doses, marker_val='')
        
        timestep += cohort_size

    # Train model final time
    # Construct training data
    task_indices = torch.LongTensor(patients)
    x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
    y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)
    x_test = torch.tensor(np_x_test, dtype=torch.float32)

    # Train model
    tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, tox_lengthscale_init,
                                               tox_mean_init)
    tox_runner.train(x_train, y_tox_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=set_lmc)

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                            dose_labels, eff_lengthscale_init,
                                            eff_mean_init)
    eff_runner.train(x_train, y_eff_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=set_lmc)
    
    # Get model predictions
    y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    
    
    y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)

    # Select final dose
    final_selected_doses, final_utilities = select_final_dose(num_subgroups, 
                                                              dose_labels, max_doses,
                                                              y_tox_posteriors,
                                                              y_eff_posteriors,
                                                              dose_scenario.toxicity_threshold,
                                                              dose_scenario.efficacy_threshold,
                                                              final_beta_param, 
                                                              dose_scenario.p_param,
                                                              use_thall)
    experiment_metrics = DoseExperimentMetrics(dose_scenario, patients, selected_doses,
                                               tox_outcomes, eff_outcomes, y_tox_posteriors.mean[:, x_mask],
                                               y_tox_posteriors.upper[:, x_mask], y_eff_posteriors.mean[:, x_mask],
                                               final_selected_doses, final_utilities)
    experiment_metrics.save_metrics(filepath)

    # Calculate utilities
    util_func = np.empty((num_subgroups, len(np_x_test)))
    for subgroup_idx in range(num_subgroups):
        util_func[subgroup_idx, :] = calculate_utility_thall(y_tox_posteriors.mean[subgroup_idx, :],
                                                    y_eff_posteriors.mean[subgroup_idx, :],
                                                    dose_scenario.toxicity_threshold,
                                                    dose_scenario.efficacy_threshold,
                                                    dose_scenario.p_param)

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            np_x_test, y_tox_posteriors, y_eff_posteriors, util_func, final_selected_doses,
            markevery, x_mask, f"{filepath}/final_gp_plot", dose_scenario.optimal_doses, marker_val='')
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            np_x_test, y_tox_latents, y_eff_latents, util_func, final_selected_doses,
            markevery, x_mask, f"{filepath}/final_gp_latents_plot", dose_scenario.optimal_doses, marker_val='')
    
    return experiment_metrics, y_tox_posteriors, y_eff_posteriors, util_func


def online_dose_finding_trials(results_dir, num_trials, dose_scenario, patient_scenario,
                               num_samples, num_latents, beta_param, learning_rate, 
                               tox_lengthscale_init, eff_lengthscale_init, 
                               tox_mean_init, eff_mean_init, final_beta_param,
                               sampling_timesteps, increase_beta_param, use_utility, use_lcb_init,
                               use_lcb_exp, set_lmc, use_thall, use_gpu):
    metrics = []
    x_true = dose_scenario.dose_labels.astype(np.float32)
    x_test = torch.tensor(x_true, dtype=torch.float32)

    tox_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    eff_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    util_vals = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    for trial in range(num_trials):
        print(f"Trial {trial}")
        filepath = f"{results_dir}/trial{trial}"
        trial_metrics, tox_posteriors, eff_posteriors, util_func = online_dose_finding(
            filepath, dose_scenario, patient_scenario, num_samples, num_latents, beta_param,
            learning_rate, tox_lengthscale_init, eff_lengthscale_init,
            tox_mean_init, eff_mean_init,
            final_beta_param, sampling_timesteps, increase_beta_param,
            use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
        metrics.append(trial_metrics)

        # for subgroup_idx in range(patient_scenario.num_subgroups):
        #     tox_means[trial, subgroup_idx, :] = tox_posteriors.mean[subgroup_idx, :]
        #     eff_means[trial, subgroup_idx, :] = eff_posteriors.mean[subgroup_idx, :]
        #     util_vals[trial, subgroup_idx, :] = util_func[subgroup_idx, :]

        # with open(f"{results_dir}/trial{trial}/tox_means.npy", 'wb') as f:
        #     np.save(f, tox_means[trial, :, :])
        # with open(f"{results_dir}/trial{trial}/eff_means.npy", 'wb') as f:
        #     np.save(f, eff_means[trial, :, :])
        # with open(f"{results_dir}/trial{trial}/util_vals.npy", 'wb') as f:
        #     np.save(f, util_vals[trial, :, :])
    
    DoseExperimentMetrics.save_merged_metrics(metrics, results_dir)
    x_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[x_mask].tolist()
    plot_gp_trials(tox_means, eff_means, util_vals, x_test,
                   dose_scenario.dose_labels, dose_scenario.toxicity_probs,
                   dose_scenario.efficacy_probs,
                   patient_scenario.num_subgroups, markevery, results_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="File path name")
    parser.add_argument("--scenario", type=int, help="Dose scenario")
    parser.add_argument("--beta_param", type=float, help="Beta param for toxicity confidence interval.")
    parser.add_argument("--num_samples", type=int, help="Number of samples.")
    parser.add_argument("--sampling_timesteps", type=int, help="Number of timesteps to run burn-in procedure.")
    parser.add_argument("--tox_lengthscale", type=float, help="Tox GP Kernel lengthscale.")
    parser.add_argument("--eff_lengthscale", type=float, help="Eff GP Kernel lengthscale")
    parser.add_argument("--tox_mean", type=float, help="Tox mean constant")
    parser.add_argument("--eff_mean", type=float, help="Eff mean constant")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_latents", type=int, help="Number of GP latents")
    parser.add_argument("--use_lcb_init", action="store_true", help="Use LCB for initial stage.")
    parser.add_argument("--use_lcb_exp", action="store_true", help="Use LCB for exploitation stage.")
    parser.add_argument("--set_lmc", action="store_true", help="Fix LMC coeffs for final model if true.")
    parser.add_argument("--use_thall", action="store_true", help="Use Thall utility.")
    parser.add_argument("--run_one", action="store_true", help="Run just one iteration")
    parser.add_argument("--group_ratio", type=float, help="Subgroup skew.")
    parser.add_argument("--num_trials", type=int, help="Number of trials")
    args = parser.parse_args()

    scenarios = {
        1: DoseFindingScenarios.continuous_subgroups_example_1(),
        2: DoseFindingScenarios.continuous_subgroups_example_2(),
        3: DoseFindingScenarios.continuous_subgroups_example_3(),
        4: DoseFindingScenarios.continuous_subgroups_example_4()
    }

    filepath = args.filepath
    scenario = args.scenario
    beta_param = args.beta_param
    num_samples = args.num_samples
    sampling_timesteps = args.sampling_timesteps
    tox_lengthscale = args.tox_lengthscale
    eff_lengthscale = args.eff_lengthscale
    tox_mean = args.tox_mean
    eff_mean = args.eff_mean
    learning_rate = args.learning_rate
    num_latents = args.num_latents
    use_lcb_init = args.use_lcb_init
    use_lcb_exp = args.use_lcb_exp
    set_lmc = args.set_lmc
    use_thall = args.use_thall
    run_one = args.run_one
    group_ratio = args.group_ratio
    num_trials = args.num_trials
    return filepath, scenarios[scenario], beta_param, num_samples, sampling_timesteps,\
           tox_lengthscale, eff_lengthscale, tox_mean, eff_mean, learning_rate, num_latents, use_lcb_init, use_lcb_exp, set_lmc, use_thall, run_one, group_ratio, num_trials


if __name__ == "__main__":
    filepath, dose_scenario, beta_param, num_samples, sampling_timesteps, tox_lengthscale_init, \
        eff_lengthscale_init, tox_mean_init, eff_mean_init, learning_rate, num_latents, use_lcb_init, use_lcb_exp, set_lmc, use_thall, run_one, group_ratio, num_trials = parse_args()

    increase_beta_param = False
    use_utility = False
    use_gpu = False
    final_beta_param = 0.

    patient_scenario = TrialPopulationScenarios.skewed_dual_population(group_ratio)
    
    if run_one:
        online_dose_finding(filepath, dose_scenario, patient_scenario,
                            num_samples, num_latents, beta_param, learning_rate,
                            tox_lengthscale_init, eff_lengthscale_init,
                            tox_mean_init, eff_mean_init,
                            final_beta_param, sampling_timesteps, increase_beta_param,
                            use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
    else:
        online_dose_finding_trials(filepath, num_trials, dose_scenario,
                                patient_scenario, num_samples, num_latents,
                                beta_param, learning_rate, 
                                tox_lengthscale_init, eff_lengthscale_init, 
                                tox_mean_init, eff_mean_init, final_beta_param,
                                sampling_timesteps, increase_beta_param, use_utility,
                                use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
