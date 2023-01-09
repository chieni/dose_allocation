import os
import argparse
import math
import pickle

import numpy as np
import scipy.stats
import torch
import gpytorch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from new_gp import MultitaskClassificationRunner, MultitaskGPModel
from plots import plot_gp, plot_gp_timestep
from experiment_utils import PosteriorPrediction


class DoseExperimentMetrics:
    def __init__(self, dose_scenario, subgroup_indices, selected_doses, tox_outcomes, eff_outcomes,
                 tox_predicted, eff_predicted, final_selected_doses, final_utilities):
        self.num_subgroups = final_selected_doses.shape[0]
        self.subgroup_indices = subgroup_indices
        self.selected_doses = selected_doses
        self.tox_outcomes = tox_outcomes
        self.eff_outcomes = eff_outcomes
        self.tox_predicted = tox_predicted
        self.eff_predicted = eff_predicted
        self.final_selected_doses = final_selected_doses
        self.optimal_doses = dose_scenario.optimal_doses
        self.final_utilities = final_utilities

        # Calculate dose error
        optimal_dose_per_sample = np.array([self.optimal_doses[subgroup_idx] for subgroup_idx in subgroup_indices])
        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)

        # Calculate final dose error
        final_dose_error = (final_selected_doses != self.optimal_doses).astype(np.float32)

        # Calculate utility
        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        utilities = calculate_utility(selected_tox_probs, selected_eff_probs,
                                      dose_scenario.toxicity_threshold,
                                      dose_scenario.efficacy_threshold,
                                      dose_scenario.tox_weight,
                                      dose_scenario.eff_weight)

        self.metrics_frame = pd.DataFrame({
            'subgroup_idx': subgroup_indices,
            'tox_outcome': tox_outcomes,
            'eff_outcome': eff_outcomes,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utility': utilities
        })

        grouped_metrics_frame = self.metrics_frame.groupby(['subgroup_idx']).mean()
        total_frame = pd.DataFrame(self.metrics_frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        grouped_metrics_frame = pd.concat([grouped_metrics_frame, total_frame])
        grouped_metrics_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / self.num_subgroups]])
        self.grouped_metrics_frame = grouped_metrics_frame

        self.final_dose_rec = pd.DataFrame({'final_dose_rec': final_selected_doses})
        self.final_dose_rec['subgroup_idx'] = self.final_dose_rec.index

        self.subgroup_predictions = []
        for subgroup_idx in range(self.num_subgroups):
            predicted_frame = pd.DataFrame({
                'tox_predicted': self.tox_predicted[subgroup_idx, :],
                'eff_predicted': self.eff_predicted[subgroup_idx, :],
                'final_utilities': self.final_utilities[subgroup_idx, :]
            })
            self.subgroup_predictions.append(predicted_frame)
    
    def save_metrics(self, filepath):
        self.metrics_frame.to_csv(f"{filepath}/timestep_metrics.csv")
        overall_metrics_frame = self.grouped_metrics_frame[['tox_outcome', 'eff_outcome', 'utility', 'dose_error', 'final_dose_error']]
        overall_metrics_frame.to_csv(f"{filepath}/overall_metrics.csv")
        
        print(self.metrics_frame)
        print(overall_metrics_frame)
        for subgroup_idx in range(self.num_subgroups):
            print(f"Subgroup: {subgroup_idx}")
            # mask = self.subgroup_indices == subgroup_idx
            # dose_selections = np.array(np.unique(self.selected_doses[mask], return_counts=True)).T
            # with open(f"{filepath}/{subgroup_idx}_dose_selections.npy", 'wb') as f:
            #     np.save(f, dose_selections)
            
            # print(f"Dose allocations: {dose_selections}")
            self.subgroup_predictions[subgroup_idx].to_csv(f"{filepath}/{subgroup_idx}_predictions.csv")
            print(f"Predictions: {self.subgroup_predictions[subgroup_idx]}")

        self.final_dose_rec.to_csv(f"{filepath}/final_dose_rec.csv")

def calculate_utility(tox_means, eff_means, tox_thre, eff_thre, tox_weight, eff_weight):
    tox_term = (tox_means - tox_thre) ** 2
    eff_term = (eff_means - eff_thre) ** 2
    tox_term[tox_means > tox_thre] = 0.
    eff_term[eff_means < eff_thre] = 0.
    return (tox_weight * tox_term) + (eff_weight * eff_term)

def calculate_utility_thall(tox_probs, eff_probs, tox_thre, eff_thre, p_param):
    tox_term = (tox_probs / tox_thre) ** p_param
    eff_term = ((1. - eff_probs) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities
 
def get_bernoulli_confidence_region(posterior_latent_dist, likelihood_model, num_samples):
    samples = posterior_latent_dist.sample_n(num_samples)
    likelihood_samples = likelihood_model(samples)
    lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
    upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
    mean = likelihood_samples.mean.mean(axis=0)
    variance = likelihood_samples.mean.var(axis=0)
    return mean, lower, upper, variance

def get_model_predictions(runner, num_subgroups, x_test, num_confidence_samples, use_gpu):
    y_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    y_latents = PosteriorPrediction(num_subgroups, len(x_test))

    for subgroup_idx in range(num_subgroups):
        test_task_indices = torch.LongTensor(np.repeat(subgroup_idx, len(x_test)))

        post_latents, _ = runner.predict(x_test, test_task_indices, use_gpu)
        mean, lower, upper, variance = get_bernoulli_confidence_region(post_latents, runner.likelihood, num_confidence_samples)
        latent_lower, latent_upper = post_latents.confidence_region()

        y_posteriors.set_variables(subgroup_idx, mean, lower, upper, variance)
        y_latents.set_variables(subgroup_idx, post_latents.mean, latent_lower, latent_upper)
    return y_posteriors, y_latents

def select_final_dose(num_subgroups, num_doses, dose_labels, x_test,
                      tox_posteriors, eff_posteriors, 
                      tox_thre, eff_thre, tox_weight, eff_weight, beta_param):
    x_mask = np.isin(x_test, dose_labels)
    dose_rec = np.ones(num_subgroups) * num_doses
    final_utilities = np.empty((num_subgroups, num_doses))
        
    # Select dose with highest utility that is below toxicity threshold
    for subgroup_idx in range(num_subgroups):
        # Find UCB for toxicity posteriors to determine safe set
        tox_mean = tox_posteriors.mean[subgroup_idx, :]
        tox_upper = tox_posteriors.upper[subgroup_idx, :]
        tox_conf_interval = tox_upper - tox_mean
        tox_ucb = tox_mean + (beta_param * tox_conf_interval)

        dose_set_mask = tox_ucb[x_mask] <= tox_thre
        print(f"Dose set: {dose_set_mask}")

        # Calculate utilities
        tox_mean = tox_mean[x_mask]
        eff_mean = eff_posteriors.mean[subgroup_idx, :][x_mask]
        utilities = calculate_utility(tox_mean, eff_mean, tox_thre, eff_thre,
                                      tox_weight, eff_weight)
        utilities[~dose_set_mask] = -np.inf
        final_utilities[subgroup_idx, :] = utilities
        print(f"Final utilities: {utilities}")
        max_utility = utilities.max()
        best_dose_idx = np.where(utilities == max_utility)[0][-1]
        best_dose_val = eff_mean[best_dose_idx]
        print(best_dose_idx, best_dose_val)

        # If recommended dose is above eff threshold assign this dose, else assign no dose.
        if best_dose_val >= eff_thre: 
            dose_rec[subgroup_idx] = best_dose_idx
    
    print(f"Final doses: {dose_rec}")
    return dose_rec, final_utilities
        

def offline_dose_finding():
    '''
    Offline dose finding procedure with multi-output GPs representing
    dose-toxicity and dose-efficacy relationships. Each output corresponds
    to a patient subgroup.
    '''
    # Filepath to read offline data from
    filepath = "results/74_example"
    save_filepath = "results/75_example"
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)

    # Hyperparameters
    num_latents = 2
    num_epochs = 500
    learning_rate = 0.01
    use_gpu = False
    num_confidence_samples = 1000

    # lengthscale_prior = gpytorch.priors.LogNormalPrior(0.4, 0.5)
    # outputscale_prior = gpytorch.priors.LogNormalPrior(-0.25, 0.5)
    lengthscale_prior = None
    outputscale_prior = None

    dose_scenario = DoseFindingScenarios.subgroups_example_1()
    patient_scenario = TrialPopulationScenarios.equal_population(2)
    dose_labels = dose_scenario.dose_labels
    num_tasks = patient_scenario.num_subgroups

    frame = pd.read_csv(f"{filepath}/raw_metrics.csv")
    subgroup_indices = frame['subgroup_idx'].values
    selected_dose_indices = frame['selected_dose'].values
    selected_dose_vals = [dose_labels[dose_idx] for dose_idx in selected_dose_indices]
    tox_responses = frame['toxicity'].values
    eff_responses = frame['efficacy'].values

    # Construct training data
    task_indices = torch.LongTensor(subgroup_indices)
    x_train = torch.tensor(selected_dose_vals, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_responses, dtype=torch.float32)
    y_eff_train = torch.tensor(eff_responses, dtype=torch.float32)

    # Construct test data
    x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    x_test = np.unique(x_test)
    np.sort(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    # Train model
    tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, lengthscale_prior=lengthscale_prior,
                                               outputscale_prior=outputscale_prior)
    tox_runner.train(x_train, y_tox_train, task_indices,
                     num_epochs, learning_rate, use_gpu)
    # Print model params
    for name, param in tox_runner.model.named_parameters():
        print(name, param.data)

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, lengthscale_prior=lengthscale_prior,
                                               outputscale_prior=outputscale_prior)
    eff_runner.train(x_train, y_eff_train, task_indices,
                     num_epochs, learning_rate, use_gpu)
    # Print model params
    for name, param in eff_runner.model.named_parameters():
        print(name, param.data)

    # Get model predictions
    y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices,
            patient_scenario.num_subgroups, x_test, y_tox_posteriors, y_eff_posteriors,
            f"{save_filepath}/gp_plot")

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices,
            patient_scenario.num_subgroups, x_test, y_tox_latents, y_eff_latents,
            f"{save_filepath}/gp_latents_plot")

def online_dose_finding():
    # Filepath to save data to
    filepath = "results/87_example"
    plots_filepath = f"{filepath}/gp_plots"
    latent_plots_filepath = f"{filepath}/latent_gp_plots"
    if not os.path.exists(filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(plots_filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(latent_plots_filepath):
        os.makedirs(latent_plots_filepath)

    # Hyperparameters
    beta_param = 0.5
    num_latents = 2
    num_epochs = 300
    learning_rate = 0.01
    use_gpu = False
    num_confidence_samples = 1000
    num_samples = 51

    lengthscale_prior = gpytorch.priors.LogNormalPrior(0.4, 0.5)
    outputscale_prior = gpytorch.priors.LogNormalPrior(-0.25, 0.5)
    # lengthscale_prior = None
    # outputscale_prior = None

    cohort_size = 3

    dose_scenario = DoseFindingScenarios.subgroups_example_1()
    patient_scenario = TrialPopulationScenarios.equal_population(2)

    dose_labels = dose_scenario.dose_labels
    num_subgroups = patient_scenario.num_subgroups
    num_tasks = patient_scenario.num_subgroups
    patients = patient_scenario.generate_samples(num_samples)
    num_doses = dose_scenario.num_doses

    timestep = 0
    max_doses = np.zeros(num_subgroups)

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
    x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    x_test = np.unique(x_test)
    np.sort(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    x_mask = np.isin(x_test, dose_labels)
    timestep += cohort_size

    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        # Train model

        # Construct training data
        task_indices = torch.LongTensor(patients[:timestep])
        x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
        y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
        y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)

        print(f"task_indices: {task_indices}")
        print(f"x_train: {x_train}")
        print(f"y_tox_train: {y_tox_train}")
        print(f"y_eff_train: {y_eff_train}")

        # Train model
        tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, lengthscale_prior=lengthscale_prior,
                                                   outputscale_prior=outputscale_prior)
        tox_runner.train(x_train, y_tox_train, task_indices,
                         num_epochs, learning_rate, use_gpu)

        eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, lengthscale_prior=lengthscale_prior,
                                                   outputscale_prior=outputscale_prior)
        eff_runner.train(x_train, y_eff_train, task_indices,
                         num_epochs, learning_rate, use_gpu)
        
        for name, param in tox_runner.model.named_parameters():
                print(name, param.data)
        for name, param in eff_runner.model.named_parameters():
                print(name, param.data)

        # Get model predictions
        y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)
        y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)

        # Get dose selections
        selected_dose_by_subgroup = np.empty(num_subgroups, dtype=np.int32)
        tox_acqui_funcs = np.empty((num_subgroups, len(x_test)))
        eff_acqui_funcs = np.empty((num_subgroups, len(x_test)))

        for subgroup_idx in range(patient_scenario.num_subgroups):
            ## Select ideal dose for subgroup
            # Available doses are current max dose idx + 1
            available_dose_indices = np.arange(max_doses[subgroup_idx] + 2)
            available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
            print(f"Available doses: {available_doses_mask}")

            # Find UCB for toxicity posteriors to determine safe set
            tox_mean = y_tox_posteriors.mean[subgroup_idx, :]
            tox_upper = y_tox_posteriors.upper[subgroup_idx, :]
            tox_conf_interval = tox_upper - tox_mean
            tox_ucb = tox_mean + (beta_param * tox_conf_interval)
            tox_acqui_funcs[subgroup_idx, :] = tox_ucb
            safe_doses_mask = tox_ucb[x_mask] <= dose_scenario.toxicity_threshold
            print(f"Safe doses: {safe_doses_mask}")

            dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
            print(f"Dose set: {safe_doses_mask}")

            ## Select optimal dose using EI of efficacy posteriors
            # eff_upper = y_eff_posteriors.upper[subgroup_idx, :]
            # eff_conf_interval = eff_upper - eff_mean
            # eff_ucb = eff_mean + (beta_param * eff_conf_interval)
            # eff_acqui_funcs[subgroup_idx, :] = eff_ucb
            # dose_eff_ucb = eff_ucb[x_mask]
            # dose_eff_ucb[~dose_set_mask] = -np.inf

            # max_eff_ucb = dose_eff_ucb.max()
            # selected_dose = np.where(dose_eff_ucb == max_eff_ucb)[0][-1]

            tradeoff_param = 0.01
            eff_mean = y_eff_posteriors.mean[subgroup_idx, :]
            eff_stdev = np.sqrt(y_eff_posteriors.variance[subgroup_idx, :])
            eff_opt = eff_mean.max()
            improvement = eff_mean - eff_opt - tradeoff_param
            z_val = improvement / eff_stdev

            eff_ei = (improvement * scipy.stats.norm.cdf(z_val)) + (eff_stdev * scipy.stats.norm.pdf(z_val))
            eff_ei[eff_stdev == 0.] = 0.
            eff_acqui_funcs[subgroup_idx, :] = eff_ei

            dose_eff_ei = eff_ei[x_mask]
            dose_eff_ei[~dose_set_mask] = -np.inf
            max_eff_ei = dose_eff_ei.max()
            selected_dose = np.where(dose_eff_ei == max_eff_ei)[0][-1]

            # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
            if dose_set_mask.sum() == 0:
                selected_dose = 0
            
            selected_dose_by_subgroup[subgroup_idx] = selected_dose
        print(f"Selected dose by subgroup: {selected_dose_by_subgroup}")

        # Assign dose to each patient in cohort, update outcomes.
        cohort_patients = patients[timestep: timestep + cohort_size]
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
            if selected_dose > max_doses[subgroup_idx]:
                max_doses[subgroup_idx] = selected_dose
        
        print(f"Selected doses: {selected_doses[timestep:]}")
        print(f"Max doses: {max_doses}")

        # Plot timestep
        plot_gp_timestep(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                         num_subgroups, x_test, y_tox_posteriors, y_eff_posteriors, 
                         tox_acqui_funcs, eff_acqui_funcs, selected_dose_by_subgroup,
                         f"{plots_filepath}/timestep{timestep}")

        plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                patient_scenario.num_subgroups, x_test, y_tox_latents, y_eff_latents,
                f"{latent_plots_filepath}/timestep{timestep}")
        
        timestep += cohort_size

    # Train model final time
    # Construct training data
    task_indices = torch.LongTensor(patients)
    x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
    y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)

    # Train model
    tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, lengthscale_prior=lengthscale_prior,
                                               outputscale_prior=outputscale_prior)
    tox_runner.train(x_train, y_tox_train, task_indices,
                     num_epochs, learning_rate, use_gpu)

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                            dose_labels, lengthscale_prior=lengthscale_prior,
                                            outputscale_prior=outputscale_prior)
    eff_runner.train(x_train, y_eff_train, task_indices,
                    num_epochs, learning_rate, use_gpu)
    
    # Get model predictions
    y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)

    # Select final dose
    final_selected_doses, final_utilities = select_final_dose(num_subgroups, num_doses, 
                                                              dose_labels, x_test, y_tox_posteriors,
                                                              y_eff_posteriors,
                                                              dose_scenario.toxicity_threshold,
                                                              dose_scenario.efficacy_threshold,
                                                              dose_scenario.tox_weight,
                                                              dose_scenario.eff_weight, beta_param)
    print(f"Final selected doses: {final_selected_doses}")

    experiment_metrics = DoseExperimentMetrics(dose_scenario, patients, selected_doses,
                                               tox_outcomes, eff_outcomes, y_tox_posteriors.mean[:, x_mask], y_eff_posteriors.mean[:, x_mask], final_selected_doses, final_utilities)
    experiment_metrics.save_metrics(filepath)
    
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            x_test, y_tox_posteriors, y_eff_posteriors, f"{filepath}/final_gp_plot")
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            x_test, y_tox_latents, y_eff_latents, f"{filepath}/final_gp_latents_plot")


online_dose_finding()