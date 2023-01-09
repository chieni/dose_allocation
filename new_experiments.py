import os
import argparse
import math
import pickle

import numpy as np
import scipy.stats
import torch
import gpytorch
import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from new_gp import MultitaskClassificationRunner, MultitaskGPModel
from plots import plot_gp, plot_gp_timestep


class PosteriorPrediction:
    def __init__(self, num_subgroups, x_size):
        self.mean = np.empty((num_subgroups, x_size))
        self.lower = np.empty((num_subgroups, x_size))
        self.upper = np.empty((num_subgroups, x_size))
        self.variance = np.empty((num_subgroups, x_size))

    def set_variables(self, subgroup_idx, mean, lower, upper, variance=None):
        self.mean[subgroup_idx, :] = mean
        self.lower[subgroup_idx, :] = lower
        self.upper[subgroup_idx, :] = upper
        if variance is not None:
            self.variance[subgroup_idx, :] = variance


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
    filepath = "results/80_example"
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
    num_epochs = 500
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
            # Select ideal dose for subgroup
            x_mask = np.isin(x_test, dose_labels)
            
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

            # Select optimal dose using EI of efficacy posteriors
            # TODO: I'm not sure that this approach for EI for binary data makes sense. So
            # use ucb as placeholder
            eff_mean = y_eff_posteriors.mean[subgroup_idx, :]
            eff_upper = y_eff_posteriors.upper[subgroup_idx, :]
            eff_conf_interval = eff_upper - eff_mean
            eff_ucb = eff_mean + (beta_param * eff_conf_interval)
            eff_acqui_funcs[subgroup_idx, :] = eff_ucb
            dose_eff_ucb = eff_ucb[x_mask]
            dose_eff_ucb[~dose_set_mask] = -np.inf

            max_eff_ucb = dose_eff_ucb.max()
            selected_dose = np.where(dose_eff_ucb == max_eff_ucb)[0][-1]

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

    # TODO: Select final dose

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            x_test, y_tox_posteriors, y_eff_posteriors, f"{filepath}/final_gp_plot")


online_dose_finding()