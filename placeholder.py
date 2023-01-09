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


def offline_dose_finding():
    '''
    Offline dose finding procedure with multi-output GPs representing
    dose-toxicity and dose-efficacy relationships. Each output corresponds
    to a patient subgroup.
    '''
    # Filepath to read offline data from
    filepath = "results/74_example"

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

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, lengthscale_prior=lengthscale_prior,
                                               outputscale_prior=outputscale_prior)
    eff_runner.train(x_train, y_eff_train, task_indices,
                     num_epochs, learning_rate, use_gpu)
    
    # Print model params
    for name, param in tox_runner.model.named_parameters():
        print(name, param.data)

    for name, param in eff_runner.model.named_parameters():
        print(name, param.data)
    
    # Get model predictions
    y_tox_posterior = PosteriorPrediction(patient_scenario.num_subgroups, len(x_test))
    y_tox_latent = PosteriorPrediction(patient_scenario.num_subgroups, len(x_test))

    y_eff_posterior = PosteriorPrediction(patient_scenario.num_subgroups, len(x_test))
    y_eff_latent = PosteriorPrediction(patient_scenario.num_subgroups, len(x_test))

    for subgroup_idx in range(patient_scenario.num_subgroups):
        test_task_indices = torch.LongTensor(np.repeat(subgroup_idx, len(x_test)))

        post_latents, _ = tox_runner.predict(x_test, test_task_indices, use_gpu)
        mean, lower, upper, variance = get_bernoulli_confidence_region(post_latents, tox_runner.likelihood, num_confidence_samples)
        latent_lower, latent_upper = post_latents.confidence_region()

        y_tox_posterior.set_variables(subgroup_idx, mean, lower, upper, variance)
        y_tox_latent.set_variables(subgroup_idx, post_latents.mean, latent_lower, latent_upper)
    
        post_latents, _ = eff_runner.predict(x_test, test_task_indices, use_gpu)
        mean, lower, upper, variance = get_bernoulli_confidence_region(post_latents, eff_runner.likelihood, num_confidence_samples)
        latent_lower, latent_upper = post_latents.confidence_region()

        y_eff_posterior.set_variables(subgroup_idx, mean, lower, upper, variance)
        y_eff_latent.set_variables(subgroup_idx, post_latents.mean, latent_lower, latent_upper)

    
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices,
            patient_scenario.num_subgroups, x_test, y_tox_posterior, y_eff_posterior,
            f"{filepath}/gp_plot")

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices,
            patient_scenario.num_subgroups, x_test, y_tox_latent, y_eff_latent,
            f"{filepath}/gp_latents_plot")
