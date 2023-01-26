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
from new_experiments import get_model_predictions, calculate_utility, select_final_dose


def offline_dose_finding():
    '''
    Offline dose finding procedure with multi-output GPs representing
    dose-toxicity and dose-efficacy relationships. Each output corresponds
    to a patient subgroup.
    '''
    # Filepath to read offline data from
    dose_scenario = DoseFindingScenarios.paper_example_6()
    filepath = "results/121_example"
    save_filepath = "results/122_example"
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)

    # Hyperparameters
    num_latents = 3
    num_epochs = 300
    learning_rate = 0.01
    use_gpu = False
    num_confidence_samples = 1000

    tox_lengthscale = 4.
    eff_lengthscale = 2.
    tox_mean_init = NormalDist().inv_cdf(dose_scenario.toxicity_threshold - 0.1)
    eff_mean_init = 0.

    
    patient_scenario = TrialPopulationScenarios.equal_population(2)
    dose_labels = dose_scenario.dose_labels
    num_tasks = patient_scenario.num_subgroups

    frame = pd.read_csv(f"{filepath}/timestep_metrics.csv")
    subgroup_indices = frame['subgroup_idx'].values
    selected_dose_indices = frame['selected_dose'].values
    selected_dose_vals = [dose_labels[dose_idx] for dose_idx in selected_dose_indices]
    tox_responses = frame['tox_outcome'].values
    eff_responses = frame['eff_outcome'].values

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

    np_x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    np_x_test = np.unique(np_x_test)
    np.sort(np_x_test)
    
    x_mask = np.isin(np_x_test, dose_labels)
    markevery = np.arange(len(np_x_test))[x_mask].tolist()

    # Train model
    tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, tox_lengthscale, tox_mean_init)
    tox_runner.train(x_train, y_tox_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=False)
    # Print model params
    for name, param in tox_runner.model.named_parameters():
        print(name, param.data)

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, eff_lengthscale, eff_mean_init)
    eff_runner.train(x_train, y_eff_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=False)
    # Print model params
    for name, param in eff_runner.model.named_parameters():
        print(name, param.data)

    # Get model predictions
    y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    # Select final dose
    final_selected_doses, final_utilities = select_final_dose(2, 5, 
                                                              dose_labels, x_test, y_tox_posteriors,
                                                              y_eff_posteriors,
                                                              dose_scenario.toxicity_threshold,
                                                              dose_scenario.efficacy_threshold,
                                                              dose_scenario.tox_weight,
                                                              dose_scenario.eff_weight, 0)
    # Calculate utilities
    util_func = np.empty((2, len(np_x_test)))
    for subgroup_idx in range(2):
        util_func[subgroup_idx, :] = calculate_utility(y_tox_posteriors.mean[subgroup_idx, :],
                                                       y_eff_posteriors.mean[subgroup_idx, :],
                                                       dose_scenario.toxicity_threshold,
                                                       dose_scenario.efficacy_threshold,
                                                       dose_scenario.tox_weight, dose_scenario.eff_weight)

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices, 2,
            np_x_test, y_tox_posteriors, y_eff_posteriors, util_func, final_selected_doses,
            markevery, x_mask, f"{save_filepath}/final_gp_plot",
            set_axis=True)
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, subgroup_indices, 2,
            np_x_test, y_tox_latents, y_eff_latents, util_func, final_selected_doses,
            markevery, x_mask, f"{save_filepath}/final_gp_latents_plot")


offline_dose_finding()