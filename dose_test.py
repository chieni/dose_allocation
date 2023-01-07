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
from dose_finding_experiments import DoseFindingExperiment, DoseExperimentSubgroupMetrics


dose_scenario = DoseFindingScenarios.subgroups_example_1()
patient_scenario = TrialPopulationScenarios.equal_population(2)
experiment = DoseFindingExperiment(dose_scenario, patient_scenario)

num_tasks = patient_scenario.num_subgroups
num_inducing_pts = dose_scenario.num_doses
num_subgroups = dose_scenario.num_subgroups

num_reps = 100
cohort_size = 3
num_samples = 51
num_epochs = 500
num_confidence_samples = 10000
num_latents = 2
learning_rate = 0.01
beta_param = 0.5
filepath = "results/74_example"

offline_data_frame = pd.read_csv("results/56_example/raw_metrics.csv")

patients = offline_data_frame['subgroup_idx'].values
selected_doses = offline_data_frame['selected_dose'].values
selected_dose_vals = [dose_scenario.dose_labels[dose_idx] for dose_idx in selected_doses]
train_x = torch.tensor(selected_dose_vals, dtype=torch.float32)
tox_train_y = torch.tensor(offline_data_frame['toxicity'].values, dtype=torch.float32)
eff_train_y = torch.tensor(offline_data_frame['efficacy'].values, dtype=torch.float32)

dose_labels = dose_scenario.dose_labels.astype(np.float32)
test_x = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
test_x = np.unique(test_x)
np.sort(test_x)
test_x = torch.tensor(test_x, dtype=torch.float32)

tox_runner, eff_runner, tox_dists, eff_dists \
    = experiment.run_separate_subgroup_gps(num_latents, num_tasks, num_inducing_pts, num_epochs, train_x,
                                            tox_train_y, eff_train_y, test_x, patients, num_subgroups,
                                            num_confidence_samples, learning_rate, False, None, None)

for name, param in tox_runner.model.named_parameters():
        print(name, param.data)
tox_lengthscale = np.squeeze(tox_runner.model.covar_module.base_kernel.kernels[0].lengthscale.detach().cpu().numpy())
tox_variance = np.squeeze(tox_runner.model.covar_module.base_kernel.kernels[1].variance.detach().cpu().numpy())

for name, param in eff_runner.model.named_parameters():
        print(name, param.data)
eff_lengthscale = np.squeeze(eff_runner.model.covar_module.base_kernel.kernels[0].lengthscale.detach().cpu().numpy())
eff_variance = np.squeeze(eff_runner.model.covar_module.base_kernel.kernels[1].variance.detach().cpu().numpy())
outputscale = np.squeeze(eff_runner.model.covar_module.outputscale.detach().cpu().numpy())

model_params_frame = pd.DataFrame({'tox_lengthscale': tox_lengthscale,
                                    'tox_variance': tox_variance,
                                    'eff_lengthscale': eff_lengthscale,
                                    'eff_variance': eff_variance,
                                    'output_scale': outputscale},
                                    index=np.arange(num_latents))

final_dose_error, final_utilities, final_dose_rec \
        = experiment.select_final_dose_subgroups_utility(dose_labels, test_x, tox_dists, eff_dists, beta_param)

utilities = [experiment.calculate_dose_utility(dose_scenario.get_toxicity_prob(arm_idx, group_idx), dose_scenario.get_efficacy_prob(arm_idx, group_idx))\
             for arm_idx, group_idx in zip(selected_doses, patients)]


if not os.path.exists(filepath):
    os.makedirs(filepath)


experiment_metrics = DoseExperimentSubgroupMetrics(num_samples, patients, num_subgroups, tox_train_y.numpy(), eff_train_y.numpy(), 
                                                    selected_doses, dose_scenario.optimal_doses, final_dose_error, utilities,
                                                    final_utilities, final_dose_rec, model_params_frame)

experiment_metrics.save_metrics(filepath)
experiment.plot_subgroup_gp_results(train_x.numpy(), tox_train_y.numpy(), eff_train_y.numpy(), patients, num_subgroups,
                                    test_x, tox_dists, eff_dists, filepath)