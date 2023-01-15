import pandas as pd
import numpy as np
import torch

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from dose_finding_experiments import DoseFindingExperiment
from plots import plot_gp_trials


dose_scenario = DoseFindingScenarios.subgroups_example_1()
patient_scenario = TrialPopulationScenarios.equal_population(2)
experiment = DoseFindingExperiment(dose_scenario, patient_scenario)

x_true = dose_scenario.dose_labels.astype(np.float32)
x_test = np.concatenate([np.arange(x_true.min(), x_true.max(), 0.05, dtype=np.float32), x_true])
x_test = np.unique(x_test)
np.sort(x_test)
x_test = torch.tensor(x_test, dtype=torch.float32)
x_mask = np.isin(x_test, x_true)
markevery = np.arange(len(x_test))[x_mask].tolist()

filepath = "results/exp2"
num_trials = 100

dose_counts_list = []
metrics_list = []
grouped_metrics_list = []
final_dose_recs_list = []

tox_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
eff_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
util_vals = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))

for trial in range(num_trials):
    trial_path = f"{filepath}/trial{trial}"
    dose_counts = pd.read_csv(f"{trial_path}/dose_counts.csv")
    metrics = pd.read_csv(f"{trial_path}/timestep_metrics.csv")
    grouped_metrics = pd.read_csv(f"{trial_path}/overall_metrics.csv")
    final_dose_rec = pd.read_csv(f"{trial_path}/final_dose_rec.csv")

    dose_counts_list.append(dose_counts)
    metrics_list.append(metrics)
    grouped_metrics_list.append(grouped_metrics)
    final_dose_recs_list.append(final_dose_rec)

    tox_means[trial, :, :] = np.load(f"{trial_path}/tox_means.npy")
    eff_means[trial, :, :] = np.load(f"{trial_path}/eff_means.npy")
    util_vals[trial, :, :] = np.load(f"{trial_path}/util_vals.npy")
    
all_dose_recs = pd.concat(final_dose_recs_list)
all_dose_recs_grouped = all_dose_recs.groupby('subgroup_idx')['final_dose_rec'].value_counts()
all_dose_recs_grouped.to_csv(f"{filepath}/final_dose_recs.csv")

plot_gp_trials(tox_means, eff_means, util_vals, x_test,
                dose_scenario.dose_labels, dose_scenario.toxicity_probs,
                dose_scenario.efficacy_probs,
                patient_scenario.num_subgroups, markevery, filepath)
