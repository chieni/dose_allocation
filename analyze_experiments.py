import pandas as pd
import numpy as np
import torch

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from dose_finding_experiments import DoseFindingExperiment


dose_scenario = DoseFindingScenarios.subgroups_example_1()
patient_scenario = TrialPopulationScenarios.equal_population(2)
experiment = DoseFindingExperiment(dose_scenario, patient_scenario)

filepath = "results/exp12"
num_trials = 89
metrics_frames = []
dose_rec_frames = []
model_params_frames = []

for trial in range(num_trials):
    metrics_frame = pd.read_csv(f"{filepath}/trial{trial}/metrics.csv")
    metrics_frames.append(metrics_frame)

    dose_rec_frame = pd.read_csv(f"{filepath}/trial{trial}/final_dose_rec.csv")
    dose_rec_frames.append(dose_rec_frame)

    model_params_frame = pd.read_csv(f"{filepath}/trial{trial}/final_model_params.csv")
    model_params_frames.append(model_params_frame)

frame = pd.concat(metrics_frames)
grouped_frame = frame.groupby(frame.index)
mean_frame = grouped_frame.mean()
var_frame = grouped_frame.var()
mean_frame = mean_frame[['toxicity', 'efficacy', 'utility', 'dose_error', 'final_dose_error']]
var_frame = var_frame[['toxicity', 'efficacy', 'utility', 'dose_error', 'final_dose_error']]
print(mean_frame)
print(var_frame)
mean_frame.to_csv(f"{filepath}/final_metric_means.csv")
var_frame.to_csv(f"{filepath}/final_metric_var.csv")

dose_recs = pd.concat(dose_rec_frames)
dose_recs_grouped = dose_recs.groupby('subgroup_idx').value_counts().reset_index()
dose_recs_grouped = dose_recs_grouped.rename(columns={0: 'count'})
print(dose_recs_grouped)
dose_recs_grouped.to_csv(f"{filepath}/final_dose_recs.csv")

model_params_trials = pd.concat(model_params_frames)
model_params_grouped = model_params_trials.groupby(model_params_trials.index).mean()
print(model_params_grouped)
model_params_grouped.to_csv(f"{filepath}/trials_model_params.csv")


true_x = dose_scenario.dose_labels.astype(np.float32)
test_x = np.concatenate([np.arange(true_x.min(), true_x.max(), 0.05, dtype=np.float32), true_x])
test_x = np.unique(test_x)
np.sort(test_x)
test_x = torch.tensor(test_x, dtype=torch.float32)

tox_means = np.load(f"{filepath}/trial{num_trials}/tox_means.npy")
eff_means = np.load(f"{filepath}/trial{num_trials}/eff_means.npy")

experiment.plot_subgroup_trial_gp_results(tox_means, eff_means, test_x, patient_scenario.num_subgroups, filepath)