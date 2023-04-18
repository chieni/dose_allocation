import os
import numpy as np
import pandas as pd
import torch

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from dir_gp import DirichletRunner
from experiment_utils import PosteriorPrediction
from new_experiments_clean import select_final_dose, calculate_utility_thall
from plots import plot_gp


def offline_dose_finding(input_filepath, output_filepath, dose_scenario, patient_scenario, num_samples):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    # Hyperparameters
    learning_rate = 0.005
    num_epochs = 300
    num_confidence_samples = 1000
    cohort_size = 3
    final_beta_param = 0.
    tox_mean_init = 0
    tox_ls_init = 4
    eff_mean_init = 0
    eff_ls_init = 4

    dose_labels = dose_scenario.dose_labels
    num_doses = dose_scenario.num_doses

    num_subgroups = patient_scenario.num_subgroups
    num_tasks = patient_scenario.num_subgroups
    patients = patient_scenario.generate_samples(num_samples)
    
    frame = pd.read_csv(f"{input_filepath}/timestep_metrics.csv")
    subgroup_indices = frame['subgroup_idx'].values
    selected_dose_indices = frame['selected_dose'].values
    selected_dose_vals = [dose_labels[dose_idx] for dose_idx in selected_dose_indices]
    tox_responses = frame['tox_outcome'].values
    eff_responses = frame['eff_outcome'].values

    # Construct training data
    x_train = torch.tensor(selected_dose_vals, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_responses, dtype=torch.long)
    y_eff_train = torch.tensor(eff_responses, dtype=torch.long)
    task_indices = torch.LongTensor(subgroup_indices).reshape((x_train.shape[0], 1))

    # Construct test data
    x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    x_test = np.unique(x_test)
    np.sort(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    tox_runner = DirichletRunner(x_train, y_tox_train, task_indices, tox_mean_init, tox_ls_init)
    tox_runner.train(num_epochs, learning_rate)

    for name, param in tox_runner.model.named_parameters():
        print(name, param.data)

    eff_runner = DirichletRunner(x_train, y_eff_train, task_indices, eff_mean_init, eff_ls_init)
    eff_runner.train(num_epochs, learning_rate)

    for name, param in eff_runner.model.named_parameters():
        print(name, param.data)

    y_tox_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    y_tox_latents = PosteriorPrediction(num_subgroups, len(x_test))
    y_eff_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    y_eff_latents = PosteriorPrediction(num_subgroups, len(x_test))

    max_doses = np.zeros(num_subgroups)
    for subgroup_idx in range(num_subgroups):
        test_task_indices = torch.full((x_test.shape[0], 1), dtype=torch.long, fill_value=subgroup_idx)
        tox_posterior = tox_runner.predict(x_test, test_task_indices)
        tox_latent_lower, tox_latent_upper = tox_posterior.confidence_region()
        y_tox_latents.set_variables(subgroup_idx, tox_posterior.mean[1], tox_latent_lower[1], tox_latent_upper[1])

        tox_probs = tox_runner.get_posterior_estimate(tox_posterior)
        y_tox_posteriors.set_variables(subgroup_idx, tox_probs[1])

        eff_posterior = eff_runner.predict(x_test, test_task_indices)
        eff_latent_lower, eff_latent_upper = eff_posterior.confidence_region()
        y_eff_latents.set_variables(subgroup_idx, eff_posterior.mean[1], eff_latent_lower[1], eff_latent_upper[1])

        eff_probs = eff_runner.get_posterior_estimate(eff_posterior)
        y_eff_posteriors.set_variables(subgroup_idx, eff_probs[1])

        max_doses[subgroup_idx] = np.max(selected_dose_indices[subgroup_indices == subgroup_idx])


    final_selected_doses, final_utilities = select_final_dose(num_subgroups, num_doses, dose_labels, x_test, max_doses,
                                                  y_tox_posteriors, y_eff_posteriors,
                                                  dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
                                                  dose_scenario.p_param, final_beta_param)


    # Calculate utilities
    util_func = np.empty((2, len(x_test)))
    for subgroup_idx in range(2):
        util_func[subgroup_idx, :] = calculate_utility_thall(y_tox_posteriors.mean[subgroup_idx, :],
                                                             y_eff_posteriors.mean[subgroup_idx, :],
                                                             dose_scenario.toxicity_threshold,
                                                             dose_scenario.efficacy_threshold,
                                                             dose_scenario.p_param)

    x_mask = np.isin(x_test, dose_labels)
    markevery = np.arange(len(x_test))[x_mask].tolist()


    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            x_test, y_tox_posteriors, y_eff_posteriors, util_func, final_selected_doses,
            markevery, x_mask, f"{output_filepath}/final_gp_plot", dose_scenario.optimal_doses)
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            x_test, y_tox_latents, y_eff_latents, util_func, final_selected_doses,
            markevery, x_mask, f"{output_filepath}/final_gp_latent_plot", dose_scenario.optimal_doses)

dose_scenario = DoseFindingScenarios.paper_example_8()
patient_scenario = TrialPopulationScenarios.skewed_dual_population(0.5)
input_filepath = "results/1example"
output_filepath = "results/2example"
num_samples = 51

offline_dose_finding(input_filepath, output_filepath, dose_scenario, patient_scenario, num_samples)