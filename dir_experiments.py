import numpy as np
import pandas as pd
import torch

from data_generation import DoseFindingScenarios
from dir_gp import DirichletRunner
from experiment_utils import PosteriorPrediction

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


def offline_dose_finding(dose_scenario, patient_scenario, num_samples):
    # Hyperparameters
    learning_rate = 0.01
    num_epochs = 300
    num_confidence_samples = 1000
    cohort_size = 3

    dose_labels = dose_scenario.dose_labels
    num_doses = dose_scenario.num_doses

    num_subgroups = patient_scenario.num_subgroups
    num_tasks = patient_scenario.num_subgroups
    patients = patient_scenario.generate_samples(num_samples)
    
    filepath = "results/1example"
    frame = pd.read_csv(f"{filepath}/timestep_metrics.csv")
    subgroup_indices = frame['subgroup_idx'].values
    selected_dose_indices = frame['selected_dose'].values
    selected_dose_vals = [dose_labels[dose_idx] for dose_idx in selected_dose_indices]
    tox_responses = frame['tox_outcome'].values
    eff_responses = frame['eff_outcome'].values

    # Construct training data
    x_train = torch.tensor(selected_dose_vals, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_responses, dtype=torch.float32)
    y_eff_train = torch.tensor(eff_responses, dtype=torch.float32)
    task_indices = torch.LongTensor(subgroup_indices).reshape((x_train.shape[0], 1))

    # Construct test data
    x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    x_test = np.unique(x_test)
    np.sort(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    tox_runner = DirichletRunner(x_train, y_tox_train, task_indices)
    tox_runner.train(x_train, y_tox_train, num_epochs, learning_rate=learning_rate)

    eff_runner = DirichletRunner(x_train, y_eff_train, task_indices)
    eff_runner.train(x_train, y_eff_train, num_epochs, learning_rate=learning_rate)

    y_tox_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    y_eff_posteriors = PosteriorPrediction(num_subgroups, len(x_test))
    for subgroup_idx in range(num_subgroups):
        test_task_indices = torch.full((x_test.shape[0], 1), dtype=torch.long, fill_value=subgroup_idx)
        tox_posterior = tox_runner.predict(x_test, test_task_indices)
        tox_probs = tox_runner.get_posterior_estimate(tox_posterior)

        eff_posterior = eff_runner.predict(x_test, test_task_indices)
        eff_probs = eff_runner.get_posterior_estimate(eff_posterior)





dose_scenario = DoseFindingScenarios.paper_example_8()