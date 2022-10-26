import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from gp import ClassificationRunner

def plot_gp_results(train_x, train_y, test_x, posterior_probs, true_x, true_y):
    sns.set()
    plt.scatter(train_x.numpy(), train_y.numpy(), s=40, c='k', alpha=0.1, label='Training Data')
    plt.plot(test_x.numpy(), posterior_probs, 'b-', label='Predicted')
    plt.plot(true_x, true_y, 'g-', marker='o', label='True')
    plt.legend()
    plt.show()

def classification_example():
    train_x = torch.arange(0, 5, 0.2)
    train_y = torch.zeros(len(train_x))
    train_y[:int(len(train_x) / 2)] = 1.
    test_x = torch.arange(torch.min(train_x), torch.max(train_x), 0.1)

    runner = ClassificationRunner()
    model, likelihood = runner.train(train_x, train_y, training_iter=500)
    posterior_probs = runner.predict(model, likelihood, test_x)
    plot_gp_results(train_x, train_y, test_x, posterior_probs, [], [])

def dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    # dose_scenario.plot_true_curves()

    patient_scenario = TrialPopulationScenarios.homogenous_population()
    patients = patient_scenario.generate_samples(100)

    runner = ClassificationRunner()

    # Generate all data beforehand to test models (should be online data in 'real' examples)
    num_samples = 100
    arm_indices = np.arange(dose_scenario.num_doses, dtype=int)
    num_tiles = int(num_samples / dose_scenario.num_doses)
    tiled_arr = np.repeat(arm_indices, num_tiles)
    selected_arms = np.zeros(num_samples, dtype=int)
    selected_arms[num_samples - len(tiled_arr):] = tiled_arr
    selected_dose_levels = np.array([dose_scenario.dose_levels[arm_idx] for arm_idx in selected_arms])

    toxicity_data = dose_scenario.generate_toxicity_data(selected_arms)
    efficacy_data = dose_scenario.generate_efficacy_data(selected_arms)

    train_x = torch.tensor(selected_dose_levels.astype(np.float32))
    train_y = torch.tensor(toxicity_data.astype(np.float32))
    test_x = torch.arange(torch.min(train_x), torch.max(train_x), 0.1)
    # test_x = torch.tensor(dose_scenario.dose_levels.astype(np.float32))
    print(train_x)
    print(train_y)
    tox_runner = ClassificationRunner()
    tox_model, tox_likelihood = tox_runner.train(train_x, train_y, training_iter=200)
    posterior_probs = tox_runner.predict(tox_model, tox_likelihood, test_x)
    plot_gp_results(train_x, train_y, test_x, posterior_probs, dose_scenario.dose_levels, dose_scenario.toxicity_probs)

dose_example()