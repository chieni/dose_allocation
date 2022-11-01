import math
import numpy as np
import torch
import gpytorch
import seaborn as sns
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from gp import ClassificationRunner, MultitaskGPModel, MultitaskClassificationRunner

def plot_gp_results(ax, train_x, train_y, test_x, posterior_dist_mean, true_x, true_y, lower=None, upper=None):
    sns.set()
    ax.scatter(train_x, train_y, s=40, c='k', alpha=0.1, label='Training Data')
    ax.plot(test_x, posterior_dist_mean, 'b-', marker='o',label='GP Predicted')
    ax.plot(true_x, true_y, 'g-', marker='o', label='True')
    if lower is not None and upper is not None:
        ax.fill_between(test_x, lower, upper, alpha=0.5)
    ax.legend()


def get_bernoulli_confidence_region(posterior_latent_dist, likelihood_model, num_samples):
    sample_arr = np.empty((num_samples, posterior_latent_dist.mean.shape[0]))
    for sample_idx in range(num_samples):
        sample_arr[sample_idx, :] = likelihood_model(posterior_latent_dist.sample()).probs
    sample_means = sample_arr.mean(axis=0)
    sample_stds = sample_arr.std(axis=0)
    lower = sample_means - sample_stds
    upper = sample_means + sample_stds
    return lower, upper


def classification_example():
    train_x = torch.arange(0, 5, 0.2)
    train_y = torch.zeros(len(train_x))
    train_y[:int(len(train_x) / 2)] = 1.
    test_x = torch.arange(torch.min(train_x), torch.max(train_x), 0.1)

    runner = ClassificationRunner()
    model, likelihood = runner.train(train_x, train_y, training_iter=400)
    posterior_probs = runner.predict(model, likelihood, test_x)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_gp_results(ax, train_x, train_y, test_x, posterior_probs, [], [])
    plt.show()

def dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    # dose_scenario.plot_true_curves()

    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 5000
    training_iter = 100
    patients = patient_scenario.generate_samples(num_samples)

    # Generate all data beforehand to test models (should be online data in 'real' examples)
    arm_indices = np.arange(dose_scenario.num_doses, dtype=int)
    num_tiles = int(num_samples / dose_scenario.num_doses)
    tiled_arr = np.repeat(arm_indices, num_tiles)
    selected_arms = np.zeros(num_samples, dtype=int)
    selected_arms[num_samples - len(tiled_arr):] = tiled_arr
    selected_dose_levels = np.array([dose_scenario.dose_levels[arm_idx] for arm_idx in selected_arms])

    toxicity_data = dose_scenario.generate_toxicity_data(selected_arms)
    efficacy_data = dose_scenario.generate_efficacy_data(selected_arms)
    inducing_points = torch.tensor(dose_scenario.dose_levels.astype(np.float32))

    train_x = torch.tensor(selected_dose_levels.astype(np.float32))
    test_x = torch.arange(torch.min(train_x), torch.max(train_x) + 0.1, 0.1)
    #test_x = torch.tensor(dose_scenario.dose_levels.astype(np.float32))

    # Toxicity
    tox_y = torch.tensor(toxicity_data.astype(np.float32))
    tox_runner = ClassificationRunner(inducing_points)
    tox_runner.train(train_x, tox_y, training_iter=training_iter)
    posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x)
    tox_lower, tox_upper = get_bernoulli_confidence_region(posterior_latent_dist, tox_runner.likelihood, num_samples=1000)
    posterior_dist_mean = posterior_observed_dist.mean.detach().numpy()

    # Efficacy
    eff_y = torch.tensor(efficacy_data.astype(np.float32))
    eff_runner = ClassificationRunner(inducing_points)
    eff_runner.train(train_x, eff_y, training_iter=training_iter)
    eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x)
    eff_lower, eff_upper = get_bernoulli_confidence_region(eff_posterior_latent_dist, eff_runner.likelihood, num_samples=1000)
    eff_posterior_dist_mean = eff_posterior_observed_dist.mean.detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_gp_results(axs[0], train_x.numpy(), tox_y.numpy(), test_x.numpy(),
                    posterior_dist_mean, dose_scenario.dose_levels,
                    dose_scenario.toxicity_probs, tox_lower, tox_upper)
    plot_gp_results(axs[1], train_x.numpy(), eff_y.numpy(), test_x.numpy(),
                    eff_posterior_dist_mean, dose_scenario.dose_levels,
                    dose_scenario.efficacy_probs, eff_lower, eff_upper)
    plt.tight_layout()
    plt.show()


def multitask_dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    training_iter = 100
    patients = patient_scenario.generate_samples(num_samples)

    # Generate all data beforehand to test models (should be online data in 'real' examples)
    arm_indices = np.arange(dose_scenario.num_doses, dtype=int)
    num_tiles = int(num_samples / dose_scenario.num_doses)
    tiled_arr = np.repeat(arm_indices, num_tiles)
    selected_arms = np.zeros(num_samples, dtype=int)
    selected_arms[num_samples - len(tiled_arr):] = tiled_arr
    selected_dose_levels = np.array([dose_scenario.dose_levels[arm_idx] for arm_idx in selected_arms])

    toxicity_data = dose_scenario.generate_toxicity_data(selected_arms)
    efficacy_data = dose_scenario.generate_efficacy_data(selected_arms)
    inducing_points = torch.tensor(dose_scenario.dose_levels.astype(np.float32))

    train_x = torch.tensor(selected_dose_levels.astype(np.float32)) # 50 pts
    test_x = torch.arange(torch.min(train_x), torch.max(train_x) + 0.1, 0.1) # 23 pts
    #test_x = torch.tensor(dose_scenario.dose_levels.astype(np.float32))

    tox_y = torch.tensor(toxicity_data.astype(np.float32))
    eff_y = torch.tensor(efficacy_data.astype(np.float32))
    train_y = torch.stack([tox_y, eff_y], -1)

    runner = MultitaskClassificationRunner(num_latents=2, num_tasks=2, num_inducing_points=5)
    runner.train(train_x, train_y, training_iter=training_iter)
    posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
    posterior_dist_mean = posterior_observed_dist.mean.detach().numpy()
    #lower, upper = get_bernoulli_confidence_region(posterior_latent_dist, runner.likelihood, num_samples=1000)
    import pdb; pdb.set_trace()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_gp_results(axs[0], train_x.numpy(), tox_y.numpy(), test_x.numpy(),
                    posterior_dist_mean[:, 0], dose_scenario.dose_levels,
                    dose_scenario.toxicity_probs)
    plot_gp_results(axs[1], train_x.numpy(), eff_y.numpy(), test_x.numpy(),
                    posterior_dist_mean[:, 1], dose_scenario.dose_levels,
                    dose_scenario.efficacy_probs)    
    plt.tight_layout()
    plt.show()


def multitask_example():
    train_x = torch.linspace(0, 1, 100)

    train_y = torch.stack([
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.sin(train_x * (2 * math.pi)) + 2 * torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ], -1)

    num_latents = 3
    num_tasks = 4
    num_inducing_pts = 16
    model = MultitaskGPModel(num_latents, num_tasks, num_inducing_pts)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    num_epochs = 100
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    # Initialize plots
    fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    for task, ax in enumerate(axs):
        # Plot training data as black stars
        ax.plot(train_x.detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        # Predictive mean as blue line
        ax.plot(test_x.numpy(), mean[:, task].numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.show()
#multitask_example()
multitask_dose_example()