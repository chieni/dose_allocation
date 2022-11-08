import math
import numpy as np
import torch
import gpytorch
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from gp import ClassificationRunner, MultitaskGPModel, MultitaskClassificationRunner, MultitaskBernoulliLikelihood


def get_dose_data(dose_scenario, patient_scenario, num_samples):
    patients = patient_scenario.generate_samples(num_samples)

    # Generate all data beforehand to test models (should be online data in 'real' examples)
    arm_indices = np.arange(dose_scenario.num_doses, dtype=int)
    num_tiles = int(num_samples / dose_scenario.num_doses)
    tiled_arr = np.repeat(arm_indices, num_tiles)
    selected_arms = np.zeros(num_samples, dtype=int)
    selected_arms[num_samples - len(tiled_arr):] = tiled_arr
    selected_dose_labels = np.array([dose_scenario.dose_labels[arm_idx] for arm_idx in selected_arms])

    toxicity_data = dose_scenario.generate_toxicity_data(selected_arms)
    efficacy_data = dose_scenario.generate_efficacy_data(selected_arms)
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    train_x = torch.tensor(selected_dose_labels.astype(np.float32)) # 50 pts
    #test_x = torch.arange(torch.min(train_x), torch.max(train_x) + 0.1, 0.1) # 23 pts
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    tox_y = torch.tensor(toxicity_data.astype(np.float32))
    eff_y = torch.tensor(efficacy_data.astype(np.float32))
    train_y = torch.stack([tox_y, eff_y], -1)

    return patients, train_x, train_y, test_x

def select_final_dose(dose_scenario, patient_scenario, tox_mean, eff_mean):
    dose_error = np.zeros(patient_scenario.num_subgroups)
    dose_rec = dose_scenario.num_doses
    
    # Select dose with max estimate efficacy that is also below toxicity threshold
    dose_options = eff_mean.numpy() * (tox_mean.numpy() <= dose_scenario.toxicity_threshold)
    mtd_eff = np.max(dose_options)
    mtd_idx = np.argmax(dose_options)

    # If recommended dose is above eff threshold, assign this dose. Else assign no dose
    if mtd_eff >= dose_scenario.efficacy_threshold:
        dose_rec = mtd_idx

    for subgroup_idx in range(patient_scenario.num_subgroups):
        if dose_rec != dose_scenario.optimal_doses[subgroup_idx]:
            dose_error[subgroup_idx] = 1
    print(dose_rec)
    return dose_error

def plot_gp_results(ax, train_x, train_y, test_x, posterior_dist_mean, true_x, true_y, lower=None, upper=None):
    sns.set()
    ax.scatter(train_x, train_y, s=40, c='k', alpha=0.1, label='Training Data')
    ax.plot(test_x, posterior_dist_mean, 'b-', marker='o',label='GP Predicted')
    ax.plot(true_x, true_y, 'g-', marker='o', label='True')
    if lower is not None and upper is not None:
        ax.fill_between(test_x, lower, upper, alpha=0.5)
    ax.legend()

def plot_trial_gp_results(ax, rep_means, test_x, true_x, true_y):
    sns.set()
    mean = np.mean(rep_means, axis=0)
    ci = 1.96 * np.std(rep_means, axis=0) / np.sqrt(rep_means.shape[0])
    
    ax.plot(test_x, mean, 'b-', marker='o', label='GP Predicted')
    ax.plot(true_x, true_y, 'g-', marker='o', label='True')
    ax.fill_between(test_x, (mean-ci), (mean+ci), alpha=0.5)
    ax.legend()


def get_bernoulli_confidence_region(posterior_latent_dist, likelihood_model, num_samples):
    samples = posterior_latent_dist.sample_n(num_samples)
    likelihood_samples = likelihood_model(samples)
    lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
    upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
    mean = likelihood_samples.mean.mean(axis=0)
    return mean, lower, upper


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
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_epochs = 300
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    num_confidence_samples = 100

    patients, train_x, train_y, test_x = get_dose_data(dose_scenario, patient_scenario, num_samples)
    tox_runner = ClassificationRunner(inducing_points)
    tox_runner.train(train_x, train_y[:, 0], num_epochs=num_epochs)
    posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x)
    tox_mean, tox_lower, tox_upper = get_bernoulli_confidence_region(posterior_latent_dist, tox_runner.likelihood, num_confidence_samples)

    # Efficacy
    eff_runner = ClassificationRunner(inducing_points)
    eff_runner.train(train_x, train_y[:, 1], num_epochs=num_epochs)
    eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x)
    eff_mean, eff_lower, eff_upper = get_bernoulli_confidence_region(eff_posterior_latent_dist, eff_runner.likelihood, num_confidence_samples)

    dose_error = select_final_dose(dose_scenario, patient_scenario, tox_mean, eff_mean)

    print(dose_error)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_gp_results(axs[0], train_x.numpy(), train_y[:, 0].numpy(), test_x.numpy(),
                    tox_mean, dose_scenario.dose_labels,
                    dose_scenario.toxicity_probs, tox_lower, tox_upper)
    plot_gp_results(axs[1], train_x.numpy(), train_y[:, 1].numpy(), test_x.numpy(),
                    eff_mean, dose_scenario.dose_labels,
                    dose_scenario.efficacy_probs, eff_lower, eff_upper)
    plt.tight_layout()
    plt.show()

def dose_example_trials():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_inducing_pts = 5
    num_epochs = 300
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    num_confidence_samples = 100
    num_reps = 100

    dose_error = np.zeros(patient_scenario.num_subgroups)
    tox_means = np.empty((num_reps, num_inducing_pts))
    eff_means = np.empty((num_reps, num_inducing_pts))
    for rep in range(num_reps):
        print(f"Trial {rep}")
        patients, train_x, train_y, test_x = get_dose_data(dose_scenario, patient_scenario, num_samples)
        tox_runner = ClassificationRunner(inducing_points)
        tox_runner.train(train_x, train_y[:, 0], num_epochs=num_epochs)
        posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x)
        tox_mean, tox_lower, tox_upper = get_bernoulli_confidence_region(posterior_latent_dist, tox_runner.likelihood, num_confidence_samples)

        # Efficacy
        eff_runner = ClassificationRunner(inducing_points)
        eff_runner.train(train_x, train_y[:, 1], num_epochs=num_epochs)
        eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x)
        eff_mean, eff_lower, eff_upper = get_bernoulli_confidence_region(eff_posterior_latent_dist, eff_runner.likelihood, num_confidence_samples)

        dose_error += select_final_dose(dose_scenario, patient_scenario, tox_mean, eff_mean)
        tox_means[rep, :] = tox_mean
        eff_means[rep, :] = eff_mean
    
    print(dose_error / num_reps)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_trial_gp_results(axs[0], tox_means, test_x,
                          dose_scenario.dose_labels, dose_scenario.toxicity_probs)
    plot_trial_gp_results(axs[1], eff_means, test_x,
                          dose_scenario.dose_labels, dose_scenario.efficacy_probs)
    fig.tight_layout()
    plt.show()

def multitask_dose_example_trials():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_latents = 3
    num_tasks = 2
    num_inducing_pts = 5
    num_epochs = 300
    num_confidence_samples = 100
    num_reps = 100

    dose_error = np.zeros(patient_scenario.num_subgroups)
    tox_means = np.empty((num_reps, num_inducing_pts))
    eff_means = np.empty((num_reps, num_inducing_pts))
    for rep in range(num_reps):
        print(f"Trial {rep}")
        patients, train_x, train_y, test_x = get_dose_data(dose_scenario, patient_scenario, num_samples)
        runner = MultitaskClassificationRunner(num_latents, num_tasks, num_inducing_pts)
        runner.train(train_x, train_y, num_epochs)
        posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
        mean, lower, upper = get_bernoulli_confidence_region(posterior_latent_dist, runner.likelihood, num_confidence_samples)
        dose_error += select_final_dose(dose_scenario, patient_scenario, mean[:, 0], mean[:, 1])
        tox_means[rep, :] = mean[:, 0]
        eff_means[rep, :] = mean[:, 1]

    print(dose_error / num_reps)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_trial_gp_results(axs[0], tox_means, test_x,
                          dose_scenario.dose_labels, dose_scenario.toxicity_probs)
    plot_trial_gp_results(axs[1], eff_means, test_x,
                          dose_scenario.dose_labels, dose_scenario.efficacy_probs)
    fig.tight_layout()
    plt.show()


def multitask_dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_latents = 3
    num_tasks = 2
    num_inducing_pts = 5
    num_epochs = 200
    num_confidence_samples = 100

    patients, train_x, train_y, test_x = get_dose_data(dose_scenario, patient_scenario, num_samples)
    runner = MultitaskClassificationRunner(num_latents, num_tasks, num_inducing_pts)
    runner.train(train_x, train_y, num_epochs)
    posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
    mean, lower, upper = get_bernoulli_confidence_region(posterior_latent_dist, runner.likelihood, num_confidence_samples)
    dose_error = select_final_dose(dose_scenario, patient_scenario, mean[:, 0], mean[:, 1])
    print(dose_error)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")
    plot_gp_results(axs[0], train_x.numpy(), train_y[:, 0].numpy(), test_x.numpy(),
                    mean[:, 0], dose_scenario.dose_labels,
                    dose_scenario.toxicity_probs, lower[:, 0], upper[:, 0])
    plot_gp_results(axs[1], train_x.numpy(), train_y[:, 1].numpy(), test_x.numpy(),
                    mean[:, 1], dose_scenario.dose_labels,
                    dose_scenario.efficacy_probs, lower[:, 1], upper[:, 1])   

    fig.tight_layout()
    plt.show()


def multitask_classification_example():
    train_x = torch.linspace(0, 1, 100)

    train_y = (torch.stack([
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.sin(train_x * (2 * math.pi)) + 2 * torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ], -1) > 0) + 0.

    num_latents = 3
    num_tasks = 4
    num_inducing_pts = 16
    model = MultitaskGPModel(num_latents, num_tasks, num_inducing_pts)
    likelihood = MultitaskBernoulliLikelihood()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    num_epochs = 200
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        epochs_iter.set_postfix(loss=loss.item())
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
        prob_samples = predictions.mean
        mean = torch.mean(prob_samples, axis=0)
        lower = torch.quantile(prob_samples, 0.025, axis=0)
        upper = torch.quantile(prob_samples, 1 - 0.025, axis=0)

    for task, ax in enumerate(axs):
        # Plot training data as black stars
        ax.plot(train_x.detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        # Predictive mean as blue line
        ax.plot(test_x.numpy(), mean[:, task].numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.set_ylim([-0.2, 1.2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f'Task {task + 1}')
        
    fig.tight_layout()
    plt.show()


def online_dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_epochs = 300
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    num_confidence_samples = 100

    selected_doses = []
    selected_dose_values = []
    toxicity_responses = []
    efficacy_responses = []

    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    for timestep in range(num_samples):
        print(f"Timestep: {timestep}")
        if timestep == 0:
            selected_dose = 0
            selected_doses.append(selected_dose)
            selected_dose_values.append(dose_scenario.dose_labels[selected_dose])
            toxicity_responses.append(dose_scenario.sample_toxicity_event(selected_dose))
            efficacy_responses.append(dose_scenario.sample_efficacy_event(selected_dose))
        else:
            train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
            tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
            eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

            tox_runner = ClassificationRunner(inducing_points)
            tox_runner.train(train_x, tox_train_y, num_epochs=num_epochs)
            posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x)
            tox_mean, tox_lower, tox_upper = get_bernoulli_confidence_region(posterior_latent_dist, tox_runner.likelihood, num_confidence_samples)

            # Efficacy
            eff_runner = ClassificationRunner(inducing_points)
            eff_runner.train(train_x, eff_train_y, num_epochs=num_epochs)
            eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x)
            eff_mean, eff_lower, eff_upper = get_bernoulli_confidence_region(eff_posterior_latent_dist, eff_runner.likelihood, num_confidence_samples)

            # Select dose
            safe_dose_set = tox_mean <= dose_scenario.toxicity_threshold
            selected_dose = np.argmax(eff_mean * safe_dose_set)
            print(selected_dose)

            # Get responses
            selected_doses.append(selected_dose)
            selected_dose_values.append(dose_scenario.dose_labels[selected_dose])
            toxicity_responses.append(dose_scenario.sample_toxicity_event(selected_dose))
            efficacy_responses.append(dose_scenario.sample_efficacy_event(selected_dose))
    
    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

    print(np.array(np.unique(selected_doses, return_counts=True)).T)
    print(tox_train_y)
    print(eff_train_y)

    dose_error = select_final_dose(dose_scenario, patient_scenario, tox_mean, eff_mean)
    print(dose_error)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")

    plot_gp_results(axs[0], train_x.numpy(), tox_train_y.numpy(), test_x.numpy(),
                    tox_mean, dose_scenario.dose_labels,
                    dose_scenario.toxicity_probs, tox_lower, tox_upper)
    plot_gp_results(axs[1], train_x.numpy(), eff_train_y.numpy(), test_x.numpy(),
                    eff_mean, dose_scenario.dose_labels,
                    dose_scenario.efficacy_probs, eff_lower, eff_upper)
    plt.tight_layout()
    plt.show()

def online_multitask_dose_example():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 50
    num_epochs = 300
    num_confidence_samples = 100

    num_latents = 3
    num_tasks = 2
    num_inducing_pts = 5

    selected_doses = []
    selected_dose_values = []
    toxicity_responses = []
    efficacy_responses = []

    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    for timestep in range(num_samples):
        print(f"Timestep: {timestep}")
        if timestep == 0:
            selected_dose = 0
            selected_doses.append(selected_dose)
            selected_dose_values.append(dose_scenario.dose_labels[selected_dose])
            toxicity_responses.append(dose_scenario.sample_toxicity_event(selected_dose))
            efficacy_responses.append(dose_scenario.sample_efficacy_event(selected_dose))
        else:
            train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
            tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
            eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)
            train_y = torch.stack([tox_train_y, eff_train_y], -1)

            runner = MultitaskClassificationRunner(num_latents, num_tasks, num_inducing_pts)
            runner.train(train_x, train_y, num_epochs)
            posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
            mean, lower, upper = get_bernoulli_confidence_region(posterior_latent_dist, runner.likelihood, num_confidence_samples)
        
            # Select dose
            safe_dose_set = mean[:, 0] <= dose_scenario.toxicity_threshold
            selected_dose = np.argmax(mean[:, 1] * safe_dose_set)
            print(selected_dose)

            # Get responses
            selected_doses.append(selected_dose)
            selected_dose_values.append(dose_scenario.dose_labels[selected_dose])
            toxicity_responses.append(dose_scenario.sample_toxicity_event(selected_dose))
            efficacy_responses.append(dose_scenario.sample_efficacy_event(selected_dose))
    
    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

    print(np.array(np.unique(selected_doses, return_counts=True)).T)
    print(tox_train_y)
    print(eff_train_y)

    dose_error = select_final_dose(dose_scenario, patient_scenario, mean[:, 0], mean[:, 1])
    print(dose_error)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Toxicity")
    axs[1].set_title("Efficacy")

    plot_gp_results(axs[0], train_x.numpy(), tox_train_y.numpy(), test_x.numpy(),
                    mean[:, 0], dose_scenario.dose_labels,
                    dose_scenario.toxicity_probs, lower[:, 0], upper[:, 0])
    plot_gp_results(axs[1], train_x.numpy(), eff_train_y.numpy(), test_x.numpy(),
                    mean[:, 1], dose_scenario.dose_labels,
                    dose_scenario.efficacy_probs, lower[:, 1], upper[:, 1])
    plt.tight_layout()
    plt.show()

#dose_example()
#multitask_dose_example_trials()
#multitask_classification_example()
#multitask_dose_example()
#dose_example_trials()
#online_dose_example()
online_multitask_dose_example()