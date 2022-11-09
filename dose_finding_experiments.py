import math
import numpy as np
import torch
import gpytorch
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from gp import ClassificationRunner, MultitaskGPModel, MultitaskClassificationRunner, MultitaskBernoulliLikelihood


class PosteriorDist:
    def __init__(self, x_axis, mean, lower=None, upper=None):
        self.x_axis = x_axis
        self.mean = mean
        self.lower = lower
        self.upper = upper


class DoseFindingExperiment:
    def __init__(self, dose_scenario, patient_scenario):
        self.dose_scenario = dose_scenario
        self.patient_scenario = patient_scenario

    def get_offline_data(self, num_samples):
        patients = self.patient_scenario.generate_samples(num_samples)

        # Generate all data beforehand to test models (should be online data in 'real' examples)
        arm_indices = np.arange(self.dose_scenario.num_doses, dtype=int)
        num_tiles = int(num_samples / self.dose_scenario.num_doses)
        tiled_arr = np.repeat(arm_indices, num_tiles)
        selected_arms = np.zeros(num_samples, dtype=int)
        selected_arms[num_samples - len(tiled_arr):] = tiled_arr
        selected_dose_labels = np.array([self.dose_scenario.dose_labels[arm_idx] for arm_idx in selected_arms])

        toxicity_data = self.dose_scenario.generate_toxicity_data(selected_arms)
        efficacy_data = self.dose_scenario.generate_efficacy_data(selected_arms)
        inducing_points = torch.tensor(self.dose_scenario.dose_labels.astype(np.float32))

        train_x = torch.tensor(selected_dose_labels.astype(np.float32)) # 50 pts
        #test_x = torch.arange(torch.min(train_x), torch.max(train_x) + 0.1, 0.1) # 23 pts
        test_x = torch.tensor(self.dose_scenario.dose_labels.astype(np.float32))

        tox_y = torch.tensor(toxicity_data.astype(np.float32))
        eff_y = torch.tensor(efficacy_data.astype(np.float32))
        train_y = torch.stack([tox_y, eff_y], -1)

        return patients, train_x, train_y, test_x

    def select_dose(self, tox_mean, eff_mean):
        safe_dose_set = tox_mean.numpy() <= self.dose_scenario.toxicity_threshold
        selected_dose = np.argmax(eff_mean.numpy() * safe_dose_set)
        return selected_dose

    def select_final_dose(self, tox_mean, eff_mean):
        dose_error = np.zeros(self.patient_scenario.num_subgroups)
        dose_rec = self.dose_scenario.num_doses
        
        # Select dose with max estimate efficacy that is also below toxicity threshold
        dose_options = eff_mean.numpy() * (tox_mean.numpy() <= self.dose_scenario.toxicity_threshold)
        mtd_eff = np.max(dose_options)
        mtd_idx = np.argmax(dose_options)

        # If recommended dose is above eff threshold, assign this dose. Else assign no dose
        if mtd_eff >= self.dose_scenario.efficacy_threshold:
            dose_rec = mtd_idx

        for subgroup_idx in range(self.patient_scenario.num_subgroups):
            if dose_rec != self.dose_scenario.optimal_doses[subgroup_idx]:
                dose_error[subgroup_idx] = 1
        print(dose_rec, dose_error)
        return dose_error

    def plot_gp_results(self, train_x, tox_train_y, eff_train_y, tox_dist, eff_dist):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_title("Toxicity")
        axs[1].set_title("Efficacy")

        self._plot_gp_results_helper(axs[0], train_x, tox_train_y, tox_dist, self.dose_scenario.dose_labels, self.dose_scenario.toxicity_probs)
        self._plot_gp_results_helper(axs[1], train_x, eff_train_y, eff_dist, self.dose_scenario.dose_labels, self.dose_scenario.efficacy_probs)
        plt.tight_layout()
        plt.show()

    def _plot_gp_results_helper(self, ax, train_x, train_y, posterior_dist, true_x, true_y):
        sns.set()
        ax.scatter(train_x, train_y, s=40, c='k', alpha=0.1, label='Training Data')
        ax.plot(posterior_dist.x_axis, posterior_dist.mean, 'b-', marker='o',label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        if posterior_dist.lower is not None and posterior_dist.upper is not None:
            ax.fill_between(posterior_dist.x_axis, posterior_dist.lower, posterior_dist.upper, alpha=0.5)
        ax.legend()

    def plot_trial_gp_results(self, tox_means, eff_means, test_x):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_title("Toxicity")
        axs[1].set_title("Efficacy")
        self._plot_trial_gp_results_helper(axs[0], tox_means, test_x,
                            self.dose_scenario.dose_labels, self.dose_scenario.toxicity_probs)
        self._plot_trial_gp_results_helper(axs[1], eff_means, test_x,
                            self.dose_scenario.dose_labels, self.dose_scenario.efficacy_probs)
        fig.tight_layout()
        plt.show()

    def _plot_trial_gp_results_helper(self, ax, rep_means, test_x, true_x, true_y):
        sns.set()
        mean = np.mean(rep_means, axis=0)
        ci = 1.96 * np.std(rep_means, axis=0) / np.sqrt(rep_means.shape[0])
        
        ax.plot(test_x, mean, 'b-', marker='o', label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        ax.fill_between(test_x, (mean-ci), (mean+ci), alpha=0.5)
        ax.legend()

    def run_separate_gps(self, inducing_points, num_epochs, train_x, tox_train_y, eff_train_y, test_x, num_confidence_samples):
        tox_runner = ClassificationRunner(inducing_points)
        tox_runner.train(train_x, tox_train_y, num_epochs=num_epochs)
        posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x)
        tox_dist = self.get_bernoulli_confidence_region(test_x, posterior_latent_dist, tox_runner.likelihood, num_confidence_samples)

        eff_runner = ClassificationRunner(inducing_points)
        eff_runner.train(train_x, eff_train_y, num_epochs=num_epochs)
        eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x)
        eff_dist = self.get_bernoulli_confidence_region(test_x, eff_posterior_latent_dist, eff_runner.likelihood, num_confidence_samples)
        return tox_dist, eff_dist

    def run_multitask_gp(self, num_epochs, num_latents, num_tasks, num_inducing_pts, train_x, train_y, test_x, num_confidence_samples):
        runner = MultitaskClassificationRunner(num_latents, num_tasks, num_inducing_pts)
        runner.train(train_x, train_y, num_epochs)
        posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
        post_dist = self.get_bernoulli_confidence_region(test_x, posterior_latent_dist, runner.likelihood, num_confidence_samples)
        tox_dist = PosteriorDist(post_dist.x_axis, post_dist.mean[:, 0], post_dist.lower[:, 0], post_dist.upper[:, 0])
        eff_dist = PosteriorDist(post_dist.x_axis, post_dist.mean[:, 1], post_dist.lower[:, 1], post_dist.upper[:, 1])
        return tox_dist, eff_dist

    def get_bernoulli_confidence_region(self, test_x, posterior_latent_dist, likelihood_model, num_samples):
        samples = posterior_latent_dist.sample_n(num_samples)
        likelihood_samples = likelihood_model(samples)
        lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
        upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
        mean = likelihood_samples.mean.mean(axis=0)
        return PosteriorDist(test_x, mean, lower, upper)


def dose_example(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    
    patients, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
    tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                          train_y[:, 0], train_y[:, 1], test_x,
                                          num_confidence_samples)

    dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment.plot_gp_results(train_x.numpy(), train_y[:, 0].numpy(), train_y[:, 1].numpy(), tox_dist, eff_dist)

def dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, num_reps):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    dose_error = np.zeros(patient_scenario.num_subgroups)
    tox_means = np.empty((num_reps, inducing_points.shape[0]))
    eff_means = np.empty((num_reps, inducing_points.shape[0]))

    for rep in range(num_reps):
        print(f"Trial {rep}")
        patients, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
        tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                         train_y[:, 0], train_y[:, 1], test_x,
                                                         num_confidence_samples)
        dose_error += experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
        tox_means[rep, :] = tox_dist.mean
        eff_means[rep, :] = eff_dist.mean
    
    print(dose_error / num_reps)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)

def multitask_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs,
                           num_confidence_samples, num_latents, num_tasks, num_inducing_pts):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    patients, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
    tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                     train_x, train_y, test_x, num_confidence_samples)

    dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment.plot_gp_results(train_x.numpy(), train_y[:, 0].numpy(),
                               train_y[:, 1].numpy(), tox_dist, eff_dist)

def multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_reps, num_latents, num_tasks,
                                  num_inducing_pts):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    dose_error = np.zeros(patient_scenario.num_subgroups)
    tox_means = np.empty((num_reps, num_inducing_pts))
    eff_means = np.empty((num_reps, num_inducing_pts))

    for rep in range(num_reps):
        print(f"Trial {rep}")
        patients, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
        tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                         train_x, train_y, test_x, num_confidence_samples)
        dose_error += experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
        tox_means[rep, :] = tox_dist.mean
        eff_means[rep, :] = eff_dist.mean

    print(dose_error / num_reps)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)

def online_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, cohort_size):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    max_dose = 0
    timestep = 0

    selected_dose = 0
    selected_doses = [selected_dose for item in range(cohort_size)]
    selected_dose_values = [dose_scenario.dose_labels[dose] for dose in selected_doses]
    toxicity_responses = [dose_scenario.sample_toxicity_event(dose) for dose in selected_doses]
    efficacy_responses = [dose_scenario.sample_efficacy_event(dose) for dose in selected_doses]

    timestep += cohort_size
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
        tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
        eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

        tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                         tox_train_y, eff_train_y, test_x,
                                                         num_confidence_samples)

        selected_dose = experiment.select_dose(tox_dist.mean, eff_dist.mean)
        # if selected_dose > max_dose + 1:
        #     selected_dose = max_dose + 1
        #     max_dose = selected_dose
        print(f"Selected dose: {selected_dose}")

        selected_doses += [selected_dose for item in range(cohort_size)]
        selected_dose_values += [dose_scenario.dose_labels[dose] for dose in selected_doses]
        toxicity_responses += [dose_scenario.sample_toxicity_event(dose) for dose in selected_doses]
        efficacy_responses += [dose_scenario.sample_efficacy_event(dose) for dose in selected_doses]

        timestep += cohort_size

    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

    tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                     tox_train_y, eff_train_y, test_x,
                                                     num_confidence_samples)

    print(np.array(np.unique(selected_doses, return_counts=True)).T)
    dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment.plot_gp_results(train_x.numpy(), tox_train_y.numpy(),
                               eff_train_y.numpy(), tox_dist, eff_dist)


def online_multitask_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, cohort_size, num_latents,
                                  num_tasks, num_inducing_pts):
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    max_dose = 0
    timestep = 0

    selected_dose = 0
    selected_doses = [selected_dose for item in range(cohort_size)]
    selected_dose_values = [dose_scenario.dose_labels[dose] for dose in selected_doses]
    toxicity_responses = [dose_scenario.sample_toxicity_event(dose) for dose in selected_doses]
    efficacy_responses = [dose_scenario.sample_efficacy_event(dose) for dose in selected_doses]

    timestep += cohort_size
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
        tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
        eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)
        train_y = torch.stack([tox_train_y, eff_train_y], -1)

        tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                         train_x, train_y, test_x, num_confidence_samples)

        selected_dose = experiment.select_dose(tox_dist.mean, eff_dist.mean)
        if selected_dose > max_dose + 1:
            selected_dose = max_dose + 1
            max_dose = selected_dose
        print(f"Selected dose: {selected_dose}")

        # Get responses
        selected_doses += [selected_dose for item in range(cohort_size)]
        selected_dose_values += [dose_scenario.dose_labels[dose] for dose in selected_doses]
        toxicity_responses += [dose_scenario.sample_toxicity_event(dose) for dose in selected_doses]
        efficacy_responses += [dose_scenario.sample_efficacy_event(dose) for dose in selected_doses]

        timestep += cohort_size
    
    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)
    train_y = torch.stack([tox_train_y, eff_train_y], -1)

    tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                     train_x, train_y, test_x, num_confidence_samples)

    print(np.array(np.unique(selected_doses, return_counts=True)).T)
    dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment.plot_gp_results(train_x.numpy(), tox_train_y.numpy(),
                             eff_train_y.numpy(), tox_dist, eff_dist)


def main():
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_samples = 24
    num_epochs = 300
    num_confidence_samples = 10000

    num_latents = 3
    num_tasks = 2
    num_inducing_pts = 5

    #dose_example(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples)
    # multitask_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples,
    #                        num_latents, num_tasks, num_inducing_pts)

    num_reps = 10
    #dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, num_reps)
    # multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                               num_confidence_samples, num_reps, num_latents, num_tasks, num_inducing_pts)

    cohort_size = 3
    #online_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, cohort_size)
    online_multitask_dose_example(dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, cohort_size, num_latents,
                                  num_tasks, num_inducing_pts)


main()