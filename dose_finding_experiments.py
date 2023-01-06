import os
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
from gp import ClassificationRunner, MultitaskGPModel, MultitaskClassificationRunner, MultitaskBernoulliLikelihood, \
               MultitaskSubgroupClassificationRunner
from helpers import get_ucb


class PosteriorDist:
    def __init__(self, x_axis, samples, mean, variance, lower=None, upper=None, upper_width=None):
        self.x_axis = x_axis
        self.samples = samples
        self.mean = mean
        self.variance = variance
        self.lower = lower
        self.upper = upper
        self.upper_width = upper_width


class DoseExperimentMetrics:
    def __init__(self, num_samples, total_toxicity, total_efficacy,
                 selected_doses, optimal_doses, final_dose_error):
        self.num_samples = num_samples
        self.total_toxicity = total_toxicity
        self.total_efficacy = total_efficacy
        self.selected_doses = selected_doses

        self.toxicity_per_person = total_toxicity / num_samples
        self.efficacy_per_person = total_efficacy / num_samples
        self.dose_selections = np.array(np.unique(selected_doses, return_counts=True)).T
        self.dose_error_per_person = self.calculate_dose_error(selected_doses, optimal_doses, patients=None) / num_samples

        self.final_dose_error = final_dose_error
    
    @classmethod
    def print_merged_metrics(cls, metrics_list):
        toxicity_per_person = np.mean([metric.toxicity_per_person for metric in metrics_list])
        efficacy_per_person = np.mean([metric.efficacy_per_person for metric in metrics_list])
        dose_error_per_person = np.mean([metric.dose_error_per_person for metric in metrics_list])
        final_dose_error = np.mean([metric.final_dose_error for metric in metrics_list])
        metrics_frame = pd.DataFrame({
            'toxicity per person': [toxicity_per_person],
            'efficacy per person': [efficacy_per_person],
            'dose error per person': [dose_error_per_person],
            'final dose error': final_dose_error
        })
        print(metrics_frame)
        

    def calculate_dose_error(self, selected_doses, optimal_doses, patients):
        return np.sum(selected_doses != optimal_doses[0])
    
    def print_metrics(self):
        metrics_frame = pd.DataFrame({
            'toxicity per person': [self.toxicity_per_person],
            'efficacy per person': [self.efficacy_per_person],
            'dose error per person': [self.dose_error_per_person],
            'final dose error': self.final_dose_error
        })

        print(metrics_frame)
        print("Dose allocations")
        print(self.dose_selections)

class DoseExperimentSubgroupMetrics:
    def __init__(self, num_samples, patients, num_subgroups, toxicities, efficacies,
                 selected_doses, optimal_doses, final_dose_error, utilities, final_utilities,
                 final_dose_rec, model_params):
        self.num_samples = num_samples
        self.patients = patients
        self.num_subgroups = num_subgroups
        optimal_dose_per_sample = np.array([optimal_doses[subgroup_idx] for subgroup_idx in patients])
        self.selected_doses = np.array(selected_doses)
        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)

        self.final_utilities = pd.DataFrame(final_utilities, columns=np.arange(final_utilities.shape[1]))
        self.final_dose_rec = pd.DataFrame({'final_dose_rec': final_dose_rec})
        self.final_dose_rec['subgroup_idx'] = self.final_dose_rec.index

        self.model_params = model_params

        self.frame = pd.DataFrame({
            'subgroup_idx': patients,
            'toxicity': toxicities,
            'efficacy': efficacies,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utility': utilities
        })

        groups_frame = self.frame.groupby(['subgroup_idx']).mean()
        total_frame = pd.DataFrame(self.frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        groups_frame = pd.concat([groups_frame, total_frame])
        groups_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / self.num_subgroups]])
        self.groups_frame = groups_frame

    
    @classmethod
    def print_merged_metrics(cls, metrics_list, filepath):
        frame = pd.concat([df.groups_frame for df in metrics_list])
        grouped_frame = frame.groupby(frame.index)
        mean_frame = grouped_frame.mean()
        var_frame = grouped_frame.var()
        mean_frame = mean_frame[['toxicity', 'efficacy', 'utility', 'dose_error', 'final_dose_error']]
        var_frame = var_frame[['toxicity', 'efficacy', 'utility', 'dose_error', 'final_dose_error']]
        print(mean_frame)
        print(var_frame)
        mean_frame.to_csv(f"{filepath}/final_metric_means.csv")
        var_frame.to_csv(f"{filepath}/final_metric_var.csv")

        dose_recs = pd.concat([df.final_dose_rec for df in metrics_list])
        dose_recs_grouped = dose_recs.groupby('subgroup_idx').value_counts().reset_index()
        dose_recs_grouped = dose_recs_grouped.rename(columns={0: 'count'})
        print(dose_recs_grouped)
        dose_recs_grouped.to_csv(f"{filepath}/final_dose_recs.csv")

        model_params_trials = pd.concat([df.model_params for df in metrics_list])
        model_params_grouped = model_params_trials.groupby(model_params_trials.index).mean()
        print(model_params_grouped)
        model_params_grouped.to_csv(f"{filepath}/trials_model_params.csv")

    def save_metrics(self, filepath):
        self.frame.to_csv(f"{filepath}/raw_metrics.csv")
        metrics_frame = self.groups_frame[['toxicity', 'efficacy', 'utility', 'dose_error', 'final_dose_error']]
        print(metrics_frame)
        metrics_frame.to_csv(f"{filepath}/metrics.csv")
        print("Dose allocations")
        for subgroup_idx in range(self.num_subgroups):
            print(f"Subgroup: {subgroup_idx}")
            mask = self.patients == subgroup_idx
            dose_selections = np.array(np.unique(self.selected_doses[mask], return_counts=True)).T
            with open(f"{filepath}/dose_selections.npy", 'wb') as f:
                np.save(f, dose_selections)
            print(dose_selections)
        self.final_utilities.to_csv(f"{filepath}/final_utilities.csv")
        self.final_dose_rec.to_csv(f"{filepath}/final_dose_rec.csv")
        self.model_params.to_csv(f"{filepath}/final_model_params.csv")


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

        toxicity_data = self.dose_scenario.generate_toxicity_data(selected_arms, patients)
        efficacy_data = self.dose_scenario.generate_efficacy_data(selected_arms, patients)
        inducing_points = torch.tensor(self.dose_scenario.dose_labels.astype(np.float32))

        train_x = torch.tensor(selected_dose_labels.astype(np.float32)) 

        tox_y = torch.tensor(toxicity_data.astype(np.float32))
        eff_y = torch.tensor(efficacy_data.astype(np.float32))
        train_y = torch.stack([tox_y, eff_y], -1)
        patients = torch.tensor(patients, dtype=torch.long)

        return patients, selected_arms, train_x, train_y

    def select_dose(self, dose_labels, test_x, tox_dist, eff_dist, beta_param=1.):
        mask = np.isin(test_x, dose_labels)
        # Select safe doses using UCB of toxicity distribution
        # safe_dose_set = tox_dist.upper.numpy()[mask] <= self.dose_scenario.toxicity_threshold
        ucb = tox_dist.mean.cpu().numpy()[mask] + (beta_param * tox_dist.upper_width.cpu().numpy()[mask]) 
        safe_dose_set = ucb <= self.dose_scenario.toxicity_threshold

        # Select expanders of toxicity distribution?

        # Select optimal dose using EI of efficacy distribution
        xi = 0.01
        mean = eff_dist.mean.cpu().numpy()
        std = np.sqrt(eff_dist.variance.cpu().numpy())
        mean_optimum = eff_dist.samples.mean(axis=0).max().cpu().numpy()
        imp = mean - mean_optimum - xi
        Z = imp / std
        ei = (imp * scipy.stats.norm.cdf(Z)) + (std * scipy.stats.norm.pdf(Z))
        ei[std == 0.0] = 0.0
        ei = ei[mask]
        plot_ei = np.copy(ei)

        ei[~safe_dose_set] = -np.inf
        max_ei = ei.max()
        print(f"Max ei: {max_ei}")
        print(np.where(ei == max_ei))
        selected_dose = np.where(ei == max_ei)[0][-1]

        print(f"Safe dose set: {safe_dose_set}")
        print(f"Expected improvement: {ei}")
        print(f"Selected dose: {selected_dose}")

        # If all doses are unsafe, return first dose
        if safe_dose_set.sum() == 0:
            return 0, ucb, ei
        return selected_dose.item(), ucb, plot_ei

    def _plot_dose_selection_helper(self, ax, train_x, train_y, true_x, true_y, test_x, dist,
                                    acqui_vals, acqui_label, selected_dose, threshold=None):
        test_x = test_x.cpu().numpy()
        markevery_mask = np.isin(test_x, true_x)
        markevery = np.arange(len(test_x))[markevery_mask].tolist()
        gp_predicted = dist.mean.cpu().numpy()

        ax.scatter(train_x, train_y, s=40, c='k', alpha=0.1, label='Training Data')
        ax.plot(test_x, gp_predicted, 'b-', markevery=markevery, marker='o',label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        if dist.lower is not None and dist.upper is not None:
            ax.fill_between(dist.x_axis.cpu(), dist.lower.cpu(), dist.upper.cpu(), alpha=0.5)
        ax.plot(true_x, acqui_vals, 'gray', label=acqui_label, marker='o')
        ax.plot(true_x[selected_dose], acqui_vals[selected_dose], 'r', marker='o')
        if threshold is not None:
            ax.plot(test_x, np.repeat(threshold, len(test_x)), 'm', label='Toxicity Threshold')
        ax.set_ylim([0, 1.1])
        ax.legend()
    
    def plot_dose_selection(self, train_x, tox_train_y, eff_train_y, patients, num_subgroups,
                                 test_x, tox_dists, eff_dists, ucb, ei, tox_thre, selected_doses, cohort_outcomes, filepath):
        sns.set_theme(style="dark")
        fig, axs = plt.subplots(num_subgroups, 2, figsize=(8, 8))
        for subgroup_idx in range(num_subgroups):
            axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
            axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
            group_train_x = train_x[patients == subgroup_idx]
            group_tox_train_y = tox_train_y[patients == subgroup_idx]
            group_eff_train_y = eff_train_y[patients == subgroup_idx]

            self._plot_dose_selection_helper(axs[subgroup_idx, 0], group_train_x, group_tox_train_y,
                                                  self.dose_scenario.dose_labels, self.dose_scenario.toxicity_probs[subgroup_idx, :],
                                                  test_x, tox_dists[subgroup_idx], ucb[subgroup_idx, :], 'Confidence Bound', selected_doses[subgroup_idx], tox_thre)
            self._plot_dose_selection_helper(axs[subgroup_idx, 1], group_train_x, group_eff_train_y,
                                                  self.dose_scenario.dose_labels, self.dose_scenario.efficacy_probs[subgroup_idx, :],
                                                  test_x, eff_dists[subgroup_idx], ei[subgroup_idx, :], 'Expected Improvement', selected_doses[subgroup_idx])
        plt.figtext(0.5, 0.01, f"Subgroup, dose, tox, eff: {cohort_outcomes}", wrap=True, horizontalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{filepath}plot.png", dpi=300)
        plt.close()

    
    def select_dose_empirically(self, tox_estimate, eff_estimate, N_choose, beta_param=1.):
        tox_ucb = get_ucb(tox_estimate, beta_param, N_choose, N_choose.sum())
        safe_dose_set = tox_ucb <= self.dose_scenario.toxicity_threshold
        eff_ucb = get_ucb(eff_estimate, beta_param, N_choose, N_choose.sum())
        print(f"Empirical safe dose set:{safe_dose_set}")
        selected_dose = np.argmax(eff_ucb * safe_dose_set)

        return selected_dose

    def select_dose_utility(self, dose_labels, test_x, tox_dist, eff_dist, beta_param):
        mask = np.isin(test_x, dose_labels)
        #safe_dose_set = tox_dist.upper.numpy()[mask] <= self.dose_scenario.toxicity_threshold
        safe_dose_set = tox_dist.mean.numpy()[mask] + (beta_param * tox_dist.upper_width.numpy()[mask]) <= self.dose_scenario.toxicity_threshold


        # Select optimal dose using utility 
        utilities = self.calculate_dose_utility(tox_dist.mean.numpy()[mask], eff_dist.mean.numpy()[mask])
        utilities[~safe_dose_set] = -np.inf
        selected_dose = np.argmax(utilities)
        print(f"Safe dose set: {safe_dose_set}")
        print(f"Utilities: {utilities}")
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

    def select_final_dose_subgroups(self, dose_labels, test_x, tox_dists, eff_dists):
        mask = np.isin(test_x, dose_labels)
        dose_error = np.zeros(self.patient_scenario.num_subgroups)
        dose_rec = np.ones(self.patient_scenario.num_subgroups) * self.dose_scenario.num_doses
        # Select dose with max estimate efficacy that is also below toxicity threshold
        for subgroup_idx in range(self.patient_scenario.num_subgroups):
            tox_mean = tox_dists[subgroup_idx].mean[mask]
            eff_mean = eff_dists[subgroup_idx].mean[mask]
            dose_options = eff_mean.numpy() * (tox_mean.numpy() <= self.dose_scenario.toxicity_threshold)
            mtd_eff = np.max(dose_options)
            mtd_idx = np.argmax(dose_options)

            # If recommended dose is above eff threshold, assign this dose. Else assign no dose
            if mtd_eff >= self.dose_scenario.efficacy_threshold:
                dose_rec[subgroup_idx] = mtd_idx

        print(f"Final doses: {dose_rec}")
        dose_error = (dose_rec != self.dose_scenario.optimal_doses).astype(np.float32)
        return dose_error
    
    def select_final_dose_subgroups_utility(self, dose_labels, test_x, tox_dists, eff_dists, beta_param=1.):
        mask = np.isin(test_x, dose_labels)
        dose_error = np.zeros(self.patient_scenario.num_subgroups)
        dose_rec = np.ones(self.patient_scenario.num_subgroups) * self.dose_scenario.num_doses
        final_utilities = np.empty((self.patient_scenario.num_subgroups, self.dose_scenario.num_doses))
            
        # Select dose with highest utility that is below toxicity threshold
        for subgroup_idx in range(self.patient_scenario.num_subgroups):
            tox_mean = tox_dists[subgroup_idx].mean[mask].cpu().numpy()
            eff_mean = eff_dists[subgroup_idx].mean[mask].cpu().numpy()
            tox_upper_width = tox_dists[subgroup_idx].upper_width[mask].cpu().numpy()

            # safe_dose_set = tox_mean.numpy() <= self.dose_scenario.toxicity_threshold
            ucb = tox_mean + (beta_param * tox_upper_width) 
            safe_dose_set = tox_mean <= self.dose_scenario.toxicity_threshold

            print(f"Final safe dose set: {safe_dose_set}")
            utilities = self.calculate_dose_utility(tox_mean, eff_mean)
            utilities[~safe_dose_set] = -np.inf
            final_utilities[subgroup_idx, :] = utilities

            print(f"Final utilities: {utilities}")
            max_utility = utilities.max()
            best_dose_idx = np.where(utilities == max_utility)[0][-1]
            best_dose_val = eff_mean[best_dose_idx]

            # If recommended dose is above eff threshold and utility is positive, assign this dose.
            if best_dose_val >= self.dose_scenario.efficacy_threshold and utilities[best_dose_idx] >= 0:
                dose_rec[subgroup_idx] = best_dose_idx
            
            else: # Try assiging dose with highest efficacy in safe range. Else assign no dose
                print("Utilities negative, assigning highest eff.")
                eff_mean[~safe_dose_set] = -np.inf
                max_eff = eff_mean.max()
                print(f"Max eff: {max_eff}")
                if max_eff >= self.dose_scenario.efficacy_threshold:
                    max_eff_idx = np.where(eff_mean == max_eff)[0][-1]
                    dose_rec[subgroup_idx] = max_eff_idx

        print(f"Final doses: {dose_rec}")
        dose_error = (dose_rec != self.dose_scenario.optimal_doses).astype(np.float32)
        return dose_error, final_utilities, dose_rec

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
        
        ax.plot(posterior_dist.x_axis.numpy(), posterior_dist.mean, 'b-',
                markevery=np.isin(posterior_dist.x_axis.numpy(), true_x), marker='o',label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        if posterior_dist.lower is not None and posterior_dist.upper is not None:
            ax.fill_between(posterior_dist.x_axis, posterior_dist.lower, posterior_dist.upper, alpha=0.5)
        ax.set_ylim([0, 1.1])
        ax.legend()
    
    def _plot_subgroup_gp_results_helper(self, ax, train_x, train_y, true_x, true_y, test_x, dist):
        test_x = test_x.cpu().numpy()
        markevery_mask = np.isin(test_x, true_x)
        markevery = np.arange(len(test_x))[markevery_mask].tolist()
        gp_predicted = dist.mean.cpu().numpy()

        sns.set()
        ax.scatter(train_x, train_y, s=40, c='k', alpha=0.1, label='Training Data')
        ax.plot(test_x, gp_predicted, 'b-', markevery=markevery, marker='o',label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        if dist.lower is not None and dist.upper is not None:
            ax.fill_between(dist.x_axis.cpu(), dist.lower.cpu(), dist.upper.cpu(), alpha=0.5)
        ax.set_ylim([0, 1.1])
        ax.legend()
    
    def plot_subgroup_gp_results(self, train_x, tox_train_y, eff_train_y, patients, num_subgroups,
                                 test_x, tox_dists, eff_dists, filepath):
        fig, axs = plt.subplots(num_subgroups, 2, figsize=(8, 8))
        for subgroup_idx in range(num_subgroups):
            axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
            axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
            group_train_x = train_x[patients == subgroup_idx]
            group_tox_train_y = tox_train_y[patients == subgroup_idx]
            group_eff_train_y = eff_train_y[patients == subgroup_idx]

            self._plot_subgroup_gp_results_helper(axs[subgroup_idx, 0], group_train_x, group_tox_train_y,
                                                  self.dose_scenario.dose_labels, self.dose_scenario.toxicity_probs[subgroup_idx, :],
                                                  test_x, tox_dists[subgroup_idx])
            self._plot_subgroup_gp_results_helper(axs[subgroup_idx, 1], group_train_x, group_eff_train_y,
                                                  self.dose_scenario.dose_labels, self.dose_scenario.efficacy_probs[subgroup_idx, :],
                                                  test_x, eff_dists[subgroup_idx])
        plt.tight_layout()
        plt.savefig(f"{filepath}/plot.png", dpi=300)
        plt.close()

    def plot_trial_gp_results(self, tox_means, eff_means, test_x, dose_labels):
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
        ax.set_ylim([0, 1.1])
        ax.legend()

    def plot_subgroup_trial_gp_results(self, tox_means, eff_means, test_x,
                                       num_subgroups, results_dir):
        fig, axs = plt.subplots(num_subgroups, 2, figsize=(8, 8))
        for subgroup_idx in range(num_subgroups):
            axs[subgroup_idx, 0].set_title(f"Toxicity - Subgroup {subgroup_idx}")
            axs[subgroup_idx, 1].set_title(f"Efficacy - Subgroup {subgroup_idx}")
            self._plot_subgroup_trial_gp_results_helper(axs[subgroup_idx, 0], tox_means[:, subgroup_idx, :], test_x,
                                self.dose_scenario.dose_labels, self.dose_scenario.toxicity_probs[subgroup_idx, :])
            self._plot_subgroup_trial_gp_results_helper(axs[subgroup_idx, 1], eff_means[:, subgroup_idx, :], test_x,
                                self.dose_scenario.dose_labels, self.dose_scenario.efficacy_probs[subgroup_idx, :])

        fig.tight_layout()
        plt.savefig(f"{results_dir}/all_trials_plot.png")
        plt.close()

    def _plot_subgroup_trial_gp_results_helper(self, ax, rep_means, test_x, true_x, true_y):
        markevery_mask = np.isin(test_x, true_x)
        markevery = np.arange(len(test_x))[markevery_mask].tolist()

        sns.set()
        mean = np.mean(rep_means, axis=0)
        ci = 1.96 * np.std(rep_means, axis=0) / np.sqrt(rep_means.shape[0])
        ax.plot(test_x, mean, 'b-', markevery=markevery, marker='o', label='GP Predicted')
        ax.plot(true_x, true_y, 'g-', marker='o', label='True')
        ax.fill_between(test_x, (mean-ci), (mean+ci), alpha=0.5)
        ax.set_ylim([0, 1.1])
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
        return tox_runner, eff_runner, tox_dist, eff_dist

    def run_multitask_gp(self, num_epochs, num_latents, num_tasks, num_inducing_pts, train_x, train_y, test_x, num_confidence_samples):
        runner = MultitaskClassificationRunner(num_latents, num_tasks, num_inducing_pts)
        runner.train(train_x, train_y, num_epochs)
        posterior_latent_dist, posterior_observed_dist = runner.predict(test_x)
        post_dist = self.get_bernoulli_confidence_region(test_x, posterior_latent_dist, runner.likelihood, num_confidence_samples)
        tox_dist = PosteriorDist(post_dist.x_axis, post_dist.samples.mean[:, 0], post_dist.mean[:, 0], post_dist.variance[:, 0], post_dist.lower[:, 0], post_dist.upper[:, 0])
        eff_dist = PosteriorDist(post_dist.x_axis, post_dist.samples.mean[:, 1], post_dist.mean[:, 1], post_dist.variance[:, 1], post_dist.lower[:, 1], post_dist.upper[:, 1])
        return runner, tox_dist, eff_dist
    
    def run_separate_subgroup_gps(self, num_latents, num_tasks, num_inducing_pts, num_epochs, train_x,
                                  tox_train_y, eff_train_y, test_x, patients, num_subgroups,
                                  num_confidence_samples, learning_rate, use_gpu, init_lengthscale, init_variance):
        '''
        Separate for each subgroup and each task (tox/eff).
        '''
        patients = torch.LongTensor(patients)
        tox_runner = MultitaskSubgroupClassificationRunner(num_latents, num_tasks, self.dose_scenario.dose_labels)
        tox_runner.train(train_x, tox_train_y, patients, num_epochs, learning_rate, use_gpu, init_lengthscale, init_variance)
        tox_dists = []
        for subgroup in range(num_subgroups):
            task_indices = torch.LongTensor([subgroup for item in range(test_x.shape[0])])
            posterior_latent_dist, posterior_observed_dist = tox_runner.predict(test_x, task_indices, use_gpu)
            tox_dist = self.get_bernoulli_confidence_region(test_x, posterior_latent_dist, tox_runner.likelihood, num_confidence_samples)
            tox_dists.append(tox_dist)

        eff_runner = MultitaskSubgroupClassificationRunner(num_latents, num_tasks, self.dose_scenario.dose_labels)
        eff_runner.train(train_x, eff_train_y, patients, num_epochs, learning_rate, use_gpu, init_lengthscale, init_variance)
        eff_dists = []
        for subgroup in range(num_subgroups):
            task_indices = torch.LongTensor([subgroup for item in range(test_x.shape[0])])
            eff_posterior_latent_dist, eff_posterior_observed_dist = eff_runner.predict(test_x, task_indices, use_gpu)
            eff_dist = self.get_bernoulli_confidence_region(test_x, eff_posterior_latent_dist, eff_runner.likelihood, num_confidence_samples)
            eff_dists.append(eff_dist)
        return tox_runner, eff_runner, tox_dists, eff_dists

    def get_bernoulli_confidence_region(self, test_x, posterior_latent_dist, likelihood_model, num_samples):
        samples = posterior_latent_dist.sample_n(num_samples)
        likelihood_samples = likelihood_model(samples)
        lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
        upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
        mean = likelihood_samples.mean.mean(axis=0)
        variance = likelihood_samples.mean.var(axis=0)
        upper_width = upper - mean
        return PosteriorDist(test_x, samples, mean, variance, lower, upper, upper_width)

    def get_safe_dose_set_from_gradients(self, highest_safe_dose_idx, test_x, subgroup_idx, tox_runner, x_mask,
                                        subgroup_tox_mean):
        highest_safe_dose = self.dose_scenario.dose_labels[highest_safe_dose_idx]
        print(f"Highest safe dose: {highest_safe_dose_idx}")
        # Calculate gradients
        X = torch.autograd.Variable(test_x, requires_grad=True)
        task_indices = torch.LongTensor([subgroup_idx for item in range(test_x.shape[0])])
        observed_pred = tox_runner.likelihood(tox_runner.model(X, task_indices=task_indices))
        gradient = torch.autograd.grad(observed_pred.mean.sum(), X)[0][x_mask]
        gradient[gradient < 0] = 0.0
        print(f"Gradients: {gradient}")

        safe_dose_set = np.empty(self.dose_scenario.dose_labels.shape, dtype=np.bool)

        for idx, dose in enumerate(self.dose_scenario.dose_labels):
            if idx <= highest_safe_dose_idx:
                safe_dose_set[idx] = True
            else:
                expected_tox = subgroup_tox_mean[highest_safe_dose_idx] + (gradient[highest_safe_dose_idx] * (dose - highest_safe_dose))
                print(f"Dose {idx} expected tox: {expected_tox}")
                safe_dose_set[idx] = expected_tox <= self.dose_scenario.toxicity_threshold
        print(f"Safe dose set gradient: {safe_dose_set}")
        return safe_dose_set
    
    def get_safe_dose_set(self, prev_safe_dose_set, test_x, subgroup_idx, tox_runner, x_mask, subgroup_tox_mean):
        # For all doses outside of the safe dose set
        # Estimate toxicity wrt the all doses inside the safe dose set
        # If it is safe for all estimates, then dose can be included in new safe dose set
        pass

    def calculate_dose_utility(self, tox_values, eff_values):
        tox_threshold = self.dose_scenario.toxicity_threshold
        eff_threshold = self.dose_scenario.efficacy_threshold
        p_param = self.dose_scenario.p_param
        tox_term = (tox_values / tox_threshold) ** p_param
        eff_term = ((1. - eff_values) / (1. - eff_threshold)) ** p_param
        utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
        return utilities


##### Examples #####
def dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples, show_plot=True):
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    
    patients, selected_arms, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
    tox_runner, eff_runner, tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                     train_y[:, 0], train_y[:, 1], test_x,
                                                     num_confidence_samples)

    final_dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment_metrics = DoseExperimentMetrics(num_samples, np.sum(train_y[:, 0].numpy()), np.sum(train_y[:, 1].numpy()),
                                               selected_arms, dose_scenario.optimal_doses, final_dose_error)
    if show_plot:
        experiment_metrics.print_metrics()
        experiment.plot_gp_results(train_x.numpy(), train_y[:, 0].numpy(), train_y[:, 1].numpy(), tox_dist, eff_dist)
    return experiment_metrics, tox_dist, eff_dist

def multitask_dose_example(experiment, dose_scenario, num_samples, num_epochs,
                           num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
                           show_plot=True):
    _, selected_arms, train_x, train_y, test_x = experiment.get_offline_data(num_samples)
    runner, tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                     train_x, train_y, test_x, num_confidence_samples)

    final_dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment_metrics = DoseExperimentMetrics(num_samples, np.sum(train_y[:, 0].numpy()), np.sum(train_y[:, 1].numpy()),
                                               selected_arms, dose_scenario.optimal_doses, final_dose_error)
    if show_plot:
        experiment_metrics.print_metrics()
        experiment.plot_gp_results(train_x.numpy(), train_y[:, 0].numpy(), train_y[:, 1].numpy(), tox_dist, eff_dist)
    return experiment_metrics, tox_dist, eff_dist

def subgroups_dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples,
                           num_latents, num_tasks, num_inducing_pts, learning_rate, filepath):
    inducing_points = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    patients, selected_arms, train_x, train_y = experiment.get_offline_data(num_samples)
    tox_train_y = train_y[:, 0]
    eff_train_y = train_y[:, 1]
    num_subgroups = dose_scenario.num_subgroups

    dose_labels = dose_scenario.dose_labels.astype(np.float32)
    test_x = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    test_x = np.unique(test_x)
    np.sort(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)

    tox_runner, eff_runner, tox_dists, eff_dists \
        = experiment.run_separate_subgroup_gps(num_latents, num_tasks, num_inducing_pts, num_epochs, train_x,
                                               tox_train_y, eff_train_y, test_x, patients, num_subgroups,
                                               num_confidence_samples, learning_rate)

    final_dose_error = experiment.select_final_dose_subgroups(dose_labels, test_x, tox_dists, eff_dists)
    utilities = [experiment.calculate_dose_utility(dose_scenario.get_toxicity_prob(arm_idx, group_idx), dose_scenario.get_efficacy_prob(arm_idx, group_idx))\
                 for arm_idx, group_idx in zip(selected_arms, patients)]
    experiment_metrics = DoseExperimentSubgroupMetrics(num_samples, patients, num_subgroups, tox_train_y.numpy(), eff_train_y.numpy(), 
                                                       selected_arms, dose_scenario.optimal_doses, final_dose_error, np.array(utilities))
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    experiment_metrics.save_metrics(filepath)
    experiment.plot_subgroup_gp_results(train_x.numpy(), tox_train_y.numpy(), eff_train_y.numpy(), patients, num_subgroups,
                                        test_x, tox_dists, eff_dists, filepath)
    return experiment_metrics, tox_dists, eff_dists


##### Online Examples #####
def online_dose_example(experiment, dose_scenario, num_samples, num_epochs,
                        num_confidence_samples, cohort_size, show_plot=True):
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

        tox_runner, eff_runner, tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                                                 tox_train_y, eff_train_y, test_x,
                                                                                  num_confidence_samples)

        selected_dose = experiment.select_dose(tox_dist, eff_dist)
        if selected_dose > max_dose + 1:
            selected_dose = max_dose + 1
            max_dose = selected_dose
        print(f"Selected dose: {selected_dose}")

        selected_doses += [selected_dose for item in range(cohort_size)]
        selected_dose_values += [dose_scenario.dose_labels[selected_dose] for item in range(cohort_size)]
        toxicity_responses += [dose_scenario.sample_toxicity_event(selected_dose) for item in range(cohort_size)]
        efficacy_responses += [dose_scenario.sample_efficacy_event(selected_dose) for item in range(cohort_size)]

        timestep += cohort_size

    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)
    
    tox_runner, eff_runner, tox_dist, eff_dist = experiment.run_separate_gps(inducing_points, num_epochs, train_x,
                                                                             tox_train_y, eff_train_y, test_x,
                                                                             num_confidence_samples)

    final_dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment_metrics = DoseExperimentMetrics(num_samples, np.sum(toxicity_responses), np.sum(efficacy_responses),
                                               selected_doses, dose_scenario.optimal_doses, final_dose_error)
    if show_plot:
        experiment_metrics.print_metrics()
        plot_test_x = torch.cat((torch.arange(torch.min(test_x), torch.max(test_x) + 0.1, 0.1), test_x)).unique()
        plot_test_x.sort()
        tox_latent_dist, _ = tox_runner.predict(plot_test_x)
        eff_latent_dist, _ = eff_runner.predict(plot_test_x)

        tox_plot_dist = experiment.get_bernoulli_confidence_region(plot_test_x,
                                                                  tox_latent_dist,
                                                                  tox_runner.likelihood,
                                                                  num_confidence_samples)
        eff_plot_dist = experiment.get_bernoulli_confidence_region(plot_test_x,
                                                                   eff_latent_dist,
                                                                   eff_runner.likelihood,
                                                                   num_confidence_samples)

        experiment.plot_gp_results(train_x.numpy(), tox_train_y.numpy(),
                                   eff_train_y.numpy(), tox_plot_dist, eff_plot_dist)
    return experiment_metrics, tox_dist, eff_dist

def online_multitask_dose_example(experiment, dose_scenario, num_samples, num_epochs,
                                  num_confidence_samples, cohort_size, num_latents,
                                  num_tasks, num_inducing_pts, show_plot=True):
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

        runner, tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                                 train_x, train_y, test_x, num_confidence_samples)
        print(f"Tox dist: {tox_dist.mean}")
        selected_dose = experiment.select_dose(tox_dist, eff_dist)
        if selected_dose > max_dose + 1:
            selected_dose = max_dose + 1
            max_dose = selected_dose
        print(f"Selected dose: {selected_dose}")

        # Get responses
        selected_doses += [selected_dose for item in range(cohort_size)]
        selected_dose_values += [dose_scenario.dose_labels[selected_dose] for item in range(cohort_size)]
        toxicity_responses += [dose_scenario.sample_toxicity_event(selected_dose) for item in range(cohort_size)]
        efficacy_responses += [dose_scenario.sample_efficacy_event(selected_dose) for item in range(cohort_size)]

        timestep += cohort_size
    
    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)
    train_y = torch.stack([tox_train_y, eff_train_y], -1)

    runner, tox_dist, eff_dist = experiment.run_multitask_gp(num_epochs, num_latents, num_tasks, num_inducing_pts,
                                                     train_x, train_y, test_x, num_confidence_samples)
    print(f"Tox dist: {tox_dist.mean}")
    final_dose_error = experiment.select_final_dose(tox_dist.mean, eff_dist.mean)
    experiment_metrics = DoseExperimentMetrics(num_samples, np.sum(toxicity_responses), np.sum(efficacy_responses),
                                               selected_doses, dose_scenario.optimal_doses, final_dose_error)
    
    if show_plot:
        experiment_metrics.print_metrics()
        experiment.plot_gp_results(train_x.numpy(), tox_train_y.numpy(),
                                   eff_train_y.numpy(), tox_dist, eff_dist)

    return experiment_metrics, tox_dist, eff_dist

def online_subgroups_dose_example(experiment, dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
                                  cohort_size, learning_rate, beta_param, filepath, use_gpu, init_lengthscale, init_variance):
    patients = patient_scenario.generate_samples(num_samples)
    num_subgroups = dose_scenario.num_subgroups
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    timestep = 0
    max_doses = np.zeros(num_subgroups)

    empirical_efficacy_estimate = np.zeros((num_subgroups, dose_scenario.num_doses))
    empirical_toxicity_estimate = np.zeros((num_subgroups, dose_scenario.num_doses))
    N_choose = np.zeros((num_subgroups, dose_scenario.num_doses))

    # For the first cohort, start with the lowest dose
    selected_doses = [0 for item in range(cohort_size)]
    cohort_patients = patients[timestep:timestep + cohort_size]
    selected_dose_values = [dose_scenario.dose_labels[dose] for dose in selected_doses]
    toxicity_responses = [dose_scenario.sample_toxicity_event(dose, subgroup_idx) for dose, subgroup_idx in zip(selected_doses, cohort_patients)]
    efficacy_responses = [dose_scenario.sample_efficacy_event(dose, subgroup_idx) for dose, subgroup_idx in zip(selected_doses, cohort_patients)]
    for t in range(timestep, timestep + cohort_size):
        subgroup_idx = patients[t]
        dose_idx = selected_doses[t]
        empirical_efficacy_estimate[subgroup_idx, dose_idx] = \
            ((empirical_efficacy_estimate[subgroup_idx, dose_idx] * N_choose[subgroup_idx, dose_idx]) + efficacy_responses[t])/\
            (N_choose[subgroup_idx, dose_idx] + 1.)
        empirical_toxicity_estimate[subgroup_idx, dose_idx] = \
            ((empirical_toxicity_estimate[subgroup_idx, dose_idx] * N_choose[subgroup_idx, dose_idx]) + toxicity_responses[t])/\
            (N_choose[subgroup_idx, dose_idx] + 1.)
        N_choose[subgroup_idx, dose_idx] += 1

    timestep += cohort_size
    
    dose_labels = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    test_x = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    test_x = np.unique(test_x)
    np.sort(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    mask = np.isin(test_x, dose_labels)
    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        # Train model on seen patients
        patient_indices = patients[:timestep]
        train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
        tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
        eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

        tox_runner, eff_runner, tox_dists, eff_dists \
            = experiment.run_separate_subgroup_gps(num_latents, num_tasks, num_inducing_pts, num_epochs, train_x,
                                                   tox_train_y, eff_train_y, test_x, patient_indices, num_subgroups,
                                                   num_confidence_samples, learning_rate, use_gpu, init_lengthscale, init_variance)
            
        cohort_patients = patients[timestep: timestep + cohort_size]
        print(f"Cohort patients: {cohort_patients}")

        # Calculate dose to select for each subgroup
        timestep_selected_doses = np.ones(num_subgroups, dtype=np.int32) * dose_scenario.num_doses
        timestep_ucbs = np.empty((num_subgroups, dose_scenario.num_doses))
        timestep_eis = np.empty((num_subgroups, dose_scenario.num_doses))
        for subgroup_idx in range(num_subgroups):
            print(f"Tox dist {subgroup_idx}: {tox_dists[subgroup_idx].mean[mask]}")
            print(f"Eff dist {subgroup_idx}: {eff_dists[subgroup_idx].mean[mask]}")
            selected_dose, ucb, ei = experiment.select_dose(dose_labels, test_x, tox_dists[subgroup_idx],
                                                            eff_dists[subgroup_idx], beta_param=beta_param)
            # Only increase one dose at a time max
            if selected_dose > max_doses[subgroup_idx] + 1:
                selected_dose = max_doses[subgroup_idx] + 1
            
            timestep_selected_doses[subgroup_idx] = selected_dose
            timestep_ucbs[subgroup_idx, :] = ucb
            timestep_eis[subgroup_idx, :] = ei
        
        # Assign doses to each patient in cohort, update metrics
        for subgroup_idx in cohort_patients:
            selected_dose = timestep_selected_doses[subgroup_idx]
            selected_doses.append(selected_dose)
            selected_dose_values.append(dose_scenario.dose_labels[selected_dose])
            toxicity_responses.append(dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx))
            efficacy_responses.append(dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)) 

        # Update max doses
        for subgroup_idx in range(num_subgroups):
            subgroup_mask = (cohort_patients == subgroup_idx).tolist()
            subgroup_doses = np.array(selected_doses[timestep: timestep + cohort_size])[subgroup_mask]
            if len(subgroup_doses) > 0:
                if subgroup_doses.max() > max_doses[subgroup_idx]:
                    max_doses[subgroup_idx] = subgroup_doses.max()
    
        print(f"Max doses: {max_doses}")
        print(f"Timestep selected doses: {timestep_selected_doses}")
        print(f"Selected doses: {selected_doses[timestep: timestep+cohort_size]}")
        prev_cohort_outcomes = [patients[timestep-cohort_size:timestep], selected_doses[timestep-cohort_size:timestep],
                                toxicity_responses[timestep-cohort_size:timestep],
                                efficacy_responses[timestep-cohort_size:timestep]]
        experiment.plot_dose_selection(train_x.numpy(), tox_train_y.numpy(), eff_train_y.numpy(), patient_indices, num_subgroups,
                                        test_x, tox_dists, eff_dists, timestep_ucbs, timestep_eis, dose_scenario.toxicity_threshold,
                                        timestep_selected_doses, prev_cohort_outcomes, f"{filepath}/timestep{timestep}")
        
        # Update empirical estimates
        for t in range(timestep, timestep + cohort_size):
            subgroup_idx = patients[t]
            dose_idx = selected_doses[t]
            empirical_efficacy_estimate[subgroup_idx, dose_idx] = \
                ((empirical_efficacy_estimate[subgroup_idx, dose_idx] * N_choose[subgroup_idx, dose_idx]) + efficacy_responses[t])/\
                (N_choose[subgroup_idx, dose_idx] + 1.)
            empirical_toxicity_estimate[subgroup_idx, dose_idx] = \
                ((empirical_toxicity_estimate[subgroup_idx, dose_idx] * N_choose[subgroup_idx, dose_idx]) + toxicity_responses[t])/\
                (N_choose[subgroup_idx, dose_idx] + 1.)
            N_choose[subgroup_idx, dose_idx] += 1

        timestep += cohort_size
        
        
    train_x = torch.tensor(selected_dose_values, dtype=torch.float32)
    tox_train_y = torch.tensor(toxicity_responses, dtype=torch.float32)
    eff_train_y = torch.tensor(efficacy_responses, dtype=torch.float32)

    print(selected_doses)
    print(train_x)
    print(tox_train_y)
    print(eff_train_y)
    print(list(patients))
    
    # Run final model
    tox_runner, eff_runner, tox_dists, eff_dists \
        = experiment.run_separate_subgroup_gps(num_latents, num_tasks, num_inducing_pts, num_epochs, train_x,
                                               tox_train_y, eff_train_y, test_x, patients, num_subgroups,
                                               num_confidence_samples, learning_rate, use_gpu, init_lengthscale, init_variance)
    for subgroup_idx in range(num_subgroups):
        print(f"Tox dist {subgroup_idx}: {tox_dists[subgroup_idx].mean[mask]}")
        print(f"Eff dist {subgroup_idx}: {eff_dists[subgroup_idx].mean[mask]}")

    for name, param in tox_runner.model.named_parameters():
            print(name, param.data)
    tox_lengthscale = np.squeeze(tox_runner.model.covar_module.base_kernel.kernels[0].lengthscale.detach().cpu().numpy())
    tox_variance = np.squeeze(tox_runner.model.covar_module.base_kernel.kernels[1].variance.detach().cpu().numpy())

    for name, param in eff_runner.model.named_parameters():
            print(name, param.data)
    eff_lengthscale = np.squeeze(eff_runner.model.covar_module.base_kernel.kernels[0].lengthscale.detach().cpu().numpy())
    eff_variance = np.squeeze(eff_runner.model.covar_module.base_kernel.kernels[1].variance.detach().cpu().numpy())

    model_params_frame = pd.DataFrame({'tox_lengthscale': tox_lengthscale,
                                       'tox_variance': tox_variance,
                                       'eff_lengthscale': eff_lengthscale,
                                       'eff_variance': eff_variance})
    print(model_params_frame)
        
    final_dose_error, final_utilities, final_dose_rec \
        = experiment.select_final_dose_subgroups_utility(dose_labels, test_x, tox_dists, eff_dists, beta_param)

    # Calculate utilities retrospectively
    utilities = [experiment.calculate_dose_utility(dose_scenario.get_toxicity_prob(arm_idx, group_idx), dose_scenario.get_efficacy_prob(arm_idx, group_idx))\
                 for arm_idx, group_idx in zip(selected_doses, patients)]
    utilities = np.array(utilities, dtype=np.float32)

    experiment_metrics = DoseExperimentSubgroupMetrics(num_samples, patients, num_subgroups, tox_train_y.numpy(), eff_train_y.numpy(), 
                                                       selected_doses, dose_scenario.optimal_doses, final_dose_error, utilities,
                                                       final_utilities, final_dose_rec, model_params_frame)

    experiment_metrics.save_metrics(filepath)
    experiment.plot_subgroup_gp_results(train_x.numpy(), tox_train_y.numpy(), eff_train_y.numpy(), patients, num_subgroups,
                                        test_x, tox_dists, eff_dists, filepath)

    return experiment_metrics, tox_dists, eff_dists

            # subgroup_tox_mean = tox_dists[subgroup_idx].mean.numpy()[mask] 
            # subgroup_tox_upper = tox_dists[subgroup_idx].upper.numpy()[mask] 

            # safe_dose_set = subgroup_tox_upper <= dose_scenario.toxicity_threshold
            # lowest_unsafe_dose_idx = np.where(safe_dose_set == False)
            # print(f"Initial safe dose set: {safe_dose_set}")

            # if lowest_unsafe_dose_idx[0].shape[0] == 0:
            #     # if all doses are deemed safe, examine safety based on last seen dose
            #     subgroup_seen_doses = np.array(seen_doses)[patient_indices == subgroup_idx]
            #     highest_safe_dose_idx = 0
            #     if subgroup_seen_doses.shape[0] > 0:
            #         highest_safe_dose_idx = subgroup_seen_doses.max()
            #     safe_dose_set = experiment.get_safe_dose_set_from_gradients(highest_safe_dose_idx, test_x, subgroup_idx, tox_runner, mask,
            #                                                                 subgroup_tox_mean)

            # # All doses unsafe, pick lowest dose again
            # # TODO add hard constraint here to stop trial if this happens too many times
            # elif lowest_unsafe_dose_idx[0][0] == 0:
            #     safe_dose_set = np.empty(dose_labels.shape, dtype=np.bool)
            #     safe_dose_set.fill(False)
            #     safe_dose_set[0] = True
    
            # else:
            #     # Calculate expected toxicity of next doses based on gradient at highest safe and seen dose
            #     subgroup_seen_doses = np.array(seen_doses)[patient_indices == subgroup_idx]
            #     if subgroup_seen_doses.shape[0] > 0:
            #         highest_safe_dose_idx = min(lowest_unsafe_dose_idx[0][0] - 1, subgroup_seen_doses.max())
            #     else:
            #         highest_safe_dose_idx = lowest_unsafe_dose_idx[0][0] - 1
            #     safe_dose_set = experiment.get_safe_dose_set_from_gradients(highest_safe_dose_idx, test_x, subgroup_idx, tox_runner, mask,
            #                                                                 subgroup_tox_mean)

            # eff_dist = eff_dists[subgroup_idx]
            # xi = 0.01
            # mean = eff_dist.mean
            # std = np.sqrt(eff_dist.variance)
            # mean_optimum = eff_dist.samples.mean(axis=0).max()
            # imp = mean - mean_optimum - xi
            # Z = imp / std
            # ei = (imp * scipy.stats.norm.cdf(Z)) + (std * scipy.stats.norm.pdf(Z))
            # ei[std == 0.0] = 0.0
            # ei = ei[mask]
            # selected_dose = np.argmax(ei * safe_dose_set)
            # print(f"Expected improvement: {ei}")

##### Trials Examples #####
def dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, num_reps):
    metrics = []
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    tox_means = np.empty((num_reps, test_x.shape[0]))
    eff_means = np.empty((num_reps, test_x.shape[0]))

    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    
    for trial in range(num_reps):
        print(f"Trial {trial}")
        trial_metrics, tox_dist, eff_dist = dose_example(experiment, dose_scenario,
                                                         num_samples, num_epochs, num_confidence_samples,
                                                         show_plot=False)
        metrics.append(trial_metrics)
        tox_means[trial, :] = tox_dist.mean
        eff_means[trial, :] = eff_dist.mean
    
    DoseExperimentMetrics.print_merged_metrics(metrics)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)

def multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_latents, num_tasks, num_inducing_pts, num_reps):
    metrics = []
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    tox_means = np.empty((num_reps, test_x.shape[0]))
    eff_means = np.empty((num_reps, test_x.shape[0]))

    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    
    for trial in range(num_reps):
        print(f"Trial {trial}")
        trial_metrics, tox_dist, eff_dist = multitask_dose_example(experiment, dose_scenario,
                                                                   num_samples, num_epochs, num_confidence_samples,
                                                                   num_latents, num_tasks, num_inducing_pts,
                                                                   show_plot=False)
        metrics.append(trial_metrics)
        tox_means[trial, :] = tox_dist.mean
        eff_means[trial, :] = eff_dist.mean
    
    DoseExperimentMetrics.print_merged_metrics(metrics)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)

def subgroups_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_latents, num_tasks, num_inducing_pts, num_reps,
                                  learning_rate, results_dir):
    metrics = []
    # test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    true_x = dose_scenario.dose_labels.astype(np.float32)
    test_x = np.concatenate([np.arange(true_x.min(), true_x.max(), 0.05, dtype=np.float32), true_x])
    test_x = np.unique(test_x)
    np.sort(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)

    tox_means = np.empty((num_reps, patient_scenario.num_subgroups, test_x.shape[0]))
    eff_means = np.empty((num_reps, patient_scenario.num_subgroups, test_x.shape[0]))

    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    for trial in range(num_reps):
        print(f"Trial {trial}")
        filepath = f"{results_dir}/trial{trial}"
        trial_metrics, tox_dists, eff_dists = subgroups_dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples,
                           num_latents, num_tasks, num_inducing_pts, learning_rate, filepath)
        metrics.append(trial_metrics)

        for subgroup_idx in range(patient_scenario.num_subgroups):
            tox_means[trial, subgroup_idx, :] = tox_dists[subgroup_idx].mean
            eff_means[trial, subgroup_idx, :] = eff_dists[subgroup_idx].mean
    
    DoseExperimentSubgroupMetrics.print_merged_metrics(metrics, results_dir)
    experiment.plot_subgroup_trial_gp_results(tox_means, eff_means, test_x, patient_scenario.num_subgroups, results_dir)


def online_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                               num_confidence_samples, cohort_size, num_reps):
    metrics = []
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))

    tox_means = np.empty((num_reps, test_x.shape[0]))
    eff_means = np.empty((num_reps, test_x.shape[0]))
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    
    for trial in range(num_reps):
        print(f"Trial {trial}")
        trial_metrics, tox_dist, eff_dist = online_dose_example(experiment, dose_scenario, num_samples, num_epochs,
                                                                num_confidence_samples, cohort_size, show_plot=False)
        metrics.append(trial_metrics)
        tox_means[trial, :] = tox_dist.mean
        eff_means[trial, :] = eff_dist.mean
    
    DoseExperimentMetrics.print_merged_metrics(metrics)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)


def online_multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                                         num_confidence_samples, cohort_size, num_latents,
                                         num_tasks, num_inducing_pts, num_reps):
        
    metrics = []
    test_x = torch.tensor(dose_scenario.dose_labels.astype(np.float32))
    tox_means = np.empty((num_reps, test_x.shape[0]))
    eff_means = np.empty((num_reps, test_x.shape[0]))

    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)
    
    for trial in range(num_reps):
        print(f"Trial {trial}")
        trial_metrics, tox_dist, eff_dist = online_multitask_dose_example(experiment, dose_scenario, num_samples,
                                                                          num_epochs, num_confidence_samples, cohort_size,
                                                                          num_latents, num_tasks, num_inducing_pts, show_plot=False)
        metrics.append(trial_metrics)
        tox_means[trial, :] = tox_dist.mean
        eff_means[trial, :] = eff_dist.mean
    
    DoseExperimentMetrics.print_merged_metrics(metrics)
    experiment.plot_trial_gp_results(tox_means, eff_means, test_x)

def online_subgroup_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
                                        num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
                                        cohort_size, learning_rate, num_reps, beta_param, results_dir,
                                        use_gpu, init_lengthscale, init_variance):
    metrics = []
    true_x = dose_scenario.dose_labels.astype(np.float32)
    test_x = np.concatenate([np.arange(true_x.min(), true_x.max(), 0.05, dtype=np.float32), true_x])
    test_x = np.unique(test_x)
    np.sort(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)

    tox_means = np.empty((num_reps, patient_scenario.num_subgroups, test_x.shape[0]))
    eff_means = np.empty((num_reps, patient_scenario.num_subgroups, test_x.shape[0]))

    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)

    for trial in range(num_reps):
        print(f"Trial {trial}")
        filepath = f"{results_dir}/trial{trial}"
        trial_metrics, tox_dists, eff_dists = online_subgroups_dose_example(experiment, dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
                                  cohort_size, learning_rate, beta_param, filepath, use_gpu, init_lengthscale, init_variance)
        metrics.append(trial_metrics)
        for subgroup_idx in range(patient_scenario.num_subgroups):
            tox_means[trial, subgroup_idx, :] = tox_dists[subgroup_idx].mean.cpu()
            eff_means[trial, subgroup_idx, :] = eff_dists[subgroup_idx].mean.cpu()
        
    
        with open(f"{filepath}/tox_means.npy", 'wb') as f:
            np.save(f, tox_means)
        with open(f"{filepath}/eff_means.npy", 'wb') as f:
            np.save(f, eff_means)
    
    DoseExperimentSubgroupMetrics.print_merged_metrics(metrics, results_dir)
    experiment.plot_subgroup_trial_gp_results(tox_means, eff_means, test_x, patient_scenario.num_subgroups, results_dir)



def main():
    dose_scenario = DoseFindingScenarios.subgroups_example_1()
    patient_scenario = TrialPopulationScenarios.equal_population(2)
    experiment = DoseFindingExperiment(dose_scenario, patient_scenario)

    num_samples = 51
    num_epochs = 300
    num_confidence_samples = 10000

    num_latents = 3
    num_tasks = patient_scenario.num_subgroups
    num_inducing_pts = dose_scenario.num_doses

    num_reps = 2
    cohort_size = 3
    learning_rate = 0.01
    beta_param = 0.2
    use_gpu = False
    init_lengthscale = None
    init_variance = None

    true_utilities = experiment.calculate_dose_utility(dose_scenario.toxicity_probs, dose_scenario.efficacy_probs)
    print(f"True utilities: {true_utilities}")

    # dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples)
    # multitask_dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples,
    #                        num_latents, num_tasks, num_inducing_pts)

    # online_dose_example(experiment, dose_scenario, num_samples, num_epochs, num_confidence_samples, cohort_size)
    # online_multitask_dose_example(experiment, dose_scenario, num_samples, num_epochs,
    #                               num_confidence_samples, cohort_size, num_latents,
    #                               num_tasks, num_inducing_pts)
    
    # dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs, num_confidence_samples, num_reps)
    # multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                               num_confidence_samples, num_latents, num_tasks, num_inducing_pts, num_reps)
    # online_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                            num_confidence_samples, cohort_size, num_reps)
    # online_multitask_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                                      num_confidence_samples, cohort_size, num_latents,
    #                                      num_tasks, num_inducing_pts, num_reps)



    # subgroups_dose_example(experiment, dose_scenario, num_samples, num_epochs,
    #                        num_confidence_samples, num_latents, num_tasks, num_inducing_pts, learning_rate)

    online_subgroups_dose_example(experiment, dose_scenario, patient_scenario, num_samples, num_epochs,
                                  num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
                                  cohort_size, learning_rate, beta_param, "results/34_example",
                                  use_gpu=use_gpu, init_lengthscale=init_lengthscale, init_variance=init_variance)
    # subgroups_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                               num_confidence_samples, num_latents, num_tasks, num_inducing_pts, num_reps,
    #                               learning_rate, "results/exp5")
    
    # online_subgroup_dose_example_trials(dose_scenario, patient_scenario, num_samples, num_epochs,
    #                                     num_confidence_samples, num_latents, num_tasks, num_inducing_pts,
    #                                     cohort_size, learning_rate, num_reps, beta_param, "results/exp11",
    #                                     use_gpu, init_lengthscale, init_variance)


main()

