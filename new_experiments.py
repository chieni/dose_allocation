import os
import argparse
import math
import pickle
from statistics import NormalDist

import numpy as np
import scipy.stats
import torch
import gpytorch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from new_gp import MultitaskClassificationRunner, MultitaskGPModel
from plots import plot_gp, plot_gp_timestep, plot_gp_trials
from experiment_utils import PosteriorPrediction


class DoseExperimentMetrics:
    def __init__(self, dose_scenario, subgroup_indices, selected_doses, tox_outcomes, eff_outcomes,
                 tox_predicted, tox_upper, eff_predicted, final_selected_doses, final_utilities):
        self.num_subgroups = final_selected_doses.shape[0]
        self.subgroup_indices = subgroup_indices
        self.selected_doses = selected_doses
        self.tox_outcomes = tox_outcomes
        self.eff_outcomes = eff_outcomes
        self.tox_predicted = tox_predicted
        self.tox_upper = tox_upper
        self.eff_predicted = eff_predicted
        self.final_selected_doses = final_selected_doses
        self.optimal_doses = dose_scenario.optimal_doses
        self.final_utilities = final_utilities

        # Calculate dose error
        optimal_dose_per_sample = np.array([self.optimal_doses[subgroup_idx] for subgroup_idx in subgroup_indices])
        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)

        # Calculate final dose error
        final_dose_error = (final_selected_doses != self.optimal_doses).astype(np.float32)

        # Calculate utility
        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        utilities = calculate_utility(selected_tox_probs, selected_eff_probs,
                                      dose_scenario.toxicity_threshold,
                                      dose_scenario.efficacy_threshold,
                                      dose_scenario.tox_weight,
                                      dose_scenario.eff_weight)
                                      
        # Calculate safety constraint violation
        safety_violations = np.array(selected_tox_probs > dose_scenario.toxicity_threshold, dtype=np.int32)

        self.metrics_frame = pd.DataFrame({
            'subgroup_idx': subgroup_indices,
            'tox_outcome': tox_outcomes,
            'eff_outcome': eff_outcomes,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utility': utilities,
            'safety_violations': safety_violations
        })

        grouped_metrics_frame = self.metrics_frame.groupby(['subgroup_idx']).mean()
        total_frame = pd.DataFrame(self.metrics_frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        grouped_metrics_frame = pd.concat([grouped_metrics_frame, total_frame])
        grouped_metrics_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / self.num_subgroups]])
        self.grouped_metrics_frame = grouped_metrics_frame

        self.final_dose_rec = pd.DataFrame({'final_dose_rec': final_selected_doses})
        self.final_dose_rec['subgroup_idx'] = self.final_dose_rec.index

        self.subgroup_predictions = []
        for subgroup_idx in range(self.num_subgroups):
            predicted_frame = pd.DataFrame({
                'tox_predicted': self.tox_predicted[subgroup_idx, :],
                'tox_upper': self.tox_upper[subgroup_idx, :],
                'eff_predicted': self.eff_predicted[subgroup_idx, :],
                'final_utilities': self.final_utilities[subgroup_idx, :]
            })
            self.subgroup_predictions.append(predicted_frame)

        # Get dose allocation counts
        dose_counts_frame = self.metrics_frame[['subgroup_idx', 'selected_dose']].groupby('subgroup_idx')['selected_dose'].apply(list)
        self.dose_counts_frame = dose_counts_frame.apply(lambda x: pd.Series(x).value_counts())

    @classmethod
    def save_merged_metrics(cls, metrics_list, filepath):
        frame = pd.concat([df.grouped_metrics_frame for df in metrics_list])
        grouped_frame = frame.groupby(frame.index)
        mean_frame = grouped_frame.mean()
        var_frame = grouped_frame.var()
        mean_frame = mean_frame[['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']]
        var_frame = var_frame[['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']]

        print(mean_frame)
        print(var_frame)
        mean_frame.to_csv(f"{filepath}/final_metric_means.csv")
        var_frame.to_csv(f"{filepath}/final_metric_var.csv")

        dose_counts = pd.concat([df.dose_counts_frame for df in metrics_list])
        dose_counts_mean = dose_counts.groupby(level=0).mean()
        print(dose_counts_mean)
        dose_counts_mean.to_csv(f"{filepath}/all_dose_counts.csv")

        dose_recs = pd.concat([df.final_dose_rec for df in metrics_list])
        dose_recs_grouped = dose_recs.groupby('subgroup_idx')['final_dose_rec'].value_counts()
        print(dose_recs_grouped)
        dose_recs_grouped.to_csv(f"{filepath}/final_dose_recs.csv")

    def save_metrics(self, filepath):
        self.dose_counts_frame.to_csv(f"{filepath}/dose_counts.csv")
        self.metrics_frame.to_csv(f"{filepath}/timestep_metrics.csv")
        overall_metrics_frame = self.grouped_metrics_frame[['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']]
        overall_metrics_frame.to_csv(f"{filepath}/overall_metrics.csv")
        
        print(overall_metrics_frame)
        for subgroup_idx in range(self.num_subgroups):
            print(f"Subgroup: {subgroup_idx}")
            self.subgroup_predictions[subgroup_idx].to_csv(f"{filepath}/{subgroup_idx}_predictions.csv")
            print(self.subgroup_predictions[subgroup_idx])

        print(self.final_dose_rec)
        self.final_dose_rec.to_csv(f"{filepath}/final_dose_rec.csv")

def calculate_utility(tox_means, eff_means, tox_thre, eff_thre, tox_weight, eff_weight):
    tox_term = (tox_means - tox_thre) ** 2
    eff_term = (eff_means - eff_thre) ** 2
    tox_term[tox_means > tox_thre] = 0.
    eff_term[eff_means < eff_thre] = 0.
    return (tox_weight * tox_term) + (eff_weight * eff_term)

def calculate_utility_thall(tox_probs, eff_probs, tox_thre, eff_thre, p_param):
    tox_term = (tox_probs / tox_thre) ** p_param
    eff_term = ((1. - eff_probs) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities
 
def get_bernoulli_confidence_region(posterior_latent_dist, likelihood_model, num_samples):
    samples = posterior_latent_dist.sample_n(num_samples)
    likelihood_samples = likelihood_model(samples)
    lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
    upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
    mean = likelihood_samples.mean.mean(axis=0)
    variance = likelihood_samples.mean.var(axis=0)
    return mean, lower, upper, variance

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

def sample_one_posterior(runner, num_subgroups, x_test, use_gpu):
    y_samples = PosteriorPrediction(num_subgroups, len(x_test))
    for subgroup_idx in range(num_subgroups):
        test_task_indices = torch.LongTensor(np.repeat(subgroup_idx, len(x_test)))
        _, post_observed = runner.predict(x_test, test_task_indices, use_gpu)
        y_samples.set_variables(subgroup_idx, post_observed.mean[np.random.randint(0, 10), :].cpu().numpy())
    return y_samples

def select_dose_from_sample(num_doses, max_dose, tox_mean, eff_mean, x_mask):
    ## Select ideal dose for subgroup
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    print(f"Available doses: {available_doses_mask}")

    safe_doses_mask = tox_mean[x_mask] <= dose_scenario.toxicity_threshold
    gt_threshold = np.where(tox_mean[x_mask] > dose_scenario.toxicity_threshold)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    print(f"Dose set: {safe_doses_mask}")

    eff_mean_masked = eff_mean[x_mask]
    eff_mean_masked[~dose_set_mask] = -np.inf
    max_eff = eff_mean_masked.max()
    selected_dose = np.where(eff_mean_masked == max_eff)[0][-1]

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    return selected_dose

def select_dose_confidence(num_doses, max_dose, tox_mean, tox_upper,
                           eff_upper, eff_lower, x_mask, beta_param):
    ## Select ideal dose for subgroup
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    print(f"Available doses: {available_doses_mask}")

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)

    safe_doses_mask = tox_ucb[x_mask] <= dose_scenario.toxicity_threshold
    gt_threshold = np.where(tox_ucb[x_mask] > dose_scenario.toxicity_threshold)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    print(f"Dose set: {safe_doses_mask}")

    ## Select dose with efficacy at widest confidence interval
    eff_intervals = eff_upper - eff_lower
    dose_eff_intervals = eff_intervals[x_mask]
    dose_eff_intervals[~dose_set_mask] = -np.inf
    max_eff_interval = dose_eff_intervals.max()
    selected_dose = np.where(dose_eff_intervals == max_eff_interval)[0][-1]
   

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    return selected_dose, tox_ucb, eff_intervals


def select_dose_expander(num_doses, max_dose, tox_mean, tox_upper,
                         eff_upper, eff_lower, x_mask, beta_param):
    ## Select ideal dose for subgroup
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    print(f"Available doses: {available_doses_mask}")

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)

    safe_doses_mask = tox_ucb[x_mask] <= dose_scenario.toxicity_threshold
    gt_threshold = np.where(tox_ucb[x_mask] > dose_scenario.toxicity_threshold)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    print(f"Dose set: {safe_doses_mask}")

    ## Select dose with efficacy at widest confidence interval
    eff_intervals = eff_upper - eff_lower
    dose_eff_intervals = eff_intervals[x_mask]
    dose_eff_intervals[~dose_set_mask] = -np.inf
    max_eff_interval = dose_eff_intervals.max()
    selected_dose = np.where(dose_eff_intervals == max_eff_interval)[0][-1]
   

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    return selected_dose, tox_ucb, eff_intervals


def select_dose_3(num_doses, tox_thre, tox_outcomes, max_dose, tox_mean, tox_upper,
                  x_mask, beta_param):
    # Expand to next dose unless there have been more than 1 toxic events
    if len(tox_outcomes) > 0:
        if tox_outcomes.sum() > 1:
            selected_dose = max_dose
        else:
            selected_dose = max_dose + 1
    else:
        selected_dose = max_dose
    
    # If at max dose, select highest safe dose
    tox_ucb = 0.
    if selected_dose >= num_doses:
        print("At max")
        # Find UCB for toxicity posteriors to determine safe set
        tox_conf_interval = tox_upper - tox_mean
        tox_ucb = tox_mean + (beta_param * tox_conf_interval)

        safe_doses_mask = tox_ucb[x_mask] <= tox_thre 
        gt_threshold = np.where(tox_ucb[x_mask] > tox_thre )[0]

        if gt_threshold.size:
            first_idx_above_threshold = gt_threshold[0]
            safe_doses_mask[first_idx_above_threshold:] = False
        print(f"Safe doses: {safe_doses_mask}")

        # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
        if safe_doses_mask.sum() == 0:
            selected_dose = 0
        else:
            ## Select largest safe dose
            selected_dose = np.where(safe_doses_mask == True)[0][-1]

    print(f"Tox outcomes: {tox_outcomes}")
    print(f"Max dose: {max_dose}")
    print(f"Selected dose: {selected_dose}")
    return selected_dose, tox_ucb

def select_dose_increasing(num_doses, tox_thre,
                           max_dose, tox_mean, tox_upper,
                           x_mask, beta_param):
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    print(f"Available doses: {available_doses_mask}")

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)

    safe_doses_mask = tox_ucb[x_mask] <= tox_thre
    gt_threshold = np.where(tox_ucb[x_mask] > tox_thre)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    print(f"Dose set: {safe_doses_mask}")

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    else:
        ## Select largest safe dose
        selected_dose = np.where(dose_set_mask == True)[0][-1]
    return selected_dose, tox_ucb

def select_dose_confidence_and_increasing(num_doses, max_dose, tox_mean, tox_upper,
                                          tox_lower, x_mask, beta_param, use_lcb=False):
    ## Select ideal dose for subgroup
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    # print(f"Available doses: {available_doses_mask}")

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)
    tox_lcb = tox_mean - (beta_param * (tox_mean - tox_lower))

    tox_acqui = tox_ucb
    if use_lcb:
        tox_acqui = tox_lcb
    
    safe_doses_mask = tox_acqui[x_mask] <= dose_scenario.toxicity_threshold
    gt_threshold = np.where(tox_acqui[x_mask] > dose_scenario.toxicity_threshold)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    # print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    
    # Always include previously included doses
    previous_doses_mask = np.isin(np.arange(num_doses), np.arange(max_dose + 1))
    dose_set_mask = np.logical_or(dose_set_mask, previous_doses_mask)
    # print(f"Dose set: {dose_set_mask}")

    ## Select dose with toxicity at widest confidence interval
    tox_intervals = tox_upper - tox_lower
    dose_tox_intervals = tox_intervals[x_mask]
    dose_tox_intervals[~dose_set_mask] = -np.inf
    max_tox_interval = dose_tox_intervals.max()

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    else:
        # Select largest safe dose
        selected_dose = np.where(dose_set_mask == True)[0][-1]
        # print(f"Selected dose: {selected_dose}")
        # If this dose has already been seen, instead select based on widest confidence interval
        if selected_dose <= max_dose:
            # print("Selecting based on confidence")
            selected_dose = np.where(dose_tox_intervals == max_tox_interval)[0][-1]

    return selected_dose, tox_acqui, dose_set_mask


def select_dose_separated(dose_set_mask, tox_mean,
                          eff_mean, eff_variance, x_mask,
                          tox_thre, eff_thre, tox_weight, eff_weight,
                          use_utility=False):

    ## Select optimal dose using EI of efficacy posteriors
    tradeoff_param = 0.01
    eff_stdev = np.sqrt(eff_variance)
    eff_opt = eff_mean.max()
    improvement = eff_mean - eff_opt - tradeoff_param
    z_val = improvement / eff_stdev

    eff_ei = (improvement * scipy.stats.norm.cdf(z_val)) + (eff_stdev * scipy.stats.norm.pdf(z_val))
    eff_ei[eff_stdev == 0.] = 0.

    if not use_utility:
        dose_eff_ei = eff_ei[x_mask]
        dose_eff_ei[~dose_set_mask] = -np.inf
        max_eff_ei = dose_eff_ei.max()
        selected_dose = np.where(dose_eff_ei == max_eff_ei)[0][-1]
    else:
        ## Select optimal dose using utility
        utilities = calculate_utility(tox_mean[x_mask], eff_mean[x_mask], tox_thre, eff_thre,
                                    tox_weight, eff_weight)
        utilities[~dose_set_mask] = -np.inf
        max_utility = utilities.max()
        selected_dose = np.where(utilities == max_utility)[0][-1]

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    return selected_dose, 0., eff_ei


def select_dose(num_doses, max_dose, tox_mean, tox_upper,
                tox_lower,
                eff_mean, eff_variance, x_mask, beta_param,
                tox_thre, eff_thre, tox_weight, eff_weight,
                use_utility=False, use_lcb=False):
    ## Select ideal dose for subgroup
    # Available doses are current max dose idx + 1
    available_dose_indices = np.arange(max_dose + 2)
    available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)
    # print(f"Available doses: {available_doses_mask}")

    # Find UCB for toxicity posteriors to determine safe set
    tox_conf_interval = tox_upper - tox_mean
    tox_ucb = tox_mean + (beta_param * tox_conf_interval)
    tox_lcb = tox_mean - (beta_param * (tox_mean - tox_lower))
    tox_acqui = tox_ucb
    if use_lcb:
        tox_acqui = tox_lcb

    safe_doses_mask = tox_acqui[x_mask] <= tox_thre
    gt_threshold = np.where(tox_acqui[x_mask] > tox_thre)[0]

    if gt_threshold.size:
        first_idx_above_threshold = gt_threshold[0]
        safe_doses_mask[first_idx_above_threshold:] = False
    # print(f"Safe doses: {safe_doses_mask}")

    dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
    # print(f"Dose set: {safe_doses_mask}")

    ## Select optimal dose using EI of efficacy posteriors
    tradeoff_param = 0.01
    eff_stdev = np.sqrt(eff_variance)
    eff_opt = eff_mean.max()
    improvement = eff_mean - eff_opt - tradeoff_param
    z_val = improvement / eff_stdev

    eff_ei = (improvement * scipy.stats.norm.cdf(z_val)) + (eff_stdev * scipy.stats.norm.pdf(z_val))
    eff_ei[eff_stdev == 0.] = 0.

    if not use_utility:
        dose_eff_ei = eff_ei[x_mask]
        dose_eff_ei[~dose_set_mask] = -np.inf
        max_eff_ei = dose_eff_ei.max()
        selected_dose = np.where(dose_eff_ei == max_eff_ei)[0][-1]
    else:
        ## Select optimal dose using utility
        utilities = calculate_utility(tox_mean[x_mask], eff_mean[x_mask], tox_thre, eff_thre,
                                    tox_weight, eff_weight)
        utilities[~dose_set_mask] = -np.inf
        max_utility = utilities.max()
        selected_dose = np.where(utilities == max_utility)[0][-1]

    # If all doses are unsafe, return first dose. If this happens enough times, stop trial.
    if dose_set_mask.sum() == 0:
        selected_dose = 0
    return selected_dose, tox_acqui, eff_ei

def select_final_dose(num_subgroups, num_doses, dose_labels, x_test,
                      tox_posteriors, eff_posteriors, 
                      tox_thre, eff_thre, tox_weight, eff_weight, final_beta_param, use_thall=False):
    x_mask = np.isin(x_test, dose_labels)
    dose_rec = np.ones(num_subgroups) * num_doses
    final_utilities = np.empty((num_subgroups, num_doses))
        
    # Select dose with highest utility that is below toxicity threshold
    for subgroup_idx in range(num_subgroups):
        # Determine safe dose set on posterior means
        tox_mean = tox_posteriors.mean[subgroup_idx, :][x_mask]
        tox_upper = tox_posteriors.upper[subgroup_idx, :][x_mask]
        eff_mean = eff_posteriors.mean[subgroup_idx, :][x_mask]

        tox_conf_interval = tox_upper - tox_mean
        tox_ucb = tox_mean + (final_beta_param * tox_conf_interval)
        dose_set_mask = tox_ucb <= tox_thre

        gt_threshold = np.where(tox_ucb > tox_thre)[0]
        if gt_threshold.size:
            first_idx_above_threshold = gt_threshold[0]
            dose_set_mask[first_idx_above_threshold:] = False
        print(f"Dose set: {dose_set_mask}")

        # Calculate utilities
        utilities = calculate_utility(tox_mean, eff_mean, tox_thre, eff_thre,
                                      tox_weight, eff_weight)
        thall_utilities = calculate_utility_thall(tox_mean, eff_mean,
                                                  dose_scenario.toxicity_threshold,
                                                  dose_scenario.efficacy_threshold,
                                                  dose_scenario.p_param)

        eff_mean[~dose_set_mask] = -np.inf
        utilities[~dose_set_mask] = -np.inf
        thall_utilities[~dose_set_mask] = -np.inf

        max_eff = eff_mean.max()
        max_eff_idx = np.where(eff_mean == max_eff)[0][-1]

        if use_thall:
            max_utility = thall_utilities.max()
            max_util_idx = np.where(thall_utilities == max_utility)[0][-1]
            final_utilities[subgroup_idx, :] = thall_utilities
            
        else:
            max_utility = utilities.max()
            max_util_idx = np.where(utilities == max_utility)[0][-1]
            final_utilities[subgroup_idx, :] = utilities

        # If recommended dose is above eff threshold assign this dose, else assign no dose.
        if eff_mean[max_util_idx] >= eff_thre: 
            dose_rec[subgroup_idx] = max_util_idx
        
        else: # Try assiging dose with highest efficacy in safe range. Else assign no dose
            print("Best dose has efficacy that is too low. Try assigning dose w/ highest efficacy.")
            if max_eff >= eff_thre:
                dose_rec[subgroup_idx] = max_eff_idx
    
    print(f"Final doses: {dose_rec}")
    return dose_rec, final_utilities
    

def online_dose_finding(filepath, dose_scenario, patient_scenario,
                        num_samples, num_latents, beta_param, learning_rate,
                        tox_lengthscale_init, eff_lengthscale_init,
                        tox_mean_init, eff_mean_init,
                        final_beta_param, sampling_timesteps, increase_beta_param,
                        use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu):
    plots_filepath = f"{filepath}/gp_plots"
    latent_plots_filepath = f"{filepath}/latent_gp_plots"
    if not os.path.exists(filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(plots_filepath):
        os.makedirs(plots_filepath)
    if not os.path.exists(latent_plots_filepath):
        os.makedirs(latent_plots_filepath)

    # Hyperparameters
    num_epochs = 300
    num_confidence_samples = 1000
    cohort_size = 3

    dose_labels = dose_scenario.dose_labels
    num_subgroups = patient_scenario.num_subgroups
    num_tasks = patient_scenario.num_subgroups
    patients = patient_scenario.generate_samples(num_samples)
    num_doses = dose_scenario.num_doses

    timestep = 0
    max_doses = np.zeros(num_subgroups)

    # Initialize arrays
    selected_doses = []
    selected_dose_values = []
    tox_outcomes = []
    eff_outcomes = []

    # Initialize first cohort with lowest dose
    for subgroup_idx in patients[timestep: timestep + cohort_size]:
        selected_dose = 0
        selected_dose_val = dose_labels[selected_dose]
        tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
        eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

        selected_doses.append(selected_dose)
        selected_dose_values.append(selected_dose_val)
        tox_outcomes.append(tox_outcome)
        eff_outcomes.append(eff_outcome)
    
    # Construct test data (works for all models)
    np_x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
    np_x_test = np.unique(np_x_test)
    np.sort(np_x_test)
    
    x_mask = np.isin(np_x_test, dose_labels)
    markevery = np.arange(len(np_x_test))[x_mask].tolist()
    timestep += cohort_size

    dose_set_mask = np.empty((num_subgroups, num_doses), dtype=np.bool)
    while timestep < num_samples:
        print(f"Timestep: {timestep}")
        current_beta_param = np.float32(beta_param)
        if increase_beta_param:
            param_val = beta_param - (2 * beta_param) + (0.05 * (timestep/cohort_size))
            if param_val < beta_param:
                current_beta_param = np.float32(param_val)
        print(f"Beta param: {current_beta_param}")

        # Construct training data
        task_indices = torch.LongTensor(patients[:timestep])
        x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
        y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
        y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)
        x_test = torch.tensor(np_x_test, dtype=torch.float32)

        # Train model
        tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, tox_lengthscale_init,
                                                   tox_mean_init)
        tox_runner.train(x_train, y_tox_train, task_indices,
                         num_epochs, learning_rate, use_gpu, set_lmc=True)

        eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                                   dose_labels, eff_lengthscale_init,
                                                   eff_mean_init)
        eff_runner.train(x_train, y_eff_train, task_indices,
                         num_epochs, learning_rate, use_gpu, set_lmc=True)

        # Get model predictions
        y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)
        y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                                x_test, num_confidence_samples, use_gpu)

        # # Sample one func
        # y_tox_sample = sample_one_posterior(tox_runner, patient_scenario.num_subgroups,
        #                                     x_test, use_gpu)
        # y_eff_sample = sample_one_posterior(eff_runner, patient_scenario.num_subgroups,
        #                                     x_test, use_gpu)

        # Get dose selections
        selected_dose_by_subgroup = np.empty(num_subgroups, dtype=np.int32)
        tox_acqui_funcs = np.empty((num_subgroups, len(np_x_test)))
        eff_acqui_funcs = np.empty((num_subgroups, len(np_x_test)))
        util_func = np.empty((num_subgroups, len(np_x_test)))
        cohort_patients = patients[timestep: timestep + cohort_size]

        for subgroup_idx in range(patient_scenario.num_subgroups):
            if timestep <= sampling_timesteps:
                # print("Sampling one function")
                # selected_dose_by_subgroup[subgroup_idx]  = \
                # select_dose_from_sample(num_doses, max_doses[subgroup_idx],
                #                         y_tox_sample.mean[subgroup_idx, :],
                #                         y_eff_sample.mean[subgroup_idx, :], x_mask)
                # tox_acqui_funcs[subgroup_idx, :] = y_tox_sample.mean[subgroup_idx, :]
                # eff_acqui_funcs[subgroup_idx, :] = y_eff_sample.mean[subgroup_idx, :]

                # print("Select based on eff confidence interval.")
                # selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :],\
                # eff_acqui_funcs[subgroup_idx, :] = select_dose_confidence(num_doses, max_doses[subgroup_idx],
                #                                                           y_tox_posteriors.mean[subgroup_idx, :],
                #                                                           y_tox_posteriors.upper[subgroup_idx, :],
                #                                                           y_eff_posteriors.upper[subgroup_idx, :],
                #                                                           y_eff_posteriors.lower[subgroup_idx, :],
                #                                                           x_mask, current_beta_param)

                selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :], dose_set_mask[subgroup_idx, :] \
                 = select_dose_confidence_and_increasing(num_doses, max_doses[subgroup_idx],
                                                                                         y_tox_posteriors.mean[subgroup_idx, :], 
                                                                                         y_tox_posteriors.upper[subgroup_idx, :],
                                                                                         y_tox_posteriors.lower[subgroup_idx, :],
                                                                                         x_mask, current_beta_param,
                                                                                         use_lcb=use_lcb_init)
                eff_acqui_funcs[subgroup_idx, :] = 0.

                # print("Select highest possible safe dose.")
                # cohort_patients_mask = cohort_patients == subgroup_idx
                # selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :] = \
                #     select_dose_3(num_doses, dose_scenario.toxicity_threshold, np.array(tox_outcomes[timestep - cohort_size: timestep])[cohort_patients_mask],
                #                   max_doses[subgroup_idx], y_tox_posteriors.mean[subgroup_idx, :],
                #                   y_tox_posteriors.upper[subgroup_idx, :], x_mask, current_beta_param)
                # eff_acqui_funcs[subgroup_idx, :] = 0.

                # selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :] = \
                #     select_dose_increasing(num_doses, dose_scenario.toxicity_threshold,
                #                            max_doses[subgroup_idx],
                #                            y_tox_posteriors.mean[subgroup_idx, :],
                #                            y_tox_posteriors.upper[subgroup_idx, :],
                #                            x_mask, current_beta_param)
                # eff_acqui_funcs[subgroup_idx, :] = 0.

            else:
                selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :],\
                eff_acqui_funcs[subgroup_idx, :] = \
                select_dose(num_doses, max_doses[subgroup_idx],
                            y_tox_posteriors.mean[subgroup_idx, :],
                            y_tox_posteriors.upper[subgroup_idx, :],
                            y_tox_posteriors.lower[subgroup_idx, :],
                            y_eff_posteriors.mean[subgroup_idx, :],
                            y_eff_posteriors.variance[subgroup_idx, :], x_mask,
                            current_beta_param, dose_scenario.toxicity_threshold,
                            dose_scenario.efficacy_threshold,
                            dose_scenario.tox_weight, dose_scenario.eff_weight,
                            use_utility=use_utility, use_lcb=use_lcb_exp)

                # selected_dose_by_subgroup[subgroup_idx], tox_acqui_funcs[subgroup_idx, :],\
                # eff_acqui_funcs[subgroup_idx, :] = \
                # select_dose_separated(dose_set_mask[subgroup_idx, :],
                #                       y_tox_posteriors.mean[subgroup_idx, :],
                #                       y_eff_posteriors.mean[subgroup_idx, :],
                #                       y_eff_posteriors.variance[subgroup_idx, :], x_mask,
                #                       dose_scenario.toxicity_threshold,
                #                       dose_scenario.efficacy_threshold,
                #                       dose_scenario.tox_weight,
                #                       dose_scenario.eff_weight,
                #                       use_utility=False)

            # Calculate utility
            if use_thall:
                util_func[subgroup_idx, :] = calculate_utility_thall(y_tox_posteriors.mean[subgroup_idx, :],
                                                            y_eff_posteriors.mean[subgroup_idx, :],
                                                            dose_scenario.toxicity_threshold,
                                                            dose_scenario.efficacy_threshold,
                                                            dose_scenario.p_param)
            else:
                util_func[subgroup_idx, :] = calculate_utility(y_tox_posteriors.mean[subgroup_idx, :],
                                                            y_eff_posteriors.mean[subgroup_idx, :],
                                                            dose_scenario.toxicity_threshold,
                                                            dose_scenario.efficacy_threshold,
                                                            dose_scenario.tox_weight,
                                                            dose_scenario.eff_weight)

        print(f"Selected dose by subgroup: {selected_dose_by_subgroup}")

        # Assign dose to each patient in cohort, update outcomes.
        for subgroup_idx in cohort_patients:
            selected_dose = selected_dose_by_subgroup[subgroup_idx]
            selected_dose_val = dose_labels[selected_dose]
            tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
            eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

            selected_doses.append(selected_dose)
            selected_dose_values.append(selected_dose_val)
            tox_outcomes.append(tox_outcome)
            eff_outcomes.append(eff_outcome)

            # Update max doses
            if selected_dose > max_doses[subgroup_idx]:
                max_doses[subgroup_idx] = selected_dose
        
        print(f"Selected doses: {selected_doses[timestep:]}")
        print(f"Max doses: {max_doses}")

        # Plot timestep
        plot_gp_timestep(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                         num_subgroups, np_x_test, y_tox_posteriors, y_eff_posteriors, 
                         tox_acqui_funcs, eff_acqui_funcs, util_func, selected_dose_by_subgroup,
                         markevery, x_mask, f"{plots_filepath}/timestep{timestep}")

        plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients[:timestep],
                patient_scenario.num_subgroups, np_x_test, y_tox_latents, y_eff_latents,
                util_func, selected_dose_by_subgroup, markevery, x_mask,
                f"{latent_plots_filepath}/timestep{timestep}")
        
        timestep += cohort_size

    # Train model final time
    # Construct training data
    task_indices = torch.LongTensor(patients)
    x_train = torch.tensor(selected_dose_values, dtype=torch.float32)
    y_tox_train = torch.tensor(tox_outcomes, dtype=torch.float32)
    y_eff_train = torch.tensor(eff_outcomes, dtype=torch.float32)

    # Train model
    tox_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                               dose_labels, tox_lengthscale_init,
                                               tox_mean_init)
    tox_runner.train(x_train, y_tox_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=set_lmc)

    eff_runner = MultitaskClassificationRunner(num_latents, num_tasks,
                                            dose_labels, eff_lengthscale_init,
                                            eff_mean_init)
    eff_runner.train(x_train, y_eff_train, task_indices,
                     num_epochs, learning_rate, use_gpu, set_lmc=set_lmc)
    
    # Get model predictions
    y_tox_posteriors, y_tox_latents = get_model_predictions(tox_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)
    y_eff_posteriors, y_eff_latents = get_model_predictions(eff_runner, patient_scenario.num_subgroups,
                                                            x_test, num_confidence_samples, use_gpu)

    # Select final dose
    final_selected_doses, final_utilities = select_final_dose(num_subgroups, num_doses, 
                                                              dose_labels, x_test, y_tox_posteriors,
                                                              y_eff_posteriors,
                                                              dose_scenario.toxicity_threshold,
                                                              dose_scenario.efficacy_threshold,
                                                              dose_scenario.tox_weight,
                                                              dose_scenario.eff_weight, final_beta_param,
                                                              use_thall)

    experiment_metrics = DoseExperimentMetrics(dose_scenario, patients, selected_doses,
                                               tox_outcomes, eff_outcomes, y_tox_posteriors.mean[:, x_mask],
                                               y_tox_posteriors.upper[:, x_mask], y_eff_posteriors.mean[:, x_mask],
                                               final_selected_doses, final_utilities)
    experiment_metrics.save_metrics(filepath)
    
    # Calculate utilities
    util_func = np.empty((num_subgroups, len(np_x_test)))
    for subgroup_idx in range(num_subgroups):
        if use_thall:
            util_func[subgroup_idx, :] = calculate_utility_thall(y_tox_posteriors.mean[subgroup_idx, :],
                                                        y_eff_posteriors.mean[subgroup_idx, :],
                                                        dose_scenario.toxicity_threshold,
                                                        dose_scenario.efficacy_threshold,
                                                        dose_scenario.p_param)
        else:
            util_func[subgroup_idx, :] = calculate_utility(y_tox_posteriors.mean[subgroup_idx, :],
                                                        y_eff_posteriors.mean[subgroup_idx, :],
                                                        dose_scenario.toxicity_threshold,
                                                        dose_scenario.efficacy_threshold,
                                                        dose_scenario.tox_weight, dose_scenario.eff_weight)

    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            np_x_test, y_tox_posteriors, y_eff_posteriors, util_func, final_selected_doses,
            markevery, x_mask, f"{filepath}/final_gp_plot")
    plot_gp(dose_scenario, x_train, y_tox_train, y_eff_train, patients, num_subgroups,
            np_x_test, y_tox_latents, y_eff_latents, util_func, final_selected_doses,
            markevery, x_mask, f"{filepath}/final_gp_latents_plot")
    
    return experiment_metrics, y_tox_posteriors, y_eff_posteriors, util_func


def online_dose_finding_trials(results_dir, num_trials, dose_scenario, patient_scenario,
                               num_samples, num_latents, beta_param, learning_rate, 
                               tox_lengthscale_init, eff_lengthscale_init, 
                               tox_mean_init, eff_mean_init, final_beta_param,
                               sampling_timesteps, increase_beta_param, use_utility, use_lcb_init,
                               use_lcb_exp, set_lmc, use_thall, use_gpu):
    metrics = []
    x_true = dose_scenario.dose_labels.astype(np.float32)
    x_test = np.concatenate([np.arange(x_true.min(), x_true.max(), 0.05, dtype=np.float32), x_true])
    x_test = np.unique(x_test)
    np.sort(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    tox_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    eff_means = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    util_vals = np.empty((num_trials, patient_scenario.num_subgroups, x_test.shape[0]))
    for trial in range(num_trials):
        print(f"Trial {trial}")
        filepath = f"{results_dir}/trial{trial}"
        trial_metrics, tox_posteriors, eff_posteriors, util_func = online_dose_finding(
            filepath, dose_scenario, patient_scenario, num_samples, num_latents, beta_param,
            learning_rate, tox_lengthscale_init, eff_lengthscale_init,
            tox_mean_init, eff_mean_init,
            final_beta_param, sampling_timesteps, increase_beta_param,
            use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
        metrics.append(trial_metrics)

        for subgroup_idx in range(patient_scenario.num_subgroups):
            tox_means[trial, subgroup_idx, :] = tox_posteriors.mean[subgroup_idx, :]
            eff_means[trial, subgroup_idx, :] = eff_posteriors.mean[subgroup_idx, :]
            util_vals[trial, subgroup_idx, :] = util_func[subgroup_idx, :]

        with open(f"{results_dir}/trial{trial}/tox_means.npy", 'wb') as f:
            np.save(f, tox_means[trial, :, :])
        with open(f"{results_dir}/trial{trial}/eff_means.npy", 'wb') as f:
            np.save(f, eff_means[trial, :, :])
        with open(f"{results_dir}/trial{trial}/util_vals.npy", 'wb') as f:
            np.save(f, util_vals[trial, :, :])
    
    DoseExperimentMetrics.save_merged_metrics(metrics, results_dir)
    x_mask = np.isin(x_test, x_true)
    markevery = np.arange(len(x_test))[x_mask].tolist()
    plot_gp_trials(tox_means, eff_means, util_vals, x_test,
                   dose_scenario.dose_labels, dose_scenario.toxicity_probs,
                   dose_scenario.efficacy_probs,
                   patient_scenario.num_subgroups, markevery, results_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="File path name")
    parser.add_argument("--scenario", type=int, help="Dose scenario")
    parser.add_argument("--beta_param", type=float, help="Beta param for toxicity confidence interval.")
    parser.add_argument("--sampling_timesteps", type=int, help="Number of timesteps to run burn-in procedure.")
    parser.add_argument("--tox_lengthscale", type=float, help="Tox GP Kernel lengthscale.")
    parser.add_argument("--eff_lengthscale", type=float, help="Eff GP Kernel lengthscale")
    parser.add_argument("--tox_mean", type=float, help="Tox mean constant")
    parser.add_argument("--eff_mean", type=float, help="Eff mean constant")
    parser.add_argument("--num_latents", type=int, help="Number of GP latents")
    parser.add_argument("--use_lcb_init", action="store_true", help="Use LCB for initial stage.")
    parser.add_argument("--use_lcb_exp", action="store_true", help="Use LCB for exploitation stage.")
    parser.add_argument("--set_lmc", action="store_true", help="Fix LMC coeffs for final model if true.")
    parser.add_argument("--use_thall", action="store_true", help="Use Thall utility.")
    parser.add_argument("--run_one", action="store_true", help="Run just one iteration")
    args = parser.parse_args()

    scenarios = {
        1: DoseFindingScenarios.paper_example_1(),
        2: DoseFindingScenarios.paper_example_2(),
        3: DoseFindingScenarios.paper_example_3(),
        4: DoseFindingScenarios.paper_example_4(),
        5: DoseFindingScenarios.paper_example_5(),
        6: DoseFindingScenarios.paper_example_6(),
        7: DoseFindingScenarios.paper_example_7(),
        8: DoseFindingScenarios.paper_example_8(),
        9: DoseFindingScenarios.paper_example_9(),
        10: DoseFindingScenarios.paper_example_10(),
        11: DoseFindingScenarios.paper_example_11(),
        12: DoseFindingScenarios.paper_example_12(),
        13: DoseFindingScenarios.paper_example_13(),
        14: DoseFindingScenarios.paper_example_14(),
        15: DoseFindingScenarios.paper_example_15(),
        16: DoseFindingScenarios.paper_example_16(),
        17: DoseFindingScenarios.paper_example_17(),
        18: DoseFindingScenarios.paper_example_18()
    }

    filepath = args.filepath
    scenario = args.scenario
    beta_param = args.beta_param
    sampling_timesteps = args.sampling_timesteps
    tox_lengthscale = args.tox_lengthscale
    eff_lengthscale = args.eff_lengthscale
    tox_mean = args.tox_mean
    eff_mean = args.eff_mean
    num_latents = args.num_latents
    use_lcb_init = args.use_lcb_init
    use_lcb_exp = args.use_lcb_exp
    set_lmc = args.set_lmc
    use_thall = args.use_thall
    run_one = args.run_one
    return filepath, scenarios[scenario], beta_param, sampling_timesteps,\
           tox_lengthscale, eff_lengthscale, tox_mean, eff_mean, num_latents, use_lcb_init, use_lcb_exp, set_lmc, use_thall, run_one


if __name__ == "__main__":
    filepath, dose_scenario, beta_param, sampling_timesteps, tox_lengthscale_init, \
        eff_lengthscale_init, tox_mean_init, eff_mean_init, num_latents, use_lcb_init, use_lcb_exp, set_lmc, use_thall, run_one = parse_args()

    increase_beta_param = False
    use_utility = False
    use_gpu = False
    num_trials = 100
    num_samples = 51
    learning_rate = 0.01
    final_beta_param = 0.

    # dose_scenario = DoseFindingScenarios.paper_example_1()
    patient_scenario = TrialPopulationScenarios.equal_population(2)

    # Calculate tox_mean_init and eff_mean_init based on thresholds
    # tox_mean_init = NormalDist().inv_cdf(dose_scenario.toxicity_threshold)
    # eff_mean_init = NormalDist().inv_cdf(dose_scenario.efficacy_threshold)

    print(f"Tox mean: {tox_mean_init}")
    print(f"Eff mean: {eff_mean_init}")
    if run_one:
        online_dose_finding(filepath, dose_scenario, patient_scenario,
                            num_samples, num_latents, beta_param, learning_rate,
                            tox_lengthscale_init, eff_lengthscale_init,
                            tox_mean_init, eff_mean_init,
                            final_beta_param, sampling_timesteps, increase_beta_param,
                            use_utility, use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
    else:
        online_dose_finding_trials(filepath, num_trials, dose_scenario,
                                patient_scenario, num_samples, num_latents,
                                beta_param, learning_rate, 
                                tox_lengthscale_init, eff_lengthscale_init, 
                                tox_mean_init, eff_mean_init, final_beta_param,
                                sampling_timesteps, increase_beta_param, use_utility,
                                use_lcb_init, use_lcb_exp, set_lmc, use_thall, use_gpu)
