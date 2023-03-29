import os
import argparse
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pymc as pm
from sklearn.preprocessing import scale
import seaborn as sns
from data_generation import TrialPopulationScenarios, DoseFindingScenarios
from thall import calculate_dose_utility_thall


class CRM:
    def __init__(self, num_patients):
        self.num_patients = num_patients

    def tangent_model(dose_label, alpha):
        return ((np.tanh(dose_label) + 1.) / 2.) ** alpha
    
    def logistic_model(dose_label, alpha, beta):
        '''
        alpha: intercept
        beta: 
        '''
        return (np.exp(alpha + dose_label * np.exp(beta)))\
                / (1. + np.exp(alpha + dose_label * np.exp(beta)))
    
    def init_tangent_labels(p_true_val, a0):
        x = (p_true_val ** (1. / a0) * 2. - 1.)
        return 1./2. * np.log((1. + x)/(1. - x))
    
    def init_logistic_labels(p_true_val, a0, b0):
        return (np.log(p_true_val / (1. - p_true_val)) - a0) / np.exp(b0)

    def from_posterior(param, samples):
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 100)
        samples = samples[0]
        y = stats.gaussian_kde(samples)(x)

        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return pm.Interpolated(param, x, y)
    
    def plot_dose_toxicity_curve(dose_scenario, dose_skeleton_labels, alpha, num_subgroups, plot_filename):
        sns.set_theme()
        p_true = dose_scenario.toxicity_probs
        model_toxicities = CRM.tangent_model(dose_skeleton_labels, alpha)
        plt.plot(dose_scenario.dose_labels, model_toxicities, label="CRM")
        for subgroup_idx in range(num_subgroups):
            plt.plot(dose_scenario.dose_labels, p_true[subgroup_idx, :], label=f"Subgroup {subgroup_idx} True")
        plt.legend()
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    def jitter_dose_skeleton(self, dose_value):
        new_val = -1
        while new_val <= 0:
            new_val = dose_value + np.random.uniform(-0.15, 0.15)
        return new_val

    def run_trial(self, dose_scenario, patients, num_subgroups, out_foldername, add_jitter):
        if not os.path.exists(out_foldername):
            os.makedirs(out_foldername)

        cohort_size = 3
        max_dose = 0
        timestep = 0

        a0 = 1 / np.e
        dose_skeleton = np.mean(dose_scenario.toxicity_probs, axis=0)
        if add_jitter:
            for dose_idx in range(dose_skeleton.shape[0]):
                dose_skeleton[dose_idx] = self.jitter_dose_skeleton(dose_skeleton[dose_idx])
            print(dose_skeleton)
        dose_labels = CRM.init_tangent_labels(dose_skeleton, a0)
        patients = patients.astype(int)

        # Assign first patient to lowest dose level
        selected_doses = []
        selected_dose_values = []
        tox_outcomes = []
        eff_outcomes = []

        for subgroup_idx in patients[:cohort_size].astype(int):
            selected_dose = 0
            selected_dose_val = dose_labels[selected_dose]
            tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
            eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

            selected_doses.append(selected_dose)
            selected_dose_values.append(selected_dose_val)
            tox_outcomes.append(tox_outcome)
            eff_outcomes.append(eff_outcome)

        X = np.array(selected_dose_values).astype(np.float32)
        Y = np.array(tox_outcomes).astype(np.float32)

        model = pm.Model()
        with model:
            # Prior of parameters
            alpha = pm.Gamma("alpha", 1, 1)

            # Expected value of outcome: dose-toxicity model
            toxicity_prob = CRM.tangent_model(X, alpha)

            # Likelihood (sampling dist) of observations
            Y_obs = pm.Bernoulli("Y_obs", p=toxicity_prob, observed=Y)

            # Draw posterior samples
            trace = pm.sample(5000, chains=1)
            alpha_trace = trace.posterior['alpha']
            current_alpha_mean = np.mean(alpha_trace).item()

        timestep += cohort_size

        while timestep < self.num_patients:
            predicted_toxicities = CRM.tangent_model(dose_labels, current_alpha_mean)
            print(current_alpha_mean)
            print(predicted_toxicities)

            selected_dose = np.abs(np.array(predicted_toxicities) - dose_scenario.toxicity_threshold).argmin()
            if selected_dose > max_dose + 1:
                selected_dose = max_dose + 1
                max_dose = selected_dose
            print(f"Selected dose: {selected_dose}")

            cohort_subgroup_indices = patients[timestep: timestep+cohort_size].astype(int)
            print(f"curr_s: {cohort_subgroup_indices}")

            for subgroup_idx in cohort_subgroup_indices:
                selected_dose_val = dose_labels[selected_dose]
                tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
                eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

                selected_doses.append(selected_dose)
                selected_dose_values.append(selected_dose_val)
                tox_outcomes.append(tox_outcome)
                eff_outcomes.append(eff_outcome)

            X = np.array(selected_dose_values).astype(np.float32)
            Y = np.array(tox_outcomes).astype(np.float32)
            print(X, Y)

            model = pm.Model()
            with model:
                # Priors are posteriors from previous iteration
                alpha = CRM.from_posterior("alpha", alpha_trace)

                # Expected value of outcome: dose-toxicity model
                toxicity_prob = CRM.tangent_model(X, alpha)

                # Likelihood (sampling dist) of observations
                Y_obs = pm.Bernoulli("Y_obs", p=toxicity_prob, observed=Y)

                # Draw posterior samples
                trace = pm.sample(5000, chains=1)
                alpha_trace = trace.posterior['alpha']
                current_alpha_mean = np.mean(alpha_trace).item()

            timestep += cohort_size

        final_model_toxicities = CRM.tangent_model(dose_labels, current_alpha_mean)
        print(current_alpha_mean)
        print(final_model_toxicities)
        print(f"True tox: {dose_scenario.toxicity_probs}")

        final_selected_doses = np.empty(num_subgroups)
        for subgroup_idx in range(num_subgroups):
            safety_mask = (final_model_toxicities <= dose_scenario.toxicity_threshold)
            final_selected_doses[subgroup_idx] = np.argmax(final_model_toxicities * safety_mask)
        
        optimal_doses = dose_scenario.optimal_doses[:num_subgroups]
        optimal_dose_per_sample = np.array([optimal_doses[subgroup_idx] for subgroup_idx in patients])
        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)
        final_dose_error = (final_selected_doses != optimal_doses).astype(np.float32)

        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients)])
        safety_violations = np.array(selected_tox_probs > dose_scenario.toxicity_threshold, dtype=np.int32)
        utilities = calculate_dose_utility_thall(selected_tox_probs, selected_eff_probs,
                                            dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
                                            dose_scenario.p_param)

        metrics_frame = pd.DataFrame({
            'subgroup_idx': patients,
            'tox_outcome': tox_outcomes,
            'eff_outcome': eff_outcomes,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utilities': utilities.astype(np.float32),
            'safety_violations': safety_violations
        })

        metrics_frame.to_csv(f"{out_foldername}/timestep_metrics.csv")
        grouped_metrics_frame = metrics_frame.groupby(['subgroup_idx']).mean()
        total_frame = pd.DataFrame(metrics_frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        grouped_metrics_frame = pd.concat([grouped_metrics_frame, total_frame])
        grouped_metrics_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / num_subgroups]])
        grouped_metrics_frame['final_selected_dose'] = np.concatenate([final_selected_doses, [np.nan]])
        grouped_metrics_frame.to_csv(f"{out_foldername}/grouped_metrics.csv")
        
        CRM.plot_dose_toxicity_curve(dose_scenario, dose_labels, current_alpha_mean, num_subgroups, f"{out_foldername}/toxicity_plot.png")
        return grouped_metrics_frame, final_selected_doses


    def run_subgroups_trial(self, dose_scenario, patients, num_subgroups, out_foldername):
        if not os.path.exists(out_foldername):
            os.makedirs(out_foldername)

        cohort_size = 3
        max_dose = 0
        timestep = 0

        a0 = 1 / np.e
        dose_skeleton = np.mean(dose_scenario.toxicity_probs, axis=0)
        dose_labels = CRM.init_tangent_labels(dose_skeleton, a0)
        import pdb; pdb.set_trace()
        patients = patients.astype(int)

        # Assign first patient to lowest dose level
        selected_doses = []
        selected_dose_values = []
        tox_outcomes = []
        eff_outcomes = []

        for subgroup_idx in patients[:cohort_size].astype(int):
            selected_dose = 0
            selected_dose_val = dose_labels[selected_dose]
            tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
            eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

            selected_doses.append(selected_dose)
            selected_dose_values.append(selected_dose_val)
            tox_outcomes.append(tox_outcome)
            eff_outcomes.append(eff_outcome)

        X = np.array(selected_dose_values).astype(np.float32)
        Y = np.array(tox_outcomes).astype(np.float32)

        model = pm.Model()
        with model:
            # Prior of parameters
            alpha = pm.Gamma("alpha", 1, 1)

            # Expected value of outcome: dose-toxicity model
            toxicity_prob = CRM.tangent_model(X, alpha)

            # Likelihood (sampling dist) of observations
            Y_obs = pm.Bernoulli("Y_obs", p=toxicity_prob, observed=Y)

            # Draw posterior samples
            trace = pm.sample(5000, chains=1)
            alpha_trace = trace.posterior['alpha']
            current_alpha_mean = np.mean(alpha_trace).item()

        timestep += cohort_size

        while timestep < self.num_patients:
            predicted_toxicities = CRM.tangent_model(dose_labels, current_alpha_mean)
            print(current_alpha_mean)
            print(predicted_toxicities)

            selected_dose = np.abs(np.array(predicted_toxicities) - dose_scenario.toxicity_threshold).argmin()
            if selected_dose > max_dose + 1:
                selected_dose = max_dose + 1
                max_dose = selected_dose
            print(f"Selected dose: {selected_dose}")

            cohort_subgroup_indices = patients[timestep: timestep+cohort_size].astype(int)
            print(f"curr_s: {cohort_subgroup_indices}")

            for subgroup_idx in cohort_subgroup_indices:
                selected_dose_val = dose_labels[selected_dose]
                tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
                eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

                selected_doses.append(selected_dose)
                selected_dose_values.append(selected_dose_val)
                tox_outcomes.append(tox_outcome)
                eff_outcomes.append(eff_outcome)

            X = np.array(selected_dose_values).astype(np.float32)
            Y = np.array(tox_outcomes).astype(np.float32)
            print(X, Y)

            model = pm.Model()
            with model:
                # Priors are posteriors from previous iteration
                alpha = CRM.from_posterior("alpha", alpha_trace)

                # Expected value of outcome: dose-toxicity model
                toxicity_prob = CRM.tangent_model(X, alpha)

                # Likelihood (sampling dist) of observations
                Y_obs = pm.Bernoulli("Y_obs", p=toxicity_prob, observed=Y)

                # Draw posterior samples
                trace = pm.sample(5000, chains=1)
                alpha_trace = trace.posterior['alpha']
                current_alpha_mean = np.mean(alpha_trace).item()

            timestep += cohort_size

        final_model_toxicities = CRM.tangent_model(dose_labels, current_alpha_mean)
        print(current_alpha_mean)
        print(final_model_toxicities)
        print(f"True tox: {dose_scenario.toxicity_probs}")

        final_selected_doses = np.empty(num_subgroups)
        for subgroup_idx in range(num_subgroups):
            safety_mask = (final_model_toxicities <= dose_scenario.toxicity_threshold)
            final_selected_doses[subgroup_idx] = np.argmax(final_model_toxicities * safety_mask)
        
        optimal_doses = dose_scenario.optimal_doses[:num_subgroups]
        optimal_dose_per_sample = np.array([optimal_doses[subgroup_idx] for subgroup_idx in patients])
        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)
        final_dose_error = (final_selected_doses != optimal_doses).astype(np.float32)

        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients)])
        safety_violations = np.array(selected_tox_probs > dose_scenario.toxicity_threshold, dtype=np.int32)
        utilities = calculate_dose_utility_thall(selected_tox_probs, selected_eff_probs,
                                            dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
                                            dose_scenario.p_param)

        metrics_frame = pd.DataFrame({
            'subgroup_idx': patients,
            'tox_outcome': tox_outcomes,
            'eff_outcome': eff_outcomes,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utilities': utilities.astype(np.float32),
            'safety_violations': safety_violations
        })

        metrics_frame.to_csv(f"{out_foldername}/timestep_metrics.csv")
        grouped_metrics_frame = metrics_frame.groupby(['subgroup_idx']).mean()
        total_frame = pd.DataFrame(metrics_frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        grouped_metrics_frame = pd.concat([grouped_metrics_frame, total_frame])
        grouped_metrics_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / num_subgroups]])
        grouped_metrics_frame['final_selected_dose'] = np.concatenate([final_selected_doses, [np.nan]])
        grouped_metrics_frame.to_csv(f"{out_foldername}/grouped_metrics.csv")
        
        CRM.plot_dose_toxicity_curve(dose_scenario, dose_labels, current_alpha_mean, num_subgroups, f"{out_foldername}/toxicity_plot.png")
        return grouped_metrics_frame, final_selected_doses


def run_one_trial(results_foldername, dose_scenario, num_patients, add_jitter):
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)
    num_subgroups = 2

    # patient_scenario = TrialPopulationScenarios.lee_trial_population()
    # dose_scenario = DoseFindingScenarios.lee_synthetic_example()

    # dose_scenario = DoseFindingScenarios.oquigley_model_example()
    # patient_scenario = TrialPopulationScenarios.homogenous_population()
    patient_scenario = TrialPopulationScenarios.equal_population(num_subgroups)
    crm = CRM(num_patients)
    patients = patient_scenario.generate_samples(num_patients)
    crm.run_trial(dose_scenario, patients, patient_scenario.num_subgroups, results_foldername, add_jitter)


def run_trials(results_foldername, dose_scenario, num_patients, num_trials, add_jitter):
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)
    num_subgroups = 2
    patient_scenario = TrialPopulationScenarios.equal_population(num_subgroups)

    crm = CRM(num_patients)
    metrics_list = []
    final_selected_doses = np.empty((num_trials, num_subgroups))
    for trial_idx in range(num_trials):
        patients = patient_scenario.generate_samples(num_patients)
        trial_metrics, trial_selected_doses = crm.run_trial(dose_scenario, patients, patient_scenario.num_subgroups,
                                                            f"{results_foldername}/trial{trial_idx}", add_jitter)
        metrics_list.append(trial_metrics)
        final_selected_doses[trial_idx] = trial_selected_doses
    
    frame = pd.concat([df for df in metrics_list])
    grouped_frame = frame.groupby(frame.index)
    mean_frame = grouped_frame.mean()
    var_frame = grouped_frame.var()
    mean_frame.to_csv(f"{results_foldername}/overall_metrics.csv")

    doses_frame = pd.DataFrame(final_selected_doses)
    doses_frame.to_csv(f"{results_foldername}/final_selected_doses.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="File path name")
    parser.add_argument("--scenario", type=int, help="Dose scenario")
    parser.add_argument("--num_samples", type=int, help="Number of samples.")
    parser.add_argument("--num_trials", type=int, help="Number of trials.")
    parser.add_argument("--run_one", action="store_true", help="Run just one iteration")
    parser.add_argument("--add_jitter", action="store_true", help="Jitter dose skeleton labels")
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
        18: DoseFindingScenarios.paper_example_18(),
        19: DoseFindingScenarios.paper_example_19()
    }

    filepath = args.filepath
    scenario = scenarios[args.scenario]
    num_samples = args.num_samples
    num_trials = args.num_trials
    run_one = args.run_one
    add_jitter = args.add_jitter
    return filepath, scenario, num_samples, num_trials, run_one, add_jitter

if __name__ == "__main__":

    # python crm.py --filepath results/crm_exp6 --scenario 1 --num_samples 51 --num_trials 100
    filepath, scenario, num_samples, num_trials, run_one, add_jitter = parse_args()

    if run_one:
        run_one_trial(filepath, scenario, num_samples, add_jitter)
    else:
        run_trials(filepath, scenario, num_samples, num_trials, add_jitter)


