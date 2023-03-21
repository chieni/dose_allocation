import arviz as az
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pymc as pm
from sklearn.preprocessing import scale
import seaborn as sns
from data_generation import TrialPopulationScenarios, DoseFindingScenarios


class CRM:
    def __init__(self, num_patients):
        self.num_patients = num_patients
        self.allocated_doses = np.ones(self.num_patients, dtype=int) * -1

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
    
    def plot_dose_toxicity_curve(dose_labels, p_true, alpha, num_subgroups):
        sns.set_theme()
        model_toxicities = CRM.tangent_model(dose_labels, alpha)
        plt.plot(dose_labels, model_toxicities, 'b-', label="CRM")
        for subgroup_idx in range(num_subgroups):
            plt.plot(dose_labels, p_true[subgroup_idx, :], 'g-', label=f"Subgroup {subgroup_idx} True")
        plt.show()

    def run_trial(self, dose_scenario, patients, num_subgroups):
        cohort_size = 3
        max_dose = 0
        timestep = 0
        cohort_subgroup_indices = patients[:cohort_size].astype(int)

        # Assign first patient to lowest dose level
        self.allocated_doses[:cohort_size] = 0
        X = np.array([dose_scenario.dose_labels[0] for patient_idx in range(cohort_size)])

        # Sample toxicity
        Y = dose_scenario.generate_toxicity_data(self.allocated_doses[:cohort_size], cohort_subgroup_indices)

        model = pm.Model()
        with model:
            # Prior of parameters
            alpha = pm.Gamma("alpha", 1, 1)

            # Expected value of outcome: dose-toxicity model
            toxicity_prob = CRM.tangent_model(X, alpha)

            # Likelihood (sampling dist) of observations
            Y_obs = pm.Bernoulli("Y_obs", p=toxicity_prob, observed=Y)

            # Draw posterior samples
            # trace = pm.sample(20000, cores=1, target_accept=0.95)
            trace = pm.sample(5000, chains=1)
            alpha_trace = trace.posterior['alpha']
            current_alpha_mean = np.mean(alpha_trace).item()

            predicted_toxicities = CRM.tangent_model(dose_scenario.dose_labels, current_alpha_mean)
            print(predicted_toxicities)
            print(f"True tox: {dose_scenario.toxicity_probs}")

            # CRM.plot_dose_toxicity_curve(dose_scenario.dose_labels, dose_scenario.toxicity_probs, current_alpha_mean)
            # plt.show()

        timestep += cohort_size

        while timestep < self.num_patients:
            # Sample more data
            cohort_subgroup_indices = patients[timestep - cohort_size:timestep].astype(int)
            print(f"curr_s: {cohort_subgroup_indices}")

            predicted_toxicities = CRM.tangent_model(dose_scenario.dose_labels, current_alpha_mean)
            selected_dose = np.abs(np.array(predicted_toxicities) - dose_scenario.toxicity_threshold).argmin()
            if selected_dose > max_dose + 1:
                selected_dose = max_dose + 1
                max_dose = selected_dose
            print(predicted_toxicities)
            print(f"Selected dose: {selected_dose}")
            self.allocated_doses[timestep:timestep + cohort_size] = selected_dose

            X = np.array([dose_scenario.dose_labels[selected_dose] for patient_idx in range(cohort_size)])
            # Sample toxicity
            Y = dose_scenario.generate_toxicity_data(self.allocated_doses[timestep:timestep + cohort_size], cohort_subgroup_indices)
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
                # trace = pm.sample(20000, cores=1, target_accept=0.95)

                trace = pm.sample(5000, chains=1)
                alpha_trace = trace.posterior['alpha']
                current_alpha_mean = np.mean(alpha_trace).item()
                # print(az.summary(trace, round_to=2))

                print(current_alpha_mean)
                predicted_toxicities = CRM.tangent_model(dose_scenario.dose_labels, current_alpha_mean)
                print(predicted_toxicities)

                # CRM.plot_dose_toxicity_curve(dose_scenario.dose_labels, dose_scenario.toxicity_probs, current_alpha_mean)
                # plt.show()

            timestep += cohort_size
        final_model_toxicities = CRM.tangent_model(dose_scenario.dose_labels, current_alpha_mean)
        print(final_model_toxicities)
        print(f"True tox: {dose_scenario.toxicity_probs}")
        CRM.plot_dose_toxicity_curve(dose_scenario.dose_labels, dose_scenario.toxicity_probs, current_alpha_mean, num_subgroups)

def main():
    # patient_scenario = TrialPopulationScenarios.lee_trial_population()
    # dose_scenario = DoseFindingScenarios.lee_synthetic_example()

    # dose_scenario = DoseFindingScenarios.oquigley_model_example()
    dose_scenario = DoseFindingScenarios.paper_example_1()
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    num_patients = 12
    crm = CRM(num_patients)
    patients = patient_scenario.generate_samples(num_patients)
    crm.run_trial(dose_scenario, patients, patient_scenario.num_subgroups)



if __name__ == "__main__":
    main()