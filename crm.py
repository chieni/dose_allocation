import arviz as az
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pymc as pm
from sklearn.preprocessing import scale
import seaborn as sns


class CRM:
    def __init__(self, num_patients, num_doses):
        self.num_patients = num_patients
        self.num_doses = num_doses
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

    def gen_patients(self, arr_rate):
        '''
        Generates all patients for an experiment of length T
        '''
        # Arrival proportion of each subgroup. If arrive_rate = [5, 4, 3],
        # arrive_dist = [5/12, 4/12, 3/12]
        arrive_sum = sum(arr_rate)
        arrive_dist = [rate/arrive_sum for rate in arr_rate]
        arrive_dist.insert(0, 0)

        # [0, 5/12, 9/12, 12/12]
        arrive_dist_bins = np.cumsum(arrive_dist)

        # Random numbers between 0 and 1 in an array of shape (1, T)
        patients_gen = np.random.rand(self.num_patients)
        patients = np.digitize(patients_gen, arrive_dist_bins) - 1
        return patients

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
    
    def plot_dose_toxicity_curve(dose_labels, p_true, alpha):
        sns.set_theme()
        model_toxicities = CRM.tangent_model(dose_labels, alpha)
        frame = pd.DataFrame({'dose_labels': dose_labels,
                              'model': model_toxicities,
                              'true': p_true.mean(axis=0)})
        frame = pd.melt(frame, id_vars=['dose_labels'], var_name='toxicity', value_name='toxicity_value')
        sns.lineplot(data=frame, x='dose_labels', y='toxicity_value', hue='toxicity', markers=True)

    def run_trial(self, dose_labels, patients, p_true, tox_thre):
        cohort_size = 3
        max_dose = 0
        timestep = 0
        curr_s = patients[:cohort_size]

        # Assign first patient to lowest dose level
        self.allocated_doses[:cohort_size] = 0
        X = np.array([dose_labels[0] for patient_idx in range(cohort_size)])

        # Sample toxicity
        Y = np.array([int(np.random.rand() <= p_true[subgroup_idx, self.allocated_doses[timestep]]) for subgroup_idx in curr_s])

        model = pm.Model()
        with model:
            # Prior of parameters
            alpha = pm.Gamma("alpha", 1, 1)

            # Expected value of outcome: dose-toxicity model
            toxicity_prob = CRM.tangent_model(X, alpha)

            # Likelihood (sampling dist) of observations
            Y_obs = pm.Bernoulli("Y_obs", toxicity_prob, observed=Y)

            # Draw 1000 posterior samples
            trace = pm.sample(25000, cores=1, target_accept=0.95)
            print(az.summary(trace, round_to=2))

            current_alpha_dist = trace.posterior['alpha']
            expected_toxicities = CRM.tangent_model(dose_labels, 1. / np.e)
            print(expected_toxicities)
            CRM.plot_dose_toxicity_curve(dose_labels, p_true, current_alpha_dist.values.mean())
            plt.show()

        timestep += cohort_size

        while timestep < self.num_patients:
            # Sample more data
            curr_s = patients[timestep - cohort_size:timestep]
            print(f"curr_s: {curr_s}")

            # If skipping doses is allowed, assign to the dose whose expected toxicity
            # under the posterior dist is closest to the threshold.
            expected_toxicities = CRM.tangent_model(dose_labels, current_alpha_dist.values.mean())
            selected_dose = np.abs(np.array(expected_toxicities) - tox_thre).argmin()
            if selected_dose > max_dose + 1:
                selected_dose = max_dose + 1
                max_dose = selected_dose
            print(expected_toxicities)
            print(f"Selected dose: {selected_dose}")
            self.allocated_doses[:cohort_size] = selected_dose
            X = np.array([dose_labels[selected_dose] for patient_idx in range(cohort_size)])
            # Sample toxicity
            Y = np.array([int(np.random.rand() <= p_true[subgroup_idx, self.allocated_doses[timestep]]) for subgroup_idx in curr_s])
            print(X, Y)
            model = pm.Model()
            with model:
                # Priors are posteriors from previous iteration
                alpha = CRM.from_posterior("alpha", trace.posterior["alpha"])

                # Expected value of outcome: dose-toxicity model
                toxicity_prob = CRM.tangent_model(X, alpha)

                # Likelihood (sampling dist) of observations
                Y_obs = pm.Bernoulli("Y_obs", toxicity_prob, observed=Y)

                # draw 10000 posterior samples
                trace = pm.sample(25000, cores=1, target_accept=0.95)
                    
                print(az.summary(trace, round_to=2))
                current_alpha_dist = trace.posterior['alpha']
                print(current_alpha_dist.values.mean())
                CRM.plot_dose_toxicity_curve(dose_labels, p_true, current_alpha_dist.values.mean())
                plt.show()

            timestep += cohort_size

def main():
    tox_thre = 0.35
    # toxicity
    p_true = np.array([[0.01, 0.01, 0.05, 0.15, 0.20, 0.45],
                    [0.01, 0.05, 0.15, 0.20, 0.45, 0.60],
                    [0.01, 0.05, 0.15, 0.20, 0.45, 0.60]])
    arr_rate = [5, 4, 3]
    crm = CRM(30, 5)
    dose_skeleton = np.mean(p_true, axis=0)
    dose_labels = CRM.init_tangent_labels(dose_skeleton, 1. / np.e)
    patients = crm.gen_patients(arr_rate)
    crm.run_trial(dose_labels, patients, p_true, tox_thre)



if __name__ == "__main__":
    main()