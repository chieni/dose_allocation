import pandas as pd
import numpy as np

from data_generation import DoseFindingScenarios, TrialPopulationScenarios



def calculate_dose_utility_thall(tox_values, eff_values, tox_thre, eff_thre, p_param):
    tox_term = (tox_values / tox_thre) ** p_param
    eff_term = ((1. - eff_values) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities
    
def calculate_utility(tox_means, eff_means, tox_thre, eff_thre, tox_weight, eff_weight):
    tox_term = (tox_means - tox_thre) ** 2
    eff_term = (eff_means - eff_thre) ** 2
    tox_term[tox_means > tox_thre] = 0.
    eff_term[eff_means < eff_thre] = 0.
    return (tox_weight * tox_term) + (eff_weight * eff_term)

filepath = "results/exp21"
dose_scenario = DoseFindingScenarios.subgroups_example_1()
patient_scenario = TrialPopulationScenarios.equal_population(2)
num_trials = 100
num_subgroups = 2
final_beta_param = 0.

num_doses = dose_scenario.num_doses
dose_rec = np.ones((num_trials, num_subgroups)) * num_doses
dose_err = np.empty((num_trials, num_subgroups))
optimal_doses = dose_scenario.optimal_doses

for trial in range(num_trials):
    for subgroup_idx in range(num_subgroups):
        frame = pd.read_csv(f"{filepath}/trial{trial}/{subgroup_idx}_predictions.csv")
        tox_means = frame['tox_predicted'].values
        tox_upper = frame['tox_upper'].values
        eff_means = frame['eff_predicted'].values
        utilities = frame['final_utilities'].values

        tox_conf_interval = tox_upper - tox_means
        tox_ucb = tox_means + (final_beta_param * tox_conf_interval)
        dose_set_mask = tox_ucb <= dose_scenario.toxicity_threshold

        gt_threshold = np.where(tox_ucb > dose_scenario.toxicity_threshold)[0]
        if gt_threshold.size:
            first_idx_above_threshold = gt_threshold[0]
            dose_set_mask[first_idx_above_threshold:] = False
    

        utilities = calculate_utility(tox_means, eff_means, dose_scenario.toxicity_threshold,
                                      dose_scenario.efficacy_threshold,
                                      1, 4)
        thall_utilities = calculate_dose_utility_thall(tox_means, eff_means,
                                                       dose_scenario.toxicity_threshold,
                                                       dose_scenario.efficacy_threshold,
                                                       2.1)

        eff_means[~dose_set_mask] = -np.inf
        utilities[~dose_set_mask] = -np.inf
        thall_utilities[~dose_set_mask] = -np.inf

        # max_eff = eff_means.max()
        # if max_eff >= dose_scenario.efficacy_threshold:
        #     max_eff_idx = np.where(eff_means == max_eff)[0][-1]
        #     dose_rec[trial, subgroup_idx] = max_eff_idx

        max_util = utilities.max()
        max_util_idx = np.where(utilities == max_util)[0][-1]
        if eff_means[max_util_idx] >= dose_scenario.efficacy_threshold:
            dose_rec[trial, subgroup_idx] = max_util_idx

        # max_util = thall_utilities.max()
        # max_util_idx = np.where(thall_utilities == max_util)[0][-1]
        # dose_rec[trial, subgroup_idx] = max_util_idx
        # if eff_means[max_util_idx] >= dose_scenario.efficacy_threshold:
        #     dose_rec[trial, subgroup_idx] = max_util_idx

        
    dose_err[trial, :] = (dose_rec[trial, :] != optimal_doses).astype(np.float32)

for subgroup_idx in range(num_subgroups):
    values, counts = np.unique(dose_rec[:, subgroup_idx], return_counts=True)
    print(f"Subgroup {subgroup_idx}")
    print(values)
    print(counts)

print(dose_err.mean(axis=0))



    