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

def test_select_final_dose_method(filepath, dose_scenario, tox_weight, eff_weight, use_thall=False):
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
                                        tox_weight, eff_weight)
            thall_utilities = calculate_dose_utility_thall(tox_means, eff_means,
                                                        dose_scenario.toxicity_threshold,
                                                        dose_scenario.efficacy_threshold,
                                                        2.1)

            eff_means[~dose_set_mask] = -np.inf
            utilities[~dose_set_mask] = -np.inf
            thall_utilities[~dose_set_mask] = -np.inf

            max_eff = eff_means.max()
            max_eff_idx = np.where(eff_means == max_eff)[0][-1]

            max_util = utilities.max()
            max_util_idx = np.where(utilities == max_util)[0][-1]

            max_util_thall = thall_utilities.max()
            max_util_thall_idx = np.where(thall_utilities == max_util_thall)[0][-1]
            
            # if max_eff >= dose_scenario.efficacy_threshold:
            #     dose_rec[trial, subgroup_idx] = max_eff_idx
            if not use_thall:
                if eff_means[max_util_idx] >= dose_scenario.efficacy_threshold:
                    dose_rec[trial, subgroup_idx] = max_util_idx
                else:
                    # Try highest eff in safe range
                    if max_eff >= dose_scenario.efficacy_threshold:
                        dose_rec[trial, subgroup_idx] = max_eff_idx
            else:
                if eff_means[max_util_thall_idx] >= dose_scenario.efficacy_threshold:
                    dose_rec[trial, subgroup_idx] = max_util_thall_idx

            
        dose_err[trial, :] = (dose_rec[trial, :] != optimal_doses).astype(np.float32)

    # for subgroup_idx in range(num_subgroups):
    #     values, counts = np.unique(dose_rec[:, subgroup_idx], return_counts=True)
    #     print(f"Subgroup {subgroup_idx}")
    #     print(values)
    #     print(counts)

    print(dose_err.mean(axis=0))
    return dose_err.mean(axis=0)



# filepath = "results/fifth_pass/scenario18"
# dose_scenario = DoseFindingScenarios.paper_example_18()
# tox_weight = 1
# eff_weight = dose_scenario.toxicity_threshold/dose_scenario.efficacy_threshold

# test_select_final_dose_method(filepath, dose_scenario, tox_weight, eff_weight)


filepath = "results/fifth_pass"
use_thall = True
num_subgroups = 2

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

frame = pd.DataFrame(index=np.arange(num_subgroups))
for idx, scenario in scenarios.items():
    sub_filepath = f"{filepath}/scenario{idx}"
    tox_weight = 1
    eff_weight = scenario.toxicity_threshold/scenario.efficacy_threshold
    print(f"Eff weight: {eff_weight}")
    frame[f"scenario{idx}"] = test_select_final_dose_method(sub_filepath, scenario, tox_weight, eff_weight, use_thall)

if use_thall:
    frame.to_csv(f"{filepath}/thall_final_dose_error.csv")
else:
    frame.to_csv(f"{filepath}/test_final_dose_error.csv")


