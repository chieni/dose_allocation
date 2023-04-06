import os
import numpy as np
import pandas as pd

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from thall import calculate_dose_utility_thall
 

def three_plus_three(dose_scenario, patients, num_samples, num_subgroups):
    dose_labels = dose_scenario.dose_labels
    num_doses = dose_scenario.num_doses

    timestep = 0
    cohort_size = 3
    additional_cohort = False
    tox_break = False

    selected_doses = []
    selected_dose_values = []
    tox_outcomes = []
    eff_outcomes = []

    for subgroup_idx in patients[timestep: timestep+cohort_size]:
        selected_dose = 0
        selected_dose_val = dose_labels[selected_dose]
        tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
        eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

        selected_doses.append(selected_dose)
        selected_dose_values.append(selected_dose_val)
        tox_outcomes.append(tox_outcome)
        eff_outcomes.append(eff_outcome)

    timestep += cohort_size

    while timestep < num_samples:
        num_tox_outcomes = np.array(tox_outcomes[timestep-cohort_size: timestep]).sum()

        if num_tox_outcomes == 0:
            additional_cohort = False
            selected_dose += 1
            if selected_dose == num_doses:
                break
        elif num_tox_outcomes == 1:
            if additional_cohort:
                tox_break = True
                break
            additional_cohort = True
        else:
            tox_break = True
            break

        for subgroup_idx in patients[timestep: timestep+cohort_size]:
            selected_dose_val = dose_labels[selected_dose]
            tox_outcome = dose_scenario.sample_toxicity_event(selected_dose, subgroup_idx)
            eff_outcome = dose_scenario.sample_efficacy_event(selected_dose, subgroup_idx)

            selected_doses.append(selected_dose)
            selected_dose_values.append(selected_dose_val)
            tox_outcomes.append(tox_outcome)
            eff_outcomes.append(eff_outcome)

        timestep += cohort_size

    recommended_dose = None
    if tox_break:
        recommended_dose = selected_dose - 1
    else:
        recommended_dose = selected_dose

    return np.repeat(recommended_dose, num_subgroups), np.array(selected_doses), tox_outcomes, eff_outcomes

def three_plus_three_trials(num_trials, dose_scenario, num_samples, patient_scenario):
    num_subgroups = patient_scenario.num_subgroups
    optimal_doses = dose_scenario.optimal_doses
    metrics_list = []

    trial = 0
    while trial < num_trials:
        patients = patient_scenario.generate_samples(num_samples)
        final_selected_doses, selected_doses, tox_outcomes, eff_outcomes = three_plus_three(dose_scenario, patients,
                                                                num_samples, num_subgroups)
        num_samples_treated = len(selected_doses)
        patients_seen = patients[:num_samples_treated]

        optimal_dose_per_sample = np.array([optimal_doses[subgroup_idx] for subgroup_idx in patients_seen])

        dose_error = (selected_doses != optimal_dose_per_sample).astype(np.float32)

        final_dose_error = (final_selected_doses != optimal_doses).astype(np.float32)
        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients_seen)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, patients_seen)])
        safety_violations = np.array(selected_tox_probs > dose_scenario.toxicity_threshold, dtype=np.int32)
        utilities = calculate_dose_utility_thall(selected_tox_probs, selected_eff_probs,
                                            dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
                                            dose_scenario.p_param)
                  
        metrics_frame = pd.DataFrame({
            'subgroup_idx': patients_seen,
            'tox_outcome': tox_outcomes,
            'eff_outcome': eff_outcomes,
            'selected_dose': selected_doses,
            'dose_error': dose_error,
            'utilities': utilities.astype(np.float32),
            'safety_violations': safety_violations
        })

        grouped_metrics_frame = metrics_frame.groupby(['subgroup_idx']).mean()
        if grouped_metrics_frame.shape[0] == 1:
            continue
        total_frame = pd.DataFrame(metrics_frame.mean()).T
        total_frame = total_frame.rename(index={0: 'overall'})
        grouped_metrics_frame = pd.concat([grouped_metrics_frame, total_frame])
        grouped_metrics_frame['final_dose_error'] = np.concatenate([final_dose_error, [final_dose_error.sum() / num_subgroups]])
        grouped_metrics_frame['final_selected_dose'] = np.concatenate([final_selected_doses, [np.nan]])      
        metrics_list.append(grouped_metrics_frame)
        trial += 1

    frame = pd.concat([df for df in metrics_list])
    grouped_frame = frame.groupby(frame.index)
    mean_frame = grouped_frame.mean()
    var_frame = grouped_frame.var()

    return mean_frame


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

dose_error_dict = {}
safety_dict = {}
utility_dict = {}
results_foldername = "results/three_baseline_ratios/"
if not os.path.exists(results_foldername):
    os.makedirs(results_foldername)


# num_samples = 51
# for scenario_idx, dose_scenario in scenarios.items():
#     mean_frame = three_plus_three_trials(1000, dose_scenario)
#     utility_dict[f"scenario{scenario_idx}"] = mean_frame['utilities'].values.tolist()
#     safety_dict[f"scenario{scenario_idx}"] = mean_frame['safety_violations'].values.tolist()
#     dose_error_dict[f"scenario{scenario_idx}"] = mean_frame['final_dose_error'].values.tolist()


# patient_scenario = TrialPopulationScenarios.equal_population(2)
# test_sample_nums = np.arange(51, 546, 9)
# scenario_idx = 9 # scenario 9
# dose_scenario = scenarios[scenario_idx]
# for idx, num_samples in enumerate(test_sample_nums):
#     print(num_samples)
#     mean_frame = three_plus_three_trials(100, dose_scenario, num_samples, patient_scenario)
#     utility_dict[num_samples] = mean_frame['utilities'].values.tolist()
#     safety_dict[num_samples] = mean_frame['safety_violations'].values.tolist()
#     dose_error_dict[num_samples] = mean_frame['final_dose_error'].values.tolist()


patient_ratios = np.arange(0.1, 1.0, 0.05)
scenario_idx = 11
dose_scenario = scenarios[scenario_idx]
num_samples = 201
num_trials = 100
for patient_ratio in patient_ratios:
    print(patient_ratio)
    patient_scenario = TrialPopulationScenarios.skewed_dual_population(patient_ratio)
    print(patient_scenario.arrival_rate)
    mean_frame = three_plus_three_trials(num_trials, dose_scenario, num_samples, patient_scenario)
    utility_dict[patient_ratio] = mean_frame['utilities'].values.tolist()
    safety_dict[patient_ratio] = mean_frame['safety_violations'].values.tolist()
    dose_error_dict[patient_ratio] = mean_frame['final_dose_error'].values.tolist()


utility_frame = pd.DataFrame(utility_dict)
utility_frame.index = ['0', '1', 'overall']
safety_frame = pd.DataFrame(safety_dict)
safety_frame.index = ['0', '1', 'overall']
dose_error_frame = pd.DataFrame(dose_error_dict)
safety_frame.index = ['0', '1', 'overall']


utility_frame.to_csv(f"{results_foldername}/utility.csv")
dose_error_frame.to_csv(f"{results_foldername}/final_ose_error.csv")
safety_frame.to_csv(f"{results_foldername}/safety.csv")