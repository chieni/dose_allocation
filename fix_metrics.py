import numpy as np
import pandas as pd

from data_generation import DoseFindingScenarios, TrialPopulationScenarios


def calculate_utility_thall(tox_probs, eff_probs, tox_thre, eff_thre, p_param):
    tox_term = (tox_probs / tox_thre) ** p_param
    eff_term = ((1. - eff_probs) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities

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

filepath = "results/gp_continuous2"
num_trials = 100
num_subgroups = 2
dose_scenario = DoseFindingScenarios.continuous_subgroups_example_2()

metrics_list = []

for trial_idx in range(num_trials):
    # trial_metrics = pd.read_csv(f"{results_folder}/scenario{scenario_idx}/trial{trial_idx}/timestep_metrics.csv")
    grouped_metrics_frame = pd.read_csv(f"{filepath}/trial{trial_idx}/overall_metrics.csv")
    final_dose_rec_frame = pd.read_csv(f"{filepath}/trial{trial_idx}/final_dose_rec.csv")
    final_dose_diff = final_dose_rec_frame['final_dose_diff']
    final_dose_diff_abs = final_dose_rec_frame['final_dose_diff_abs']
    grouped_metrics_frame['final_dose_diff'] =  np.concatenate([final_dose_diff, [sum(final_dose_diff) / num_subgroups]])
    grouped_metrics_frame['final_dose_diff_abs'] = np.concatenate([final_dose_diff_abs, [sum(final_dose_diff_abs) / num_subgroups]])

    metrics_list.append(grouped_metrics_frame)
    # selected_doses = trial_metrics['selected_dose']
    # subgroup_indices = trial_metrics['subgroup_idx']
    # selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
    #                                for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
    # selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
    #                                for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])

    # utilities = calculate_utility_thall(selected_tox_probs, selected_eff_probs,
    #                                     dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
    #                                     dose_scenario.p_param)
    
    # for subgroup_idx in range(num_subgroups):
    #     subgroup_mask = (subgroup_indices == subgroup_idx)
    #     trial_utilities[subgroup_idx, trial_idx] = utilities[subgroup_mask].mean()

    # trial_utilities[num_subgroups, trial_idx] = utilities.mean()

frame = pd.concat([df for df in metrics_list])
grouped_frame = frame.groupby(frame.index)
mean_frame = grouped_frame.mean()
var_frame = grouped_frame.var()
mean_frame = mean_frame[['tox_outcome', 'eff_outcome', 'utility', 'safety_violations',
                            'dose_error', 'final_dose_error', 'final_dose_diff', 'final_dose_diff_abs']]
var_frame = var_frame[['tox_outcome', 'eff_outcome', 'utility', 'safety_violations',
                        'dose_error', 'final_dose_error', 'final_dose_diff', 'final_dose_diff_abs']]

print(mean_frame)
print(var_frame)
mean_frame.to_csv(f"{filepath}/final_metric_means.csv")
var_frame.to_csv(f"{filepath}/final_metric_var.csv")

# dose_counts = pd.concat([df.dose_counts_frame for df in metrics_list])
# dose_counts_mean = dose_counts.groupby(level=0).mean()
# print(dose_counts_mean)
# dose_counts_mean.to_csv(f"{filepath}/all_dose_counts.csv")

# dose_recs = pd.concat([df.final_dose_rec for df in metrics_list])
# dose_recs_grouped = dose_recs.groupby('subgroup_idx')['final_dose_rec'].value_counts()
# print(dose_recs_grouped)
# dose_recs_grouped.to_csv(f"{filepath}/final_dose_recs.csv")

