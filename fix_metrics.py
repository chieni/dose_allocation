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

results_folder = "results/nineteenth_pass"
num_trials = 100
num_subgroups = 2
scenario_utilities = np.empty((num_subgroups + 1, 19))

for scenario_idx, dose_scenario in scenarios.items():
    trial_utilities = np.empty((num_subgroups + 1, num_trials))

    for trial_idx in range(num_trials):
        trial_metrics = pd.read_csv(f"{results_folder}/scenario{scenario_idx}/trial{trial_idx}/timestep_metrics.csv")
        selected_doses = trial_metrics['selected_dose']
        subgroup_indices = trial_metrics['subgroup_idx']
        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])

        utilities = calculate_utility_thall(selected_tox_probs, selected_eff_probs,
                                            dose_scenario.toxicity_threshold, dose_scenario.efficacy_threshold,
                                            dose_scenario.p_param)
        
        for subgroup_idx in range(num_subgroups):
            subgroup_mask = (subgroup_indices == subgroup_idx)
            trial_utilities[subgroup_idx, trial_idx] = utilities[subgroup_mask].mean()

        trial_utilities[num_subgroups, trial_idx] = utilities.mean()

    scenario_utilities[:, scenario_idx-1] = trial_utilities.mean(axis=1)
print(scenario_utilities)

total_frame = pd.DataFrame(scenario_utilities, columns=[f"scenario{key}" for key in scenarios.keys()], index=['0', '1', 'overall'])
total_frame.to_csv("results/nineteenth_pass/thall_utilities.csv")

