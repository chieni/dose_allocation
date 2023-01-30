import pandas as pd
import numpy as np
import torch

from data_generation import DoseFindingScenarios, TrialPopulationScenarios
from new_experiments import get_model_predictions, select_final_dose
from new_gp import MultitaskClassificationRunner


def calculate_dose_utility_thall(tox_values, eff_values, tox_thre, eff_thre, p_param):
    tox_term = (tox_values / tox_thre) ** p_param
    eff_term = ((1. - eff_values) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities
    
def calculate_utility(tox_means, eff_means, tox_thre, eff_thre, tox_weight, eff_weight):
    tox_term = (tox_means - tox_thre) ** 2
    eff_term = (eff_means - eff_thre) ** 2
    # tox_term[tox_means > tox_thre] = 0.
    # eff_term[eff_means < eff_thre] = 0.
    tox_term[tox_means > tox_thre] = -tox_term[tox_means > tox_thre]
    eff_term[eff_means < eff_thre] = 0.
    return (tox_weight * tox_term) + (eff_weight * eff_term)

def test_select_final_dose_method(filepath, dose_scenario, tox_weight, eff_weight, use_thall=False, retrain_model=False):
    patient_scenario = TrialPopulationScenarios.equal_population(2)
    num_trials = 100
    num_subgroups = int(2)
    final_beta_param = 0.
    num_latents = 3
    num_tasks = 2
    tox_lengthscale_init = 4.
    eff_lengthscale_init = 2.
    tox_mean_init = 0.
    eff_mean_init = 0.
    num_epochs = 300
    learning_rate = 0.01
    use_gpu = False
    set_lmc = False
    num_confidence_samples = 1000

    num_doses = int(dose_scenario.num_doses)
    dose_rec = np.ones((num_trials, num_subgroups)) * num_doses
    dose_err = np.empty((num_trials, num_subgroups))
    optimal_doses = dose_scenario.optimal_doses

    frames_list = []

    for trial in range(num_trials):
        # Calculate trial utilities
        trial_frame = pd.read_csv(f"{filepath}/trial{trial}/timestep_metrics.csv")
        subgroup_indices = trial_frame['subgroup_idx'].values
        selected_doses = trial_frame['selected_dose'].values

        selected_tox_probs = np.array([dose_scenario.get_toxicity_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        selected_eff_probs = np.array([dose_scenario.get_efficacy_prob(arm_idx, group_idx) \
                                       for arm_idx, group_idx in zip(selected_doses, subgroup_indices)])
        trial_utilities = calculate_utility(selected_tox_probs, selected_eff_probs,
                                      dose_scenario.toxicity_threshold,
                                      dose_scenario.efficacy_threshold,
                                      dose_scenario.tox_weight,
                                      dose_scenario.eff_weight)
        trial_thall_utilities = calculate_dose_utility_thall(selected_tox_probs, selected_eff_probs,
                                                             dose_scenario.toxicity_threshold,
                                                             dose_scenario.efficacy_threshold,
                                                             dose_scenario.p_param).astype(np.float32)
        
        util_frame = pd.DataFrame({'subgroup_idx': subgroup_indices,
                                   'utility': trial_utilities,
                                   'thall_utility': trial_thall_utilities})
        grouped_util_frame = util_frame.groupby(['subgroup_idx']).mean()
        frames_list.append(grouped_util_frame)

        max_doses = np.empty(num_subgroups)
        for subgroup_idx in range(num_subgroups):
            max_doses[subgroup_idx ] = selected_doses[subgroup_indices == subgroup_idx].max()

        if retrain_model:
            dose_labels = dose_scenario.dose_labels
            # Construct test data (works for all models)
            np_x_test = np.concatenate([np.arange(dose_labels.min(), dose_labels.max(), 0.05, dtype=np.float32), dose_labels])
            np_x_test = np.unique(np_x_test)
            np.sort(np_x_test)
            x_test = torch.tensor(np_x_test, dtype=torch.float32)

            # Train model final time
            # Construct training data
            task_indices = torch.LongTensor(subgroup_indices)
            selected_dose_values = [dose_labels[idx] for idx in selected_doses]
            tox_outcomes = trial_frame['tox_outcome'].values
            eff_outcomes = trial_frame['eff_outcome'].values
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
            dose_rec[trial, :], final_utilities = select_final_dose(dose_scenario, num_subgroups, num_doses, 
                                                                    dose_labels, x_test, max_doses, y_tox_posteriors,
                                                                    y_eff_posteriors,
                                                                    dose_scenario.toxicity_threshold,
                                                                    dose_scenario.efficacy_threshold,
                                                                    dose_scenario.tox_weight,
                                                                    dose_scenario.eff_weight, final_beta_param,
                                                                    use_thall)
        else:
            for subgroup_idx in range(num_subgroups):
                frame = pd.read_csv(f"{filepath}/trial{trial}/{subgroup_idx}_predictions.csv")
                tox_means = frame['tox_predicted'].values
                tox_upper = frame['tox_upper'].values
                eff_means = frame['eff_predicted'].values
                utilities = frame['final_utilities'].values

                max_dose = selected_doses[subgroup_indices == subgroup_idx].max()
                available_dose_indices = np.arange(max_dose + 1)
                available_doses_mask = np.isin(np.arange(num_doses), available_dose_indices)


                tox_conf_interval = tox_upper - tox_means
                tox_ucb = tox_means + (final_beta_param * tox_conf_interval)
                safe_doses_mask = tox_ucb <= dose_scenario.toxicity_threshold

                gt_threshold = np.where(tox_ucb > dose_scenario.toxicity_threshold)[0]
                if gt_threshold.size:
                    first_idx_above_threshold = gt_threshold[0]
                    safe_doses_mask[first_idx_above_threshold:] = False

                utilities = calculate_utility(tox_means, eff_means, dose_scenario.toxicity_threshold,
                                            dose_scenario.efficacy_threshold,
                                            tox_weight, eff_weight)
                thall_utilities = calculate_dose_utility_thall(tox_means, eff_means,
                                                            dose_scenario.toxicity_threshold,
                                                            dose_scenario.efficacy_threshold,
                                                            dose_scenario.p_param)
                dose_set_mask = np.logical_and(available_doses_mask, safe_doses_mask)
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
                    if eff_means[max_util_idx] >= dose_scenario.efficacy_threshold and max_util >= 0.:
                        dose_rec[trial, subgroup_idx] = max_util_idx
                    # else:
                    #     # Try highest eff in safe range
                    #     if max_eff >= dose_scenario.efficacy_threshold:
                    #         dose_rec[trial, subgroup_idx] = max_eff_idx
                else:
                    if eff_means[max_util_thall_idx] >= dose_scenario.efficacy_threshold and max_util_thall >= -0.1:
                        dose_rec[trial, subgroup_idx] = max_util_thall_idx
                    # else:
                    #     # Try highest eff in safe range
                    #     if max_eff >= dose_scenario.efficacy_threshold:
                    #         dose_rec[trial, subgroup_idx] = max_eff_idx     

            
        dose_err[trial, :] = (dose_rec[trial, :] != optimal_doses).astype(np.float32)

    for subgroup_idx in range(num_subgroups):
        values, counts = np.unique(dose_rec[:, subgroup_idx], return_counts=True)
        print(f"Subgroup {subgroup_idx}")
        print(values)
        print(counts)

    all_frame = pd.concat(frames_list)
    grouped_frame = all_frame.groupby(all_frame.index)
    print(grouped_frame.mean())
    print(dose_err.mean(axis=0))
    return dose_err.mean(axis=0), grouped_frame.mean()



# filepath = "results/tenth_pass/scenario1"
# dose_scenario = DoseFindingScenarios.paper_example_1()
# tox_weight = 1
# eff_weight = dose_scenario.toxicity_threshold/dose_scenario.efficacy_threshold
# tox_weight = 1.5
# eff_weight = 1
# use_thall = True
# retrain_model = True

# dose_frame, util_frame = test_select_final_dose_method(filepath, dose_scenario, tox_weight, eff_weight, use_thall, retrain_model)

filepath = "results/eleventh_pass"
use_thall = True
retrain_model = True
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
util_frame = pd.DataFrame(index=np.arange(num_subgroups))
thall_frame = pd.DataFrame(index=np.arange(num_subgroups))

for idx, scenario in scenarios.items():
    sub_filepath = f"{filepath}/scenario{idx}"
    tox_weight = 1.
    eff_weight = scenario.toxicity_threshold/scenario.efficacy_threshold
    print(f"Eff weight: {eff_weight}")
    dose_err, utils = test_select_final_dose_method(sub_filepath, scenario, tox_weight, eff_weight, use_thall, retrain_model)
    frame[f"scenario{idx}"] = dose_err
    util_frame[f"scenario{idx}"] = utils['utility'].values
    thall_frame[f"scenario{idx}"] = utils['thall_utility'].values

util_frame.to_csv(f"{filepath}/test_utility.csv")
thall_frame.to_csv(f"{filepath}/test_thall_utility.csv")

if use_thall:
    frame.to_csv(f"{filepath}/thall_final_dose_error_retrain.csv")
else:
    frame.to_csv(f"{filepath}/test_final_dose_error.csv")
