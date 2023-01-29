import pandas as pd
import numpy as np

from data_generation import DoseFindingScenarios, TrialPopulationScenarios



def calculate_dose_utility_thall(tox_values, eff_values, tox_thre, eff_thre, p_param):
    tox_term = (tox_values / tox_thre) ** p_param
    eff_term = ((1. - eff_values) / (1. - eff_thre)) ** p_param
    utilities = 1. - ( tox_term + eff_term ) ** (1. / p_param)
    return utilities


scenario = DoseFindingScenarios.paper_example_11()
thall_utilities = calculate_dose_utility_thall(scenario.toxicity_probs, scenario.efficacy_probs,
                                                scenario.toxicity_threshold,
                                                scenario.efficacy_threshold,
                                                scenario.p_param)
print(thall_utilities)