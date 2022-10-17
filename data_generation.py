import numpy as np
import pandas as pd
from parameterized_models import OQuiqleyModel, OneParamLogisticModel, PowerModel, SigmoidalModel, TwoParamLogisticModel

import matplotlib.pyplot as plt
import seaborn as sns


# Use synthetic examples for doses
# Use real-world examples for doses
# Use generated examples from common parameterized dose-tox curves

class DoseFindingScenario:
    def __init__(self, dose_levels, toxicity_probs, efficacy_probs, optimal_doses,
                 toxicity_threshold, efficacy_threshold):
        self.dose_levels = dose_levels
        self.toxicity_probs = toxicity_probs
        self.efficacy_probs = efficacy_probs
        self.optimal_doses = optimal_doses
        self.toxicity_threshold = toxicity_threshold
        self.efficacy_threshold = efficacy_threshold

    def get_efficacy_prob(self, arm_idx, subgroup_idx=0):
        if len(self.efficacy_probs.shape) == 1:
            return self.efficacy_probs[arm_idx]
        return self.efficacy_probs[subgroup_idx][arm_idx]
    
    def get_toxicity_prob(self, arm_idx, subgroup_idx=0):
        if len(self.toxicity_probs.shape) == 1:
            return self.toxicity_probs[arm_idx]
        return self.toxicity_probs[subgroup_idx][arm_idx]

    def sample_efficacay_event(self, arm_idx, subgroup_idx=0):
        return np.random.rand() <= self.get_efficacy_prob(arm_idx, subgroup_idx)

    def sample_toxicity_event(self, arm_idx, subgroup_idx=0):
        return np.random.rand() <= self.get_toxicity_prob(arm_idx.subgroup_idx)


class DoseFindingScenarios:
    '''
    Scenarios from commonly used parameterized models
    '''
    @staticmethod
    def oquigley_model_example():
        toxicity_probs = np.array([0.15, 0.2, 0.3, 0.6, 0.9])
        efficacy_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.55])
        model = OQuiqleyModel(0.5)
        dose_labels = model.initialize_dose_label(toxicity_probs.flatten())
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def sigmoidal_model_example():
        toxicity_probs = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
        efficacy_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.55])
        model = SigmoidalModel(0.5)
        dose_labels = model.initialize_dose_label(toxicity_probs.flatten())
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    

    @staticmethod
    def power_model_example():
        toxicity_probs = np.array([0.1, 0.2, 0.35, 0.6, 0.8])
        efficacy_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.55])
        model = PowerModel(0.5)
        dose_labels = model.initialize_dose_label(toxicity_probs.flatten())
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    

    @staticmethod
    def one_param_logistic_example():
        toxicity_probs = np.array([0.1, 0.2, 0.35, 0.6, 0.8])
        efficacy_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.55])
        model = OneParamLogisticModel(0.5)
        dose_labels = model.initialize_dose_label(toxicity_probs.flatten())
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    

    @staticmethod
    def two_param_logistic_example():
        toxicity_probs = np.array([0.1, 0.2, 0.35, 0.6, 0.8])
        efficacy_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.55])
        model = TwoParamLogisticModel(0.1, 0.1)
        dose_labels = model.initialize_dose_label(toxicity_probs.flatten())
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    
    '''
    Scenarios from Lee 202
    https://arxiv.org/pdf/2001.02463.pdf
    '''
    @staticmethod
    def lee_synthetic_example():
        a0 = 0.5
        toxicity_probs = np.array([[0.01, 0.01, 0.05, 0.15, 0.20, 0.45],
                                           [0.01, 0.05, 0.15, 0.20, 0.45, 0.60],
                                           [0.01, 0.05, 0.15, 0.20, 0.45, 0.60]])
                                    
        efficacy_probs = np.array([[0.01, 0.02, 0.05, 0.10, 0.10, 0.10],
                                           [0.10, 0.20, 0.30, 0.50, 0.60, 0.65],
                                           [0.20, 0.50, 0.60, 0.80, 0.84, 0.85]])
        dose_skeleton = np.mean(toxicity_probs, axis=0)
        model = OQuiqleyModel(a0)
        dose_skeleton_labels = model.initialize_dose_label(dose_skeleton)
        optimal_doses = np.array([6, 3, 3])
        return DoseFindingScenario(dose_skeleton_labels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    '''
    Scenarios from James 2021
    https://bmccancer.biomedcentral.com/articles/10.1186/s12885-020-07703-6#Tab1 
    QD = once daily
    BID = twice daily

    No efficacy probabiltiies 
    '''
    @staticmethod
    def james_real_1():
        # AZD3514
        # 250mg QD, 500mg QD, 1000mg QD, 1000mg BID, 2000mg BID
        dose_levels = np.array([250, 500, 1000, 2000, 4000])
        toxicity_probs = np.array([0, 0, 0.17, 0.50, 1.0])
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, None,
                                   optimal_doses, toxicity_threshold=0.33, efficacy_threshold=None)

    @staticmethod
    def james_real_2():
        # AZD1208
        # 120mg QD, 240mg QD, 480mg QD, 700mg QD, 900mg QD
        dose_levels = np.array([120, 240, 480, 700, 900])
        toxicity_probs = np.array([0., 0., 0., 0.25, 0.67])
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, None,
                                   optimal_doses, toxicity_threshold=0.33, efficacy_threshold=None)
    
    @staticmethod
    def james_real_3():
        # AZD1480
        # 15mg BID, 20mg BID, 30mg BID, 35mg BID, 45mg BID
        dose_levels = np.array([15, 20, 30, 35, 45])
        toxicity_probs = np.array([0., .2, .2, .67, .67])
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, None,
                                   optimal_doses, toxicity_threshold=0.33, efficacy_threshold=None)

    @staticmethod
    def james_real_4():
        # AZD4877
        # 2mg, 4mg, 7mg, 11mg, 15mg all twice weekly
        dose_levels = np.array([2, 4, 7, 11, 15])
        toxicity_probs = np.array([0., 0., 0., 0., 1.0])
        optimal_doses = np.array([4])
        return DoseFindingScenario(dose_levels, toxicity_probs, None,
                                   optimal_doses, toxicity_threshold=0.33, efficacy_threshold=None)
    
    '''
    Scenario from Aziz 2021
    https://www.jmlr.org/papers/volume22/19-228/19-228.pdf 
    '''
    @staticmethod
    def aziz_get_dose_level(toxicity_probs, beta_0=0, beta_1=1):
        return -np.log((1 - toxicity_probs) / toxicity_probs)

    @staticmethod
    def aziz_synthetic_1():
        toxicity_probs = np.array([0.01, 0.05, 0.15, 0.2, 0.45, 0.6])
        efficacy_probs = np.array([0.1, 0.35, 0.6, 0.6, 0.6, 0.6])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)

    @staticmethod
    def aziz_synthetic_2():
        toxicity_probs = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.15])
        efficacy_probs = np.array([0.001, 0.1, 0.3, 0.5, 0.8, 0.8])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([4])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def aziz_synthetic_3():
        toxicity_probs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.7])
        efficacy_probs = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([0])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    

    @staticmethod
    def aziz_synthetic_4():
        toxicity_probs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
        efficacy_probs = np.array([0.25, 0.45, 0.65, 0.65, 0.65, 0.65])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)

    @staticmethod
    def aziz_synthetic_5():
        toxicity_probs = np.array([0.1, 0.2, 0.25, 0.4, 0.5, 0.6])
        efficacy_probs = np.array([0.3, 0.4, 0.5, 0.7, 0.7, 0.7])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)

    @staticmethod
    def aziz_synthetic_6():
        toxicity_probs = np.array([0.1, 0.3, 0.35, 0.4, 0.5, 0.6])
        efficacy_probs = np.array([0.3, 0.4, 0.5, 0.7, 0.7, 0.7])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def aziz_synthetic_7():
        toxicity_probs = np.array([0.03, 0.06, 0.1, 0.2, 0.4, 0.5])
        efficacy_probs = np.array([0.3, 0.5, 0.52, 0.54, 0.55, 0.55])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    

    @staticmethod
    def aziz_synthetic_8():
        toxicity_probs = np.array([0.02, 0.07, 0.13, 0.17, 0.25, 0.3])
        efficacy_probs = np.array([0.3, 0.5, 0.7, 0.73, 0.76, 0.77])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)

    @staticmethod
    def aziz_synthetic_9():
        toxicity_probs = np.array([0.25, 0.43, 0.50, 0.58, 0.64, 0.75])
        efficacy_probs = np.array([0.3, 0.4, 0.5, 0.6, 0.61, 0.63])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def aziz_synthetic_10():
        toxicity_probs = np.array([0.05, 0.1, 0.25, 0.55, 0.7, 0.9])
        efficacy_probs = np.array([0.01, 0.02, 0.05, 0.35, 0.55, 0.7])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
        
    @staticmethod
    def aziz_synthetic_11():
        toxicity_probs = np.array([0.5, 0.6, 0.69, 0.76, 0.82, 0.89])
        efficacy_probs = np.array([0.4, 0.55, 0.65, 0.65, 0.65, 0.65])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([6])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def aziz_synthetic_12():
        toxicity_probs = np.array([0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
        efficacy_probs = np.array([0.05, 0.25, 0.45, 0.7, 0.7, 0.7])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    @staticmethod
    def aziz_synthetic_13():
        toxicity_probs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
        efficacy_probs = np.array([0.05, 0.1, 0.2, 0.35, 0.55, 0.55])
        dose_levels = DoseFindingScenarios.aziz_get_dose_level(toxicity_probs)
        optimal_doses = np.array([4])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.35, efficacy_threshold=0.2)
    
    '''
    Scenarios from EffTox paper, Yan 2018
    https://reader.elsevier.com/reader/sd/pii/
    S0923753419355061?token=D92BE56514B6EBD88C90CF90B84
    D67749CB64C041BB90D950D637755C95C27A2F44A7B4CD8474A
    92BD7EBF6591D8AFF9&originRegion=eu-west-1&originCreation=20221014165511
    '''
    @staticmethod
    def yan_synthetic_1():
        toxicity_probs = np.array([0.02, 0.06, 0.30, 0.40, 0.50])
        efficacy_probs = np.array([0.2, 0.5, 0.51, 0.52, 0.52])
        dose_levels = np.array([25, 50, 75, 100, 125])
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.2)

    @staticmethod
    def yan_synthetic_2():
        toxicity_probs = np.array([0.04, 0.08, 0.12, 0.30, 0.40])
        efficacy_probs = np.array([0.10, 0.20, 0.60, 0.35, 0.20])
        dose_levels = np.array([25, 50, 75, 100, 125])
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.2)
    
    @staticmethod
    def yan_synthetic_3():
        '''
        Pattern often observed in cytoxic and MTA agents
        '''
        toxicity_probs = np.array([0.04, 0.06, 0.10, 0.14, 0.20])
        efficacy_probs = np.array([0.05, 0.10, 0.30, 0.50, 0.70])
        dose_levels = np.array([25, 50, 75, 100, 125])
        optimal_doses = np.array([4])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.2)
    
    @staticmethod
    def yan_synthetic_4():
        toxicity_probs = np.array([0.05, 0.10, 0.20, 0.35, 0.50])
        efficacy_probs = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
        dose_levels = np.array([25, 50, 75, 100, 125])
        optimal_doses = np.array([5])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.2)
    
    '''
    Scenarios from Takahashi 2021
    '''
    @staticmethod
    def taka_synthetic_1():
        toxicity_probs = np.array([0.05, 0.1, 0.15, 0.3, 0.7, 0.8])
        efficacy_probs = np.array([0.01, 0.15, 0.30, 0.55, 0.65, 0.70])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_2():
        toxicity_probs = np.array([0.05, 0.1, 0.15, 0.45, 0.6, 0.65])
        efficacy_probs = np.array([0.25, 0.45, 0.6, 0.5, 0.2, 0.05])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_3():
        toxicity_probs = np.array([0.05, 0.25, 0.45, 0.6, 0.7, 0.85])
        efficacy_probs = np.array([0.4, 0.5, 0.5, 0.55, 0.55, 0.6])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([1])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_4():
        toxicity_probs = np.array([0.05, 0.1, 0.15, 0.15, 0.2, 0.25])
        efficacy_probs = np.array([0.7, 0.5, 0.4, 0.3, 0.15, 0.01])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([0])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_5():
        toxicity_probs = np.array([0.01, 0.05, 0.1, 0.17, 0.2, 0.3])
        efficacy_probs = np.array([0.05, 0.2, 0.35, 0.45, 0.6, 0.8])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([4, 5])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_6():
        toxicity_probs = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.65])
        efficacy_probs = np.array([0.3, 0.4, 0.45, 0.6, 0.65, 0.7])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_7():
        toxicity_probs = np.array([0.1, 0.15, 0.25, 0.3, 0.4, 0.45])
        efficacy_probs = np.array([0.2, 0.3, 0.5, 0.7, 0.73, 0.75])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([2])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_8():
        toxicity_probs = np.array([0.05, 0.1, 0.2, 0.2, 0.25, 0.3])
        efficacy_probs = np.array([0.05, 0.07, 0.1, 0.12, 0.4, 0.5])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([4, 5])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_9():
        toxicity_probs = np.array([0.4, 0.4, 0.42, 0.45, 0.47, 0.5])
        efficacy_probs = np.array([0.05, 0.15, 0.25, 0.4, 0.5, 0.6])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_10():
        toxicity_probs = np.array([0.25, 0.4, 0.45, 0.5, 0.6, 0.65])
        efficacy_probs = np.array([0.15, 0.35, 0.5, 0.55, 0.6, 0.65])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([6])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_11():
        toxicity_probs = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        efficacy_probs = np.array([0.05, 0.2, 0.4, 0.55, 0.5, 0.25])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([3])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    
    @staticmethod
    def taka_synthetic_12():
        toxicity_probs = np.array([0.05, 0.07, 0.2, 0.1, 0.15, 0.2, 0.25])
        efficacy_probs = np.array([0.05, 0.2, 0.4, 0.55, 0.6, 0.6])
        dose_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        optimal_doses = np.array([3, 4])
        return DoseFindingScenario(dose_levels, toxicity_probs, efficacy_probs,
                                   optimal_doses, toxicity_threshold=0.3, efficacy_threshold=0.5)
    

class TrialPopulation:
    def __init__(self, num_patients, arrival_rate):
        self.num_patients = num_patients
        self.arrival_rate = arrival_rate
        
    def gen_patients(self):
        '''
        Generates all patients for an experiment of length T.
        Completely synthetic, patients from len(arr_rate) number of subgroups.

        Returns numpy list with each patient subgroup
        '''
        # Arrival proportion of each subgroup. If arrive_rate = [5, 4, 3],
        # arrive_dist = [5/12, 4/12, 3/12]
        arrive_sum = sum(self.arrival_rate)
        arrive_dist = [rate/arrive_sum for rate in self.arrival_rate]
        arrive_dist.insert(0, 0)

        # [0, 5/12, 9/12, 12/12]
        arrive_dist_bins = np.cumsum(arrive_dist)

        # Random numbers between 0 and 1 in an array of shape (1, T)
        patients_gen = np.random.rand(self.num_patients)
        patients = np.digitize(patients_gen, arrive_dist_bins) - 1
        return patients

        
class TrialPopulationScenarios:
    @staticmethod
    def lee_trial_population(num_patients):
        arr_rate = [5, 4, 3]
        return TrialPopulation(num_patients, arr_rate)


def plot_true_curves(dose_finding_scenario):
    dose_labels = dose_finding_scenario.dose_levels
    toxicity_probs = dose_finding_scenario.toxicity_probs
    efficacy_probs = dose_finding_scenario.efficacy_probs
    if efficacy_probs is None:
        frame = pd.DataFrame({'dose_labels': dose_labels,
                            'toxicity_prob': toxicity_probs})
    else:
        frame = pd.DataFrame({'dose_labels': dose_labels,
                            'toxicity_prob': toxicity_probs,
                            'efficacy_prob': efficacy_probs })
    frame = pd.melt(frame, id_vars=['dose_labels'], var_name='response', value_name='probability')

    sns.set()
    sns.lineplot(data=frame, x='dose_labels', y='probability', hue='response',style='response', markers=True)
    plt.ylim(0, 1)
    plt.xlabel('Dose Labels')
    plt.ylabel('Response')
    plt.show()



scenario = DoseFindingScenarios.oquigley_model_example()
plot_true_curves(scenario)