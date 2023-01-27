import math
import ssl
import numpy as np
from scipy.stats import beta 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import alpha_func, get_ucb
from metrics import TrialMetrics



def calculate_utility(tox_means, eff_means, tox_thre, eff_thre, tox_weight, eff_weight):
    tox_term = (tox_means - tox_thre) ** 2
    eff_term = (eff_means - eff_thre) ** 2
    # tox_term[tox_means > tox_thre] = 0.
    # eff_term[eff_means < eff_thre] = 0.
    tox_term[tox_means > tox_thre] = -tox_term[tox_means > tox_thre]
    eff_term[eff_means < eff_thre] = 0.
    return (tox_weight * tox_term) + (eff_weight * eff_term)

class DoseFindingModel:
    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate):
        self.time_horizon = time_horizon
        self.num_doses = num_doses
        self.num_subgroups = num_subgroups
        self.learning_rate = learning_rate

        self.a_max = 1
        self.c_param = 0.5

        self.delta = 1 / time_horizon * num_subgroups * np.ones(num_subgroups)
        self.metrics = TrialMetrics(num_subgroups, num_doses, time_horizon)

        self.subgroup_count = np.zeros(num_subgroups)
        self.n_choose = np.zeros((num_subgroups, num_doses)) # num of selection
        self.n_tox = np.zeros((num_subgroups, num_doses)) # num of toxic events
        self.empirical_efficacy_estimate = np.zeros((num_subgroups, num_doses)) # estimated efficacy, q_hat
        self.efficacy_ucb = np.zeros((num_subgroups, num_doses)) # estimate efficacy ucb, q_bar
        self.empirical_toxicity_estimate = np.zeros((num_subgroups, num_doses)) # estimate toxicity, p_hat
        self.model_toxicity_estimate = np.zeros((num_subgroups, num_doses)) # p_out

        self.alpha = np.zeros(num_subgroups)
        self.available_doses = np.zeros((num_subgroups, num_doses)) # Matrix for set of doses allowed, D

        self.allocated_doses = np.ones(time_horizon, dtype=int) * -1 # allocated doses, I
        self.efficacy_at_timestep = np.zeros(time_horizon) # X
        self.toxicity_at_timestep = np.zeros(time_horizon) # Y
        self.patients = patients # H
        self.Z = np.zeros(time_horizon) # Z

    
    def get_toxicity_helper(self, dose_label, alpha, beta):
        raise NotImplementedError

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        raise NotImplementedError

    def update_params(self, subgroup_idx, dose_labels):
        raise NotImplementedError

    def update_model_toxicity_estimate(self, dose_labels):
        raise NotImplementedError
    
    def finalize_parameters(self):
        raise NotImplementedError

    def update_efficacy_ucb(self, curr_s):
        for k in range(self.num_doses):
            self.efficacy_ucb[curr_s, k] = get_ucb(self.empirical_efficacy_estimate[curr_s, k], self.c_param, self.n_choose[curr_s, k], np.sum(self.n_choose[curr_s, :]))

    def update_metrics(self, timestep, curr_s, dose_labels, tox_thre, eff_thre, p_true, q_true, opt_ind):
        '''
        Was update_metrics_with_gradients
        '''
        # Regret = opt dose efficacy - q_true of selected
            
        if self.allocated_doses[timestep] == self.num_doses:
            selected_eff_regret = eff_thre
        else:
            selected_eff_regret = q_true[curr_s, self.allocated_doses[timestep]]
            
        if opt_ind[curr_s] == self.num_doses:
            optimal_eff_regret = eff_thre
            optimal_tox_regret = tox_thre
        else:
            optimal_eff_regret = q_true[curr_s, opt_ind[curr_s]]
            optimal_tox_regret = p_true[curr_s, opt_ind[curr_s]]

        curr_eff_regret = optimal_eff_regret - selected_eff_regret
        curr_tox_regret = optimal_tox_regret - tox_thre

        if curr_tox_regret < 0:
            curr_tox_regret = 0
        else:
            self.metrics.safety_violations[curr_s] += 1
        
        regret = self.allocated_doses[timestep] != opt_ind[curr_s]

        if timestep > 0:
            self.metrics.update_metrics(timestep, self.num_subgroups, curr_s, regret,
                                        self.efficacy_at_timestep[timestep], self.toxicity_at_timestep[timestep],
                                        curr_eff_regret, curr_tox_regret)

        else:
            self.metrics.update_initial_metrics(self.num_subgroups, curr_s, regret,
                                                self.efficacy_at_timestep[timestep], self.toxicity_at_timestep[timestep],
                                                curr_eff_regret, curr_tox_regret)


    def finalize_results(self, timestep, dose_labels, tox_thre, eff_thre, p_true, opt_ind, q_true, tox_weight,
                         eff_weight):
        '''
        Originally finalize_results_with_gradient_shared_param
        '''
        # Recommendation and observe results
        for s in range(self.num_subgroups):
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = self.empirical_efficacy_estimate[s, :] * (self.model_toxicity_estimate[s, :] <= tox_thre)
            #q_below_tox_thre = self.empirical_efficacy_estimate[s, :] * (p_true[s, :] <= tox_thre)
            
            max_dose_val = np.max(q_below_tox_thre)
            max_dose_idx = np.argmax(q_below_tox_thre)

            # If the recommended dose is above efficacy threshold, dose rec is correct, assign to rec
            if max_dose_val >= eff_thre:
                self.metrics.rec[s, max_dose_idx] = 1

            # If recommended dose is not above efficacy threshold, assign no dose to rec
            else:
                self.metrics.rec[s, self.num_doses] = 1
                max_dose_idx = self.num_doses

            if max_dose_idx != opt_ind[s]:
                self.metrics.rec_err[s] = 1
            
            for i in range(self.num_doses):
                if p_true[s, i] <= tox_thre and self.model_toxicity_estimate[s, i] > tox_thre:
                    self.metrics.typeI[s] += 1
                else:
                    if p_true[s, i] > tox_thre and self.model_toxicity_estimate[s, i] <= tox_thre:
                        self.metrics.typeII[s] += 1
                        
        if timestep < self.time_horizon:
            self.metrics.q_mse[:, :, timestep:] = np.tile(np.expand_dims(self.metrics.q_mse[:, :, timestep-1], axis=2), (1, 1, self.time_horizon - timestep))  
        
        selected_tox_probs = np.array([p_true[group_idx, arm_idx] for group_idx, arm_idx in zip(self.patients, self.allocated_doses)])
        safety_violations = np.array(selected_tox_probs > tox_thre, dtype=np.int32)

        # Calculate utilities
        selected_eff_probs = np.array([q_true[group_idx, arm_idx] for group_idx, arm_idx in zip(self.patients, self.allocated_doses)])
        utilities = calculate_utility(selected_tox_probs, selected_eff_probs, tox_thre, eff_thre,
                                      tox_weight=tox_weight, eff_weight=eff_weight)

        self.metrics.typeI = self.metrics.typeI / self.num_doses
        self.metrics.typeII = self.metrics.typeII / self.num_doses
        self.metrics.p_hat = self.empirical_toxicity_estimate
        self.metrics.q_hat = self.empirical_efficacy_estimate
        self.metrics.pats_count = np.unique(self.patients, return_counts=True)[1]
        for s in range(self.num_subgroups):
            self.metrics.safety_violations[s] = safety_violations[self.patients == s].sum() 
            self.metrics.utility_by_person[s] = utilities[self.patients == s].sum()
        self.metrics.safety_violations = self.metrics.safety_violations / self.metrics.pats_count
        self.metrics.utility_by_person = self.metrics.utility_by_person / self.metrics.pats_count
        self.metrics.dose_err_by_person = self.metrics.rec_err / self.metrics.pats_count
        self.metrics.cum_eff_by_person = self.metrics.cum_eff[:, -1] / self.metrics.pats_count
        self.metrics.cum_tox_by_person = self.metrics.cum_tox[:, -1] / self.metrics.pats_count
        self.metrics.eff_regret_by_person = self.metrics.eff_regret[:, -1] / self.metrics.pats_count
        self.metrics.regret_by_person = self.metrics.regret[:, -1] / self.metrics.pats_count

    def update_empirical_efficacy_estimate(self, curr_s, timestep):
        return (self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep]] * self.n_choose[curr_s, self.allocated_doses[timestep]] + \
                self.efficacy_at_timestep[timestep]) / (self.n_choose[curr_s, self.allocated_doses[timestep]] + 1)
        
    def update_empirical_toxicity_estimate(self, curr_s, timestep):
        return (self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]] * self.n_choose[curr_s, self.allocated_doses[timestep]] + \
                self.toxicity_at_timestep[timestep]) / (self.n_choose[curr_s, self.allocated_doses[timestep]] + 1)

    def run_model(self, tox_thre, eff_thre, p_true, q_true, opt_ind, dose_labels):
        timestep = 0

        while timestep < self.time_horizon:
            self.metrics.q_mse[:, :, timestep] = np.abs(q_true - self.empirical_efficacy_estimate)**2

            curr_s = self.patients[timestep]
            self.subgroup_count[curr_s] += 1

            # Initialize / burn-in
            if self.subgroup_count[self.patients[timestep]] < self.num_doses: 
                self.allocated_doses[timestep] = int(self.subgroup_count[self.patients[timestep]]) # allocated dose
                self.efficacy_at_timestep[timestep] = np.random.rand() <= q_true[curr_s, self.allocated_doses[timestep]] # Sample efficacy
                self.toxicity_at_timestep[timestep] = np.random.rand() <= p_true[curr_s, self.allocated_doses[timestep]] # Sample toxicity

                self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep]] = self.efficacy_at_timestep[timestep]
                self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]] = self.toxicity_at_timestep[timestep]

            # Normal loop
            else:
                for s in range(self.num_subgroups):
                    # Calculate alpha
                    self.alpha[s] = alpha_func(dose_labels[s, :], self.num_doses, self.delta[s], np.sum(self.n_choose[s, :])) 
                    # Use toxicity model estimates
                    self.available_doses[s, :] = self.get_available_dose_set(dose_labels, s, tox_thre)
                    # Use empirical toxicity estimates
                    #self.available_doses[s, :] = self.empirical_toxicity_estimate[s, :] <= tox_thre
                    #self.available_doses[s, :] = p_true[s, :] <= tox_thre

                self.allocated_doses[timestep] = np.argmax(self.efficacy_ucb[curr_s, :] * self.available_doses[curr_s, :])
                self.efficacy_at_timestep[timestep] = np.random.rand() <= q_true[curr_s, self.allocated_doses[timestep]]
                self.toxicity_at_timestep[timestep] = np.random.rand() <= p_true[curr_s, self.allocated_doses[timestep]]

                self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep]] = self.update_empirical_efficacy_estimate(curr_s, timestep)
                self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]] = self.update_empirical_toxicity_estimate(curr_s, timestep)

            self.n_choose[curr_s, self.allocated_doses[timestep]] += 1
            self.n_tox[curr_s, self.allocated_doses[timestep]] += self.toxicity_at_timestep[timestep]
            self.update_efficacy_ucb(curr_s)  
            self.update_metrics(timestep, curr_s, dose_labels, tox_thre, eff_thre, p_true, q_true, opt_ind)
            self.update_params(curr_s, dose_labels)

            timestep += 1

        self.update_model_toxicity_estimate(dose_labels)
        self.finalize_results(timestep, dose_labels, tox_thre, eff_thre, p_true, opt_ind)
        self.finalize_parameters()

        return self.metrics
    

class TwoParamSharedModel(DoseFindingModel):
    '''
    Two parameter, one shared
    Contextual, no budget version with gradient

    Input
    T        : time-horizon
    S        : number of subgroups
    K        : number of doses
    pats     : generated patients arrivals
    arr_rate : patients arrival rate
    tox_thre : toxicity threshold
    eff_thre : efficacy threshold
    p_true   : toxicity probabilities
    q_true   : efficacy probabilities
    opt_ind  : MTD doses

    Output
    rec      : dose recommendation
    cum_eff  : cumulative efficacy
    cum_tox  : cumulative toxicity
    typeI    : type-I error for dose safety
    typeII   : type-II error for dose safety
    q_mse    : mean squared error of efficacy
    eff_regret : efficacy regret by group (efficacy based on optimal dose - given dose)
    total_eff_regret : efficacy regret overall 
    ''' 

    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5, b0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate)
        self.a0 = a0
        self.b0 = b0

        self.current_a_hat = self.a0 # estimated overall a
        self.current_b_hat = np.ones(num_subgroups) * self.b0

    def initialize_dose_label(p_true_val, a0, b0):
        return (np.log(p_true_val / (1. - p_true_val)) - a0) / np.exp(b0)

    def get_toxicity(dose_label, alpha, beta):
        return (np.exp(alpha + dose_label * np.exp(beta)))\
                / (1 + np.exp(alpha + dose_label * np.exp(beta)))

    def plot_dose_toxicity_curve(dose_labels, p_true, a_hat_fin, b_hat_fin, p_empirical):
        num_reps = a_hat_fin.shape[0]
        num_doses = dose_labels.shape[0]
        model_toxicities = np.array([TwoParamModel.get_toxicity(dose_labels, a_hat_fin[i], b_hat_fin[i]) for i in range(num_reps)]).flatten()
        frame = pd.DataFrame({'trial': np.repeat(np.arange(num_reps), num_doses),
                            'dose_labels': np.tile(dose_labels, num_reps), 
                            'model': model_toxicities,
                            'empirical': p_empirical.flatten('F')})
        frame = pd.melt(frame, id_vars=['trial', 'dose_labels'], var_name='toxicity', value_name='toxicity_value')

        true_frame = pd.DataFrame({'trial': np.repeat(0, num_doses),
                                'dose_labels': dose_labels,
                                'toxicity': np.repeat('true', num_doses),
                                'toxicity_value': p_true})
        frame = pd.concat([frame, true_frame])
        sns.lineplot(data=frame, x='dose_labels', y='toxicity_value', hue='toxicity', style='toxicity', markers=True)
        plt.xlim(-5, 0)
        plt.ylim(0, 1)
        plt.xlabel('Dose labels')
        plt.ylabel('Toxicity')
        
    def get_toxicity_helper(self, dose_label, alpha, beta):
        return TwoParamModel.get_toxicity(dose_label, alpha, beta)

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        return self.get_toxicity_helper(dose_labels[subgroup_idx, :], self.current_a_hat, self.current_b_hat[subgroup_idx]) <= tox_thre

    def update_model_toxicity_estimate(self, dose_labels):
        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat, self.current_b_hat[s])

    def update_params(self, subgroup_idx, dose_labels):
        old_a = self.current_a_hat
        old_b = self.current_b_hat[subgroup_idx]
        subgroup_dose_labels = dose_labels[subgroup_idx, :]

        gradient_a = self._get_toxicity_two_param_model_alpha_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)
        gradient_b = self._get_toxicity_two_param_model_beta_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)

        new_a = old_a + self.learning_rate * gradient_a
        new_b = old_a + self.learning_rate + gradient_b

        # TODO: Put bounds on b?
        if new_a > self.a_max:
            new_a = self.a_max
        if new_a < 0:
            new_a = 0.01

        if new_b > self.a_max:
            new_b = self.a_max
        if new_b < 0:
            new_b = 0.01
        
        self.current_a_hat = new_a
        self.current_b_hat[subgroup_idx] = new_b
    
    def finalize_parameters(self):
        self.metrics.a_hat_fin = self.current_a_hat
        self.metrics.b_hat_fin = self.current_b_hat

    def _get_toxicity_two_param_model_alpha_gradient(self, a, b, dose_labels, subgroup_idx):
        '''
        Gradient of the log likelihood of toxicity (based on two parameter model) with respect to a
        n_choose = N_k
        n_tox = n_k
        '''
        total_output = 0
        for k in range(self.num_doses):
            N_k = self.n_choose[subgroup_idx, k]
            n_k = self.n_tox[subgroup_idx, k]
            u_k = dose_labels[k]
            prob_tox = self.get_toxicity_helper(u_k, a, b)
            denom = np.exp(a + u_k * np.exp(b)) + 1.
            output = (N_k / denom) + n_k - N_k
            # if prob_tox == 1:
            #     output = 0.01
            total_output += output
        return total_output

    def _get_toxicity_two_param_model_beta_gradient(self, a, b, dose_labels, subgroup_idx):
        '''
        Gradient of the log likelihood of toxicity (based on two parameter model) with respect to b
        '''
        total_output = 0
        for k in range(self.num_doses):
            N_k = self.n_choose[subgroup_idx, k]
            n_k = self.n_tox[subgroup_idx, k]
            u_k = dose_labels[k]
            prob_tox = self.get_toxicity_helper(u_k, a, b)
            output = u_k * np.exp(b) * (n_k - N_k * prob_tox)
            # if prob_tox == 1:
            #     output = 0.01
            total_output += output
        return total_output


class TwoParamAllSharedModel(TwoParamSharedModel):
    '''
    Two parameter model with all parameters shared (subgroups not trained separately)
    '''
    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5, b0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate, a0, b0)
        self.current_a_hat = self.a0
        self.current_b_hat = self.b0

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        return self.get_toxicity_helper(dose_labels[subgroup_idx, :], self.current_a_hat, self.current_b_hat) <= tox_thre

    def update_model_toxicity_estimate(self, dose_labels):
        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat, self.current_b_hat)

    def update_params(self, subgroup_idx, dose_labels):
        old_a = self.current_a_hat
        old_b = self.current_b_hat
        subgroup_dose_labels = dose_labels[subgroup_idx, :]

        gradient_a = self._get_toxicity_two_param_model_alpha_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)
        gradient_b = self._get_toxicity_two_param_model_beta_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)

        new_a = old_a + self.learning_rate * gradient_a
        new_b = old_a + self.learning_rate + gradient_b

        # TODO: Put bounds on b?
        if new_a > self.a_max:
            new_a = self.a_max
        if new_a < 0:
            new_a = 0.01

        if new_b > self.a_max:
            new_b = self.a_max
        if new_b < 0:
            new_b = 0.01
        
        self.current_a_hat = new_a
        self.current_b_hat = new_b


class TwoParamModel(TwoParamSharedModel):
    '''
    Two parameter contextual model
    '''

    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5, b0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate, a0, b0)
        self.current_a_hat = np.ones(num_subgroups) * self.a0
        self.current_b_hat = np.ones(num_subgroups) * self.b0

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        return self.get_toxicity_helper(dose_labels[subgroup_idx, :], self.current_a_hat[subgroup_idx], self.current_b_hat[subgroup_idx]) <= tox_thre

    def update_model_toxicity_estimate(self, dose_labels):
        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s], self.current_b_hat[s])

    def update_params(self, subgroup_idx, dose_labels):
        old_a = self.current_a_hat[subgroup_idx]
        old_b = self.current_b_hat[subgroup_idx]
        subgroup_dose_labels = dose_labels[subgroup_idx, :]

        gradient_a = self._get_toxicity_two_param_model_alpha_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)
        gradient_b = self._get_toxicity_two_param_model_beta_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)

        new_a = old_a + self.learning_rate * gradient_a
        new_b = old_a + self.learning_rate + gradient_b

        # TODO: Put bounds on b?
        if new_a > self.a_max:
            new_a = self.a_max
        if new_a < 0:
            new_a = 0.01

        if new_b > self.a_max:
            new_b = self.a_max
        if new_b < 0:
            new_b = 0.01
        
        self.current_a_hat[subgroup_idx] = new_a
        self.current_b_hat[subgroup_idx] = new_b

class TwoParamAllSharedModel(TwoParamSharedModel):
    '''
    Two parameter contextual model
    '''

    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5, b0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate, a0, b0)
        self.current_a_hat = self.a0
        self.current_b_hat = self.b0

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        return self.get_toxicity_helper(dose_labels[subgroup_idx, :], self.current_a_hat, self.current_b_hat) <= tox_thre

    def update_model_toxicity_estimate(self, dose_labels):
        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat, self.current_b_hat)

    def update_params(self, subgroup_idx, dose_labels):
        old_a = self.current_a_hat
        old_b = self.current_b_hat
        subgroup_dose_labels = dose_labels[subgroup_idx, :]

        gradient_a = self._get_toxicity_two_param_model_alpha_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)
        gradient_b = self._get_toxicity_two_param_model_beta_gradient(old_a, old_b, subgroup_dose_labels, subgroup_idx)

        new_a = old_a + self.learning_rate * gradient_a
        new_b = old_a + self.learning_rate + gradient_b

        # TODO: Put bounds on b?
        if new_a > self.a_max:
            new_a = self.a_max
        if new_a < 0:
            new_a = 0.01

        if new_b > self.a_max:
            new_b = self.a_max
        if new_b < 0:
            new_b = 0.01
        
        self.current_a_hat = new_a
        self.current_b_hat = new_b
    

class TanhModel(DoseFindingModel):
    '''
    One parameter tanh model from O'Quigley
    Contextual, no budget version with gradient
    ''' 

    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate)
        self.a0 = a0
        self.current_a_hat = np.ones(num_subgroups) * self.a0

    def initialize_dose_label(p_true_val, a0):
        x = (p_true_val ** (1. / a0) * 2. - 1.)
        return 1./2. * np.log((1. + x)/(1. - x))

    def get_toxicity(dose_label, alpha):
        return ((np.tanh(dose_label) + 1.) / 2.) ** alpha

    def plot_dose_toxicity_curve(dose_labels, p_true, a_hat_fin, p_empirical):
        num_reps = a_hat_fin.shape[0]
        num_doses = dose_labels.shape[0]
        model_toxicities = np.array([TanhModel.get_toxicity(dose_labels, a_hat_fin[i]) for i in range(num_reps)]).flatten()
        frame = pd.DataFrame({'trial': np.repeat(np.arange(num_reps), num_doses),
                            'dose_labels': np.tile(dose_labels, num_reps), 
                            'model': model_toxicities,
                            'empirical': p_empirical.flatten('F')})
        frame = pd.melt(frame, id_vars=['trial', 'dose_labels'], var_name='toxicity', value_name='toxicity_value')

        true_frame = pd.DataFrame({'trial': np.repeat(0, num_doses),
                                'dose_labels': dose_labels,
                                'toxicity': np.repeat('true', num_doses),
                                'toxicity_value': p_true})
        frame = pd.concat([frame, true_frame])
        sns.lineplot(data=frame, x='dose_labels', y='toxicity_value', hue='toxicity', style='toxicity', markers=True)
        plt.xlim(-5, 0)
        plt.ylim(0, 1)
        plt.xlabel('Dose labels')
        plt.ylabel('Toxicity')

    def get_toxicity_helper(self, dose_label, alpha, beta=None):
        return TanhModel.get_toxicity(dose_label, alpha)

    def get_available_dose_set(self, dose_labels, subgroup_idx, tox_thre):
        alpha = alpha_func(dose_labels[subgroup_idx, :], self.num_doses, self.delta[subgroup_idx], np.sum(self.n_choose[subgroup_idx, :])) 
        return self.get_toxicity_helper(dose_labels[subgroup_idx, :], self.current_a_hat[subgroup_idx] + alpha) <= tox_thre

    def update_model_toxicity_estimate(self, dose_labels):
        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s])

    def update_params(self, subgroup_idx, dose_labels):
        old_a = self.current_a_hat[subgroup_idx]
        subgroup_dose_labels = dose_labels[subgroup_idx, :]

        gradient = self._get_log_likelihood_gradient(old_a, subgroup_idx, subgroup_dose_labels)
        new_a = old_a + self.learning_rate * gradient

        # TODO: Put bounds on b?
        if new_a > self.a_max:
            new_a = self.a_max
        if new_a < 0:
            new_a = 0.01

        self.current_a_hat[subgroup_idx] = new_a

    def finalize_parameters(self):
        self.metrics.a_hat_fin = self.current_a_hat
    
    def _get_log_likelihood_gradient(self, alpha, subgroup_idx, dose_labels):
        def _dose_toxicity_helper(dose_label):
            return ((np.tanh(dose_label) + 1.) / 2.)

        total_output = 0
        for k in range(self.num_doses):
            N_k = self.n_choose[subgroup_idx, k]
            n_tox_k = self.n_tox[subgroup_idx, k]
            dose_label = dose_labels[k]
            prob_tox = self.get_toxicity_helper(dose_label, alpha)
            output = (n_tox_k - (prob_tox * N_k)) * (np.log(_dose_toxicity_helper(dose_label)) / (1. - prob_tox))
            if prob_tox == 1:
                output = 0.01
            total_output += output
        return output

class OGTanhModel(TanhModel):
    '''
    One parameter tanh model from O'Quigley
    Contextual, no budget version without gradient
    ''' 

    def __init__(self, time_horizon, num_subgroups, num_doses, patients, learning_rate, a0=0.5):
        super().__init__(time_horizon, num_subgroups, num_doses, patients, learning_rate, a0)
        self.current_a_hat = np.zeros((num_subgroups, time_horizon))
        self.ak_hat = np.ones((self.num_subgroups, self.num_doses)) * self.a0 # estimated invididual a
    
    def run_model(self, dose_scenario, dose_labels):
        timestep = 0
        tox_thre = dose_scenario.toxicity_threshold
        eff_thre = dose_scenario.efficacy_threshold
        p_true = dose_scenario.toxicity_probs
        q_true = dose_scenario.efficacy_probs
        opt_ind = dose_scenario.optimal_doses

        while timestep < self.time_horizon:
            for s in range(self.num_subgroups):
                I_est = np.argmax(self.n_choose[s, :])
                self.current_a_hat[s, timestep] = self.ak_hat[s, I_est]

            self.metrics.q_mse[:, :, timestep] = np.abs(q_true - self.empirical_efficacy_estimate)**2
            curr_s = self.patients[timestep]
            self.subgroup_count[curr_s] += 1

            # Initialize / burn-in
            if self.subgroup_count[self.patients[timestep]] < self.num_doses: 
                self.allocated_doses[timestep] = int(self.subgroup_count[self.patients[timestep]]) # allocated dose
                self.efficacy_at_timestep[timestep] = np.random.rand() <= q_true[curr_s, self.allocated_doses[timestep]] # Sample efficacy
                self.toxicity_at_timestep[timestep] = np.random.rand() <= p_true[curr_s, self.allocated_doses[timestep]] # Sample toxicity

                self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep]] = self.efficacy_at_timestep[timestep]
                self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]] = self.toxicity_at_timestep[timestep]

            # Normal loop
            else:
                for s in range(self.num_subgroups):
                    # Calculate alpha
                    self.alpha[s] = alpha_func(dose_labels[s, :], self.num_doses, self.delta[s], np.sum(self.n_choose[s, :])) 
                    # Use toxicity model estimates
                    self.available_doses[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s, timestep]) <= tox_thre
                    # Use empirical toxicity estimates
                    #self.available_doses[s, :] = self.empirical_toxicity_estimate[s, :] <= tox_thre
                    #self.available_doses[s, :] = p_true[s, :] <= tox_thre

                self.allocated_doses[timestep] = np.argmax(self.efficacy_ucb[curr_s, :] * self.available_doses[curr_s, :])
                self.efficacy_at_timestep[timestep] = np.random.rand() <= q_true[curr_s, self.allocated_doses[timestep]]
                self.toxicity_at_timestep[timestep] = np.random.rand() <= p_true[curr_s, self.allocated_doses[timestep]]

                self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep]] = self.update_empirical_efficacy_estimate(curr_s, timestep)
                self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]] = self.update_empirical_toxicity_estimate(curr_s, timestep)

            self.n_choose[curr_s, self.allocated_doses[timestep]] += 1
            self.n_tox[curr_s, self.allocated_doses[timestep]] += self.toxicity_at_timestep[timestep]
            self.update_efficacy_ucb(curr_s)  
            self.update_metrics(timestep, curr_s, dose_labels, tox_thre, eff_thre, p_true, q_true, opt_ind)
            
            
            new_a = np.log(self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep]]) / \
                            np.log((np.tanh(dose_labels[curr_s, self.allocated_doses[timestep]]) + 1.) / 2.)
            if new_a > self.a_max:
                new_a = self.a_max
            if new_a < 0:
                new_a = 0.01
            
            self.ak_hat[curr_s, self.allocated_doses[timestep]] = new_a

            timestep += 1

        for s in range(self.num_subgroups):
            self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s, -1])

        self.finalize_results(timestep, dose_labels, tox_thre, eff_thre, p_true, opt_ind,
                              q_true, dose_scenario.tox_weight, dose_scenario.eff_weight)
        self.metrics.a_hat_fin = self.current_a_hat[:, -1]

        return self.metrics

    # def run_model(self, tox_thre, eff_thre, p_true, q_true, opt_ind, dose_labels):
    #     timestep = 0
    #     cohort_size = 3

    #     # Initialize first cohort wiht lowest dose
    #     for idx, curr_s in enumerate(self.patients[timestep: timestep + cohort_size]):
    #         t = timestep + idx
    #         self.allocated_doses[t] = 0
    #         self.efficacy_at_timestep[t] = np.random.rand() <= q_true[curr_s, self.allocated_doses[t]] # Sample efficacy
    #         self.toxicity_at_timestep[t] = np.random.rand() <= p_true[curr_s, self.allocated_doses[t]] # Sample toxicity

    #         self.empirical_efficacy_estimate[curr_s, self.allocated_doses[t]] = self.efficacy_at_timestep[t]
    #         self.empirical_toxicity_estimate[curr_s, self.allocated_doses[t]] = self.toxicity_at_timestep[t]

    #     timestep += cohort_size
    #     max_doses = np.zeros(self.num_subgroups)

    #     while timestep < self.time_horizon:
    #         for s in range(self.num_subgroups):
    #             I_est = np.argmax(self.n_choose[s, :])
    #             self.current_a_hat[s, timestep] = self.ak_hat[s, I_est]
    #         self.metrics.q_mse[:, :, timestep] = np.abs(q_true - self.empirical_efficacy_estimate)**2
    #         selected_dose_by_subgroup = np.empty(self.num_subgroups, dtype=np.int32)
    #         cohort_patients = self.patients[timestep: timestep + cohort_size]

    #         for s in range(self.num_subgroups):
    #             # Calculate alpha
    #             self.alpha[s] = alpha_func(dose_labels[s, :], self.num_doses, self.delta[s], np.sum(self.n_choose[s, :])) 
    #             # Use toxicity model estimates
    #             self.available_doses[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s, timestep]) <= tox_thre
    #             # Use empirical toxicity estimates
    #             #self.available_doses[s, :] = self.empirical_toxicity_estimate[s, :] <= tox_thre
    #             #self.available_doses[s, :] = p_true[s, :] <= tox_thre
    #             selected_dose = np.argmax(self.efficacy_ucb[s, :] * self.available_doses[s, :])
    #             if selected_dose > max_doses[s] + 1:
    #                 selected_dose = max_doses[s] + 1
    #                 max_doses[s] += 1
    #             selected_dose_by_subgroup[s] = selected_dose
            
    #         for idx, curr_s in enumerate(cohort_patients):
    #             self.subgroup_count[curr_s] += 1
    #             self.allocated_doses[timestep + idx] = selected_dose_by_subgroup[curr_s]
    #             self.efficacy_at_timestep[timestep + idx] = np.random.rand() <= q_true[curr_s, self.allocated_doses[timestep + idx]]
    #             self.toxicity_at_timestep[timestep + idx] = np.random.rand() <= p_true[curr_s, self.allocated_doses[timestep + idx]]

    #             self.empirical_efficacy_estimate[curr_s, self.allocated_doses[timestep + idx]] = self.update_empirical_efficacy_estimate(curr_s, timestep + idx)
    #             self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep + idx]] = self.update_empirical_toxicity_estimate(curr_s, timestep + idx)

    #             self.n_choose[curr_s, self.allocated_doses[timestep + idx]] += 1
    #             self.n_tox[curr_s, self.allocated_doses[timestep + idx]] += self.toxicity_at_timestep[timestep + idx]
    #             self.update_efficacy_ucb(curr_s)  
    #             self.update_metrics(timestep + idx, curr_s, dose_labels, tox_thre, eff_thre, p_true, q_true, opt_ind)
            
    #             new_a = np.log(self.empirical_toxicity_estimate[curr_s, self.allocated_doses[timestep + idx]]) / \
    #                             np.log((np.tanh(dose_labels[curr_s, self.allocated_doses[timestep + idx]]) + 1.) / 2.)
    #             if new_a > self.a_max:
    #                 new_a = self.a_max
    #             if new_a < 0:
    #                 new_a = 0.01
    #             print(new_a)
    #             self.ak_hat[curr_s, self.allocated_doses[timestep + idx]] = new_a
    #             print(timestep + idx)
    #         timestep += cohort_size

    #     for s in range(self.num_subgroups):
    #         I_est = np.argmax(self.n_choose[s, :])
    #         self.current_a_hat[s, timestep - 1] = self.ak_hat[s, I_est]

    #     print(f"Final a hat: {self.current_a_hat[:, -1]}")
    #     for s in range(self.num_subgroups):
    #         self.model_toxicity_estimate[s, :] = self.get_toxicity_helper(dose_labels[s, :], self.current_a_hat[s, -1])

    #     print(self.model_toxicity_estimate)
    #     self.finalize_results(timestep, dose_labels, tox_thre, eff_thre, p_true, opt_ind, q_true)
    #     self.metrics.a_hat_fin = self.current_a_hat[:, -1]

    #     return self.metrics