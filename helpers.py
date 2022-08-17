import math
import ssl
import numpy as np
from scipy.stats import beta 
import pandas as pd
import matplotlib.pyplot as plt


class Internal_C3T_Metrics:
    def __init__(self, S, K, T):
        self.total_cum_eff = np.zeros(T)
        self.total_cum_tox = np.zeros(T)
        self.cum_eff = np.zeros((S, T))
        self.cum_tox = np.zeros((S, T))
        self.p_hat = np.zeros((S, K)) # estimated toxicity
        self.q_mse = np.zeros((S, K, T)) # mse of efficacy estimate
        self.total_eff_regret = np.zeros(T)
        self.eff_regret = np.zeros((S, T))
        self.total_tox_regret = np.zeros(T)
        self.tox_regret = np.zeros((S, T))
        self.safety_violations = np.zeros(S)

        self.rec = np.zeros((S, K+1))
        self.cum_s = np.zeros((S, T))
        self.rec_err = np.zeros(S)
        self.typeI = np.zeros(S)
        self.typeII = np.zeros(S)
        self.a_hat_fin = np.zeros(S)
        self.p_out = np.zeros((S, K))

        self.a_hat_fin_shared = 0
        self.b_hat_fin = np.zeros(S)


class C3T_Metrics:
    def __init__(self, S, K, B, T, reps):
        self.rec = np.zeros((S, K+1))
        self.total_cum_eff = np.zeros((B, reps))
        self.total_cum_tox = np.zeros((B, reps))
        self.cum_eff = np.zeros((S, B, reps))
        self.cum_tox = np.zeros((S, B, reps))
        self.cum_s = np.zeros((S, T, reps))
        self.typeI = np.zeros(S)
        self.typeII = np.zeros(S)
        self.rec_err = np.zeros((S, reps))

        self.a_hat_fin = np.zeros((S, reps))
        self.p_hat = np.zeros((S, K, reps))
        self.q_mse_reps = np.zeros((S, K, T, reps))
        self.total_eff_regret = np.zeros((T, reps))
        self.eff_regret = np.zeros((S, T, reps))
        self.total_tox_regret = np.zeros((T, reps))
        self.tox_regret = np.zeros((S, T, reps))
        self.safety_violations = np.zeros((S, reps))
        self.pats_count = np.zeros((S, reps))
        self.dose_err_by_person = np.zeros((S, reps))



def alpha_func(dose_level, K, max_toxicity_prob, subgroup_count):
    '''
    Confidence interval a_s for the dose_toxicity model of subgroup s
    '''
    C_k = -np.log((np.tanh(dose_level) + 1.) / 2.)
    C1 = np.min(np.abs(C_k))
    C1_bar = (1. / C1) ** (2 / 3) / 30
    alpha = C1_bar * K * (np.log(2. * K / max_toxicity_prob) / (2. * subgroup_count)) ** (3 / 4) 
    return alpha

def get_expected_improvement(a, b):
    alpha = 0.05

    intv = beta.ppf(1-alpha/2, a, b) - beta.ppf(alpha/2, a, b)
    intv_on = beta.ppf(1-alpha/2, a+1, b) - beta.ppf(alpha/2, a+1, b)
    intv_off = beta.ppf(1-alpha/2, a, b+1) - beta.ppf(alpha/2, a, b+1)

    intv_on = max(0, intv - intv_on)
    intv_off = max(0, intv - intv_off)

    value = (a * intv_on + b * intv_off) / (a + b)
    return value

def get_ucb(estimate_var, c_param, num_subgroup_arm, num_subgroup):
    upperbound = 2
    value = estimate_var + (c_param * np.log(num_subgroup) / num_subgroup_arm) ** (1/2)
    ucb = min(value, upperbound)
    return ucb

def gen_patients(T, arrive_rate):
    '''
    Generates all patients for an experiment of length T
    '''
    # Arrival proportion of each subgroup. If arrive_rate = [5, 4, 3],
    # arrive_dist = [5/12, 4/12, 3/12]
    arrive_sum = sum(arrive_rate)
    arrive_dist = [rate/arrive_sum for rate in arrive_rate]
    arrive_dist.insert(0, 0)

    # [0, 5/12, 9/12, 12/12]
    arrive_dist_bins = np.cumsum(arrive_dist)

    # Random numbers between 0 and 1 in an array of shape (1, T)
    patients_gen = np.random.rand(T)
    patients = np.digitize(patients_gen, arrive_dist_bins) - 1
    return patients

# Calculate dose level for each subgroup
def initialize_dose_label(p_true_val, a0):
    x = (p_true_val ** (1. / a0) * 2. - 1.)
    return 1./2. * np.log((1. + x)/(1. - x))

def get_toxicity(dose_label, a):
    return ((np.tanh(dose_label) + 1.) / 2.)**a

def get_toxicity_helper(dose_label):
    return ((np.tanh(dose_label) + 1.) / 2.)

def get_log_likelihood(a, K, subgroup_idx, dose_label, n_choose, n_tox):
    total_output = 0
    for k in range(K):
        N_k = n_choose[subgroup_idx, k]
        n_tox_k = n_tox[subgroup_idx, k]
        term1 = a * n_tox_k * np.log(get_toxicity_helper(dose_label))
        term2 = (N_k - n_tox_k) * np.log(1 - (get_toxicity(dose_label, a)))
        output = term1 + term2
        total_output += output
    return output

def get_log_likelihood_gradient(a, K, subgroup_idx, dose_labels, n_choose, n_tox):
    total_output = 0
    for k in range(K):
        N_k = n_choose[subgroup_idx, k]
        n_tox_k = n_tox[subgroup_idx, k]
        dose_label = dose_labels[k]
        prob_tox = get_toxicity(dose_label, a)
        output = (n_tox_k - (prob_tox * N_k)) * (np.log(get_toxicity_helper(dose_label)) / (1. - prob_tox))
        if prob_tox == 1:
            output = 0.01
        total_output += output
    return output

def get_log_likelihood_gradient_combined(a, K, dose_labels, n_choose, n_tox):
    total_output = 0
    for k in range(K):
        N_k = n_choose[k]
        n_tox_k = n_tox[k]
        dose_label = dose_labels[k]
        prob_tox = get_toxicity(dose_label, a)
        # print(a, dose_label, prob_tox)
        output = (n_tox_k - (prob_tox * N_k)) * (np.log(get_toxicity_helper(dose_label)) / (1. - prob_tox))
        if prob_tox == 1:
            output = 0.01
        total_output += output
    return output

def gradient_ascent_update(old_alpha, learning_rate, K, subgroup_idx, dose_label, n_choose, n_tox, is_combined_model=False):
    if is_combined_model:
        gradient = get_log_likelihood_gradient_combined(old_alpha, K, dose_label, n_choose, n_tox)
    else:
        gradient = get_log_likelihood_gradient(old_alpha, K, subgroup_idx, dose_label, n_choose, n_tox)
    new_alpha = old_alpha + learning_rate * gradient
    return new_alpha

def get_toxicity_two_param_model(dose_label, alpha, beta):
    return (np.exp(alpha + dose_label * np.exp(beta)))/(1 + np.exp(alpha + dose_label * np.exp(beta)))

def get_toxicity_two_param_model_alpha_gradient(a, b, dose_labels, K, subgroup_idx, n_choose, n_tox):
    '''
    Gradient of the log likelihood of toxicity (based on two parameter model) with respect to a
    n_choose = N_k
    n_tox = n_k
    '''
    total_output = 0
    for k in range(K):
        N_k = n_choose[subgroup_idx, k]
        n_k = n_tox[subgroup_idx, k]
        u_k = dose_labels[k]
        prob_tox = get_toxicity_two_param_model(u_k, a, b)
        denom = np.exp(a + u_k * np.exp(b)) + 1.
        output = (N_k / denom) + n_k - N_k
        # if prob_tox == 1:
        #     output = 0.01
        total_output += output
    return total_output


def get_toxicity_two_param_model_beta_gradient(a, b, dose_labels, K, subgroup_idx, n_choose, n_tox):
    '''
    Gradient of the log likelihood of toxicity (based on two parameter model) with respect to b
    '''
    total_output = 0
    for k in range(K):
        N_k = n_choose[subgroup_idx, k]
        n_k = n_tox[subgroup_idx, k]
        u_k = dose_labels[k]
        prob_tox = get_toxicity_two_param_model(u_k, a, b)
        output = u_k * np.exp(b) * (n_k - N_k * prob_tox)
        # if prob_tox == 1:
        #     output = 0.01
        total_output += output
    return total_output


def gradient_ascent_two_param_update(old_a, old_b, learning_rate, dose_labels, K, subgroup_idx, n_choose, n_tox):
    gradient_a = get_toxicity_two_param_model_alpha_gradient(old_a, old_b, dose_labels, K, subgroup_idx, n_choose, n_tox)
    gradient_b = get_toxicity_two_param_model_beta_gradient(old_a, old_b, dose_labels, K, subgroup_idx, n_choose, n_tox)

    new_a = old_a + learning_rate * gradient_a
    new_b = old_a + learning_rate + gradient_b
    return new_a, new_b
