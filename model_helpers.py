import math
import ssl
import numpy as np
import pandas as pd
from helpers import get_ucb, get_toxicity, gradient_ascent_update, Internal_C3T_Metrics


def finalize_results(metrics, T, t, K, S, X, H, n_choose, ak_hat, dose_labels,
                     tox_thre, eff_thre, q_bar, p_true, opt_ind,
                     uses_context=True):

    for s in range(S):
        if uses_context:
            I_est = np.argmax(n_choose[s, :])
            metrics.a_hat_fin[s] = ak_hat[s, I_est]
            metrics.p_out[s, :] = get_toxicity(dose_labels[s, :], metrics.a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar[s, :] * (metrics.p_out[s, :] <= tox_thre)
        else:
            I_est = np.argmax(n_choose)
            metrics.a_hat_fin[s] = ak_hat[I_est]
            metrics.p_out[s, :] = get_toxicity(dose_labels, metrics.a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar * (metrics.p_out[s, :] <= tox_thre)
        max_dose_val = np.max(q_below_tox_thre)
        max_dose_idx = np.argmax(q_below_tox_thre)

        # If the recommended dose is above efficacy threshold, dose rec is correct
        if max_dose_val >= eff_thre:
            metrics.rec[s, max_dose_idx] = 1
        else:
            metrics.rec[s, K] = 1
            max_dose_idx = K

        if max_dose_idx != opt_ind[s]:
            metrics.rec_err[s] = 1
        
        for i in range(K):
            if p_true[s, i] <= tox_thre and metrics.p_out[s, i] > tox_thre:
                metrics.typeI[s] += 1
            else:
                if p_true[s, i] > tox_thre and metrics.p_out[s, i] <= tox_thre:
                    metrics.typeII[s] += 1
                    
    if t < T:
        metrics.q_mse[:, :, t:] = np.tile(np.expand_dims(metrics.q_mse[:, :, t-1], axis=2), (1, 1, T-t))  

    for tau in range(t):
        if X[tau] > -1:
            metrics.cum_s[H[tau], tau:] += 1

    metrics.typeI = metrics.typeI / K
    metrics.typeII = metrics.typeII / K


# def finalize_results_with_gradient_shared_param(metrics: Internal_C3T_Metrics, T, t, K, S, X, H, n_choose, a_hat, b_hat, dose_labels,
#                                                 tox_thre, eff_thre, q_bar, p_true, opt_ind):
#     # Recommendation and observe results
#     for s in range(S):
#         metrics.a_hat_fin_shared = a_hat
#         metrics.b_hat_fin[s] = b_hat[s]
#         metrics.p_out[s, :] = get_toxicity_two_param_model(dose_labels[s, :], metrics.a_hat_fin_shared, metrics.b_hat_fin[s])
#         # Dose with max empirical efficacy also below toxicity threshold
#         q_below_tox_thre = q_bar[s, :] * (metrics.p_out[s, :] <= tox_thre)
#         max_dose_val = np.max(q_below_tox_thre)
#         max_dose_idx = np.argmax(q_below_tox_thre)


#         # If the recommended dose is above efficacy threshold, dose rec is correct, assign to rec
#         if max_dose_val >= eff_thre:
#             metrics.rec[s, max_dose_idx] = 1
#         # If recommended dose is not above efficacy threshold, assign no dose to rec
#         else:
#             metrics.rec[s, K] = 1
#             max_dose_idx = K

#         if max_dose_idx != opt_ind[s]:
#             metrics.rec_err[s] = 1
        
#         for i in range(K):
#             if p_true[s, i] <= tox_thre and metrics.p_out[s, i] > tox_thre:
#                 metrics.typeI[s] += 1
#             else:
#                 if p_true[s, i] > tox_thre and metrics.p_out[s, i] <= tox_thre:
#                     metrics.typeII[s] += 1
                    
#     if t < T:
#         metrics.q_mse[:, :, t:] = np.tile(np.expand_dims(metrics.q_mse[:, :, t-1], axis=2), (1, 1, T-t))  

#     for tau in range(t):
#         if X[tau] > -1:
#             metrics.cum_s[H[tau], tau:] += 1

#     metrics.typeI = metrics.typeI / K
#     metrics.typeII = metrics.typeII / K



def finalize_results_with_gradient(metrics: Internal_C3T_Metrics, T, t, K, S, X, H, n_choose, a_hat, dose_labels,
                     tox_thre, eff_thre, q_bar, p_true, opt_ind, uses_context=True):
    # Recommendation and observe results
    for s in range(S):
        metrics.a_hat_fin[s] = a_hat[s]
        if uses_context:
            metrics.p_out[s, :] = get_toxicity(dose_labels[s, :], metrics.a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar[s, :] * (metrics.p_out[s, :] <= tox_thre)
        else:
            metrics.p_out[s, :] = get_toxicity(dose_labels, metrics.a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar * (metrics.p_out[s, :] <= tox_thre)
        max_dose_val = np.max(q_below_tox_thre)
        max_dose_idx = np.argmax(q_below_tox_thre)


        # If the recommended dose is above efficacy threshold, dose rec is correct, assign to rec
        if max_dose_val >= eff_thre:
            metrics.rec[s, max_dose_idx] = 1
        # If recommended dose is not above efficacy threshold, assign no dose to rec
        else:
            metrics.rec[s, K] = 1
            max_dose_idx = K

        if max_dose_idx != opt_ind[s]:
            metrics.rec_err[s] = 1
        
        for i in range(K):
            if p_true[s, i] <= tox_thre and metrics.p_out[s, i] > tox_thre:
                metrics.typeI[s] += 1
            else:
                if p_true[s, i] > tox_thre and metrics.p_out[s, i] <= tox_thre:
                    metrics.typeII[s] += 1
                    
    if t < T:
        metrics.q_mse[:, :, t:] = np.tile(np.expand_dims(metrics.q_mse[:, :, t-1], axis=2), (1, 1, T-t))  

    for tau in range(t):
        if X[tau] > -1:
            metrics.cum_s[H[tau], tau:] += 1

    metrics.typeI = metrics.typeI / K
    metrics.typeII = metrics.typeII / K


def update_metrics_with_gradients(metrics: Internal_C3T_Metrics, X, Y, I, t, K, S, opt_ind, curr_s, p_true, q_true, tox_thre, eff_thre):
    # Regret = opt dose efficacy - q_true of selected
        
    if I[t] == K:
        selected_eff_regret = eff_thre
    else:
        selected_eff_regret = q_true[curr_s, I[t]]
        
    if opt_ind[curr_s] == K:
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
        metrics.safety_violations[curr_s] += 1

    if t > 0:
        metrics.total_cum_eff[t] = metrics.total_cum_eff[t - 1] + X[t]
        metrics.total_cum_tox[t] = metrics.total_cum_tox[t - 1] + Y[t]
        metrics.total_eff_regret[t] = metrics.total_eff_regret[t - 1] + curr_eff_regret
        metrics.total_tox_regret[t] = metrics.total_tox_regret[t - 1] + curr_tox_regret
        
        metrics.cum_eff[curr_s, t] = metrics.cum_eff[curr_s, t - 1] + X[t]
        metrics.cum_tox[curr_s, t] = metrics.cum_tox[curr_s, t - 1] + Y[t]
        metrics.eff_regret[curr_s, t] = metrics.eff_regret[curr_s, t - 1] + curr_eff_regret
        metrics.tox_regret[curr_s, t] = metrics.tox_regret[curr_s, t - 1] + curr_tox_regret
        
        for group_idx in range(S):
            if group_idx != curr_s:
                metrics.eff_regret[group_idx, t] = metrics.eff_regret[group_idx, t - 1]
                metrics.cum_eff[group_idx, t] = metrics.cum_eff[group_idx, t - 1] 
                metrics.cum_tox[group_idx, t] = metrics.cum_tox[group_idx, t - 1]
                metrics.eff_regret[group_idx, t] = metrics.eff_regret[group_idx, t - 1]

    else:
        metrics.total_cum_eff[t] = X[t]
        metrics.total_cum_tox[t] = Y[t]
        metrics.total_eff_regret[t] = curr_eff_regret
        metrics.total_tox_regret[t] = curr_tox_regret


        metrics.cum_eff[curr_s, t] = X[t]
        metrics.cum_tox[curr_s, t] = Y[t]
        metrics.eff_regret[curr_s, t] = curr_eff_regret
        metrics.tox_regret[curr_s, t] = curr_tox_regret

        for group_idx in range(S):
            if group_idx != curr_s:
                metrics.eff_regret[group_idx, t] = 0
                metrics.cum_eff[group_idx, t] = 0
                metrics.cum_tox[group_idx, t] = 0
                metrics.tox_regret[group_idx, t] = 0