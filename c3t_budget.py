import math
import ssl
import numpy as np
from scipy.stats import beta 
import pandas as pd
from helpers import alpha_func, get_expected_improvement, get_ucb, get_toxicity


def run_C3T_Budget(T, B, S, K, pats, arr_rate, tox_thre, eff_thre, p_true, q_true, opt_ind, dose_labels, no_skip=False):
    '''
    Input
    T        : time-horizon
    B        : budget
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
    cum_s    : cumulative recruitment of each subgroup
    typeI    : type-I error for dose safety
    typeII   : type-II error for dose safety
    q_mse    : mean squared error of efficacy
    ''' 
    a0 = 1/2
    a_max = 1
    c = 0.5
    delta = 1 / B * S * np.ones(S)
    eff_w = 0.6


    q_mse = np.zeros((S, K, T))
    cum_eff = np.zeros(B)
    cum_tox = np.zeros(B)

    # Define variables and parameters
    arrive_sum = sum(arr_rate)
    arrive_dist = [rate/arrive_sum for rate in arr_rate]

    t = 0 # timestep
    current_budget = 0 # budget

    s_arrive = np.zeros(S)
    n_choose = np.zeros((S, K)) # num of selection
    p_hat = np.zeros((S, K)) # estimated toxicity
    q_bar = np.zeros((S, K)) # estimated efficacy
    q_hat = np.zeros((S, K)) # estimate efficacy ucb

    alpha = np.zeros(S)
    D = np.zeros((S, K)) # Matrix for set of doses allowed
    a_hat = np.zeros((S, T)) # estimated overall a
    ak_hat = np.ones((S, K)) * a0 # estimated invididual a
    all_a_hat = np.ones(K) * a0 

    # Variables for expected improvement
    q_a = np.ones((S, K))
    q_b = np.ones((S, K))
    q_ei = np.zeros((S, K))
    q_ei_opt = np.zeros((S, K, T))
    g_ei_p = np.zeros((S, T))

    I = np.ones(T, dtype=int) * -1 # allocated doses
    X = np.zeros(T)
    Y = np.zeros(T)
    H = pats
    Z = np.zeros(T)

    while t < T and current_budget < B:
        # For each subgroup, select dose with largest value in n_choose (across doses) 
        for s in range(S):
            I_est = np.argmax(n_choose[s, :]) 
            a_hat[s, t] = ak_hat[s, I_est]
        q_ei_opt[:, :, t] = q_ei
        q_mse[:, :, t] = np.abs(q_true - q_bar)**2

        curr_s = H[t]
        s_arrive[curr_s] += 1

        # Initialize / burn-in
        if s_arrive[H[t]] < K: 
            I[t] = int(s_arrive[H[t]]) # allocated dose
            n_choose[curr_s, I[t]] = 1
            X[t] = np.random.rand() <= q_true[curr_s, I[t]] # Sample efficacy
            Y[t] = np.random.rand() <= p_true[curr_s, I[t]] # Sample toxicity
            q_bar[curr_s, I[t]] = X[t]
            q_a[curr_s, I[t]] += X[t]
            q_b[curr_s, I[t]] += 1 - X[t]

            for k in range(K):
                q_hat[curr_s, k] = get_ucb(q_bar[curr_s, k], c, n_choose[curr_s, k], np.sum(n_choose[curr_s, :]))
            
            q_ei[curr_s, I[t]] = get_expected_improvement(q_a[curr_s, I[t]], q_b[curr_s, I[t]])
            p_hat[curr_s, I[t]] = Y[t]
            Z[t] = 1
            
            if current_budget > 0:
                cum_eff[current_budget] = cum_eff[current_budget - 1] + X[t]
                cum_tox[current_budget] = cum_tox[current_budget - 1] + Y[t]
            current_budget += 1
            

        # Normal loop
        else:
            rho = (B - current_budget) / (T - t)
            ## LP calculations
            # table used to sort subgroups for LP calculations
            # table[s, 0] = expected improvement of credible interval by subgroup
            # table[s, 1] = UCB of efficacy estimates by subgroup
            # table[s, 2] = subgroup indices
            # table[s, 3] = arrival ratios by subgroup

            table = np.zeros((S, 4))

            for s in range(S):
                # Calculate alpha
                alpha[s] = alpha_func(dose_labels[s, :], K, delta[s], np.sum(n_choose[s, :])) 
                # Available set of doses
                D[s, :] = get_toxicity(dose_labels[s, :], a_hat[s, t] + alpha[s]) <= tox_thre

                # Max value of efficacy UCB (q_hat) multiplied by 
                Ix = np.max(q_hat[s, :] * D[s, :])
                # Max idx
                Ii = np.argmax(q_hat[s, :] * D[s, :])

                if q_bar[s, Ii] * eff_w + Ix * (1 - eff_w) >= eff_thre:
                    table[s, 0] = q_ei[s, Ii]
                else:
                    table[s, 0] = 0
                
                g_ei_p[s, t] = q_ei[s, Ii]
                table[s, 1] = q_hat[s, Ii]
            
            table[:, 2] = np.arange(S) # subgroup index
            table[:, 3] = arrive_dist # arrival ratios
            sorted_indices = np.lexsort((-table[:, 1], -table[:, 0]))
            table = table[sorted_indices]

            # tilde_s = max subgroup given sum of arrival rates up to it is less than the average budget
            tilde_s = 0
            tmp_dist = 0
            while tmp_dist <= rho and tilde_s < S:
                tmp_dist += table[tilde_s, 3]
                tilde_s += 1

            # tilde_s is the tilde_s in the paper 
            tilde_s -= 1

            # LP_solution
            dose_prob = np.zeros(S) # psi vector of probabilities that the agent does not skip patient in subgroup s

            if tilde_s > 0:
                dose_prob[table[:tilde_s + 1, 2].astype(int)] = 1
                if tilde_s + 1 < S:
                    dose_prob[table[tilde_s + 1, 2].astype(int)] = (rho - np.sum(table[:tilde_s + 1, 3])) / table[tilde_s + 1, 3]
                if tilde_s + 2 < S: # todo: not sure about this
                    dose_prob[table[tilde_s + 2:, 2].astype(int)] = 0

            # Determine whether to skip patient or not
            use_patient = dose_prob[curr_s] >= np.random.rand()
            if no_skip:
                use_patient = True

            # Use dose
            if use_patient: 
                Ii = np.argmax(q_hat[curr_s, :] * D[curr_s, :])
                I[t] = Ii
                X[t] = np.random.rand() <= q_true[curr_s, I[t]]
                Y[t] = np.random.rand() <= p_true[curr_s, I[t]]

                q_bar[curr_s, I[t]] = (q_bar[curr_s, I[t]] * n_choose[curr_s, I[t]] + X[t]) / (n_choose[curr_s, I[t]] + 1)
                p_hat[curr_s, I[t]] = (p_hat[curr_s, I[t]] * n_choose[curr_s, I[t]] + Y[t]) / (n_choose[curr_s, I[t]] + 1)
                n_choose[curr_s, I[t]] += 1

                Z[t] = 1
                q_a[curr_s, I[t]] += X[t]
                q_b[curr_s, I[t]] += 1 - X[t]

                for k in range(K):
                    q_hat[curr_s, k] = get_ucb(q_bar[curr_s, k], c, n_choose[curr_s, k], np.sum(n_choose[curr_s, :]))
                
                q_ei[curr_s, I[t]] = get_expected_improvement(q_a[curr_s, I[t]], q_b[curr_s, I[t]])
                if current_budget > 0:
                    cum_eff[current_budget] = cum_eff[current_budget - 1] + X[t]
                    cum_tox[current_budget] = cum_tox[current_budget - 1] + Y[t]
                current_budget += 1
                
            # Skip patient
            else: 
                X[t] = -1
                Y[t] = -1
                Z[t] = 0

        # If a dose has been allocated, update alpha params
        if I[t] != -1: 
            ak_hat[curr_s, I[t]] = np.log(p_hat[curr_s, I[t]]) / \
                                   np.log((np.tanh(dose_labels[curr_s, I[t]]) + 1.) / 2.)
            if ak_hat[curr_s, I[t]] > a_max:
                ak_hat[curr_s, I[t]] = a_max
        
        t += 1

    rec, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin = finalize_results(T, t, K, S, X, H, n_choose, ak_hat, dose_labels,
                     tox_thre, eff_thre, q_mse, q_bar, p_true, opt_ind)
    return rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat


def run_C3T_Budget_all(T, B, S, K, pats, arr_rate, tox_thre, eff_thre, p_true, q_true, opt_ind, dose_labels):
    '''
    Input
    T        : time-horizon
    B        : budget
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
    cum_s    : cumulative recruitment of each subgroup
    typeI    : type-I error for dose safety
    typeII   : type-II error for dose safety
    q_mse    : mean squared error of efficacy
    ''' 
    a0 = 1/2
    a_max = 1
    c = 0.5
    delta = 1 / B
    eff_w = 0.6

    q_mse = np.zeros((S, K, T))
    cum_eff = np.zeros(B)
    cum_tox = np.zeros(B)

    # Define variables and parameters
    arrive_sum = sum(arr_rate)
    arrive_dist = [rate/arrive_sum for rate in arr_rate]

    t = 0 # timestep
    current_budget = 0 # budget

    s_arrive = 0
    n_choose = np.zeros((S, K)) # num of selection
    n_choose_all = np.zeros(K)
    p_hat = np.zeros(K) # estimated toxicity
    q_bar = np.zeros(K) # estimated efficacy
    q_hat = np.zeros(K) # estimate efficacy ucb

    alpha = 0
    dose_set = np.zeros(K) # Matrix for set of doses allowed
    a_hat = np.zeros(T) # estimated overall a
    ak_hat = np.ones(K) * a0 # estimated invididual a

    # Variables for expected improvement
    q_a = np.ones(K)
    q_b = np.ones(K)
    q_ei = np.zeros(K)

    I = np.ones(T, dtype=int) * -1 # allocated doses
    X = np.zeros(T)
    Y = np.zeros(T)
    H = pats
    Z = np.zeros(T)

    while t < T and current_budget < B:
        # For each subgroup, select dose with largest value in n_choose (across doses) 
        I_est = np.argmax(n_choose_all)
        a_hat[t] = ak_hat[I_est]

        for s in range(S):
            q_mse[s, :, t] = np.abs(q_true[s, :] - q_bar)**2

        curr_s = H[t]
        s_arrive += 1

        # Initialize / burn-in
        if s_arrive < K: 
            I[t] = int(s_arrive)
            n_choose[curr_s, I[t]] = 1
            n_choose_all[I[t]] = 1

            X[t] = np.random.rand() <= q_true[curr_s, I[t]] # Sample efficacy
            Y[t] = np.random.rand() <= p_true[curr_s, I[t]] # Sample toxicity

            q_bar[I[t]] = X[t]
            q_a[I[t]] += X[t]
            q_b[I[t]] += 1 - X[t]

            for k in range(K):
                q_hat[k] = get_ucb(q_bar[k], c, n_choose_all[k], np.sum(n_choose_all))
            
            q_ei[I[t]] = get_expected_improvement(q_a[I[t]], q_b[I[t]])
            p_hat[I[t]] = Y[t]
            Z[t] = 1
            
            if current_budget > 0:
                cum_eff[current_budget] = cum_eff[current_budget - 1] + X[t]
                cum_tox[current_budget] = cum_tox[current_budget - 1] + Y[t]
            current_budget += 1

        # Normal loop
        else:
            rho = (B - current_budget) / (T - t)
            ## LP calculations
            # table used to sort subgroups for LP calculations
            # table[s, 0] = expected improvement of credible interval by subgroup
            # table[s, 1] = UCB of efficacy estimates by subgroup
            # table[s, 2] = subgroup indices
            # table[s, 3] = arrival ratios by subgroup

            table = np.zeros((S, 4))

            # Calculate alpha
            alpha = alpha_func(dose_labels, 1, delta, np.sum(n_choose_all)) 
            # Available set of doses
            dose_set = get_toxicity(dose_labels, a_hat[t] + alpha) <= tox_thre

            # Max value of efficacy UCB (q_hat) multiplied by 
            Ix = np.max(q_hat * dose_set)
            # Max idx
            Ii = np.argmax(q_hat * dose_set)

            # if q_bar[Ii] * eff_w + Ix * (1 - eff_w) >= eff_thre:
            #     table[:, 0] = q_ei[Ii]
            # else:
            #     table[:, 0] = 0
            
            # table[:, 1] = q_hat[Ii]
            # table[:, 2] = np.arange(S) # subgroup index
            # table[:, 3] = arrive_dist # arrival ratios
            # sorted_indices = np.lexsort((-table[:, 1], -table[:, 0]))
            # table = table[sorted_indices]

            # # tilde_s = max subgroup given sum of arrival rates up to it is less than the average budget
            # tilde_s = 0
            # tmp_dist = 0
            # while tmp_dist <= rho and tilde_s < S:
            #     tmp_dist += table[tilde_s, 3]
            #     tilde_s += 1

            # # tilde_s is the tilde_s in the paper 
            # tilde_s -= 1

            # # LP_solution
            # dose_prob = np.zeros(S) # psi vector of probabilities that the agent does not skip patient in subgroup s

            # if tilde_s > 0:
            #     dose_prob[table[:tilde_s + 1, 2].astype(int)] = 1
            #     if tilde_s + 1 < S:
            #         dose_prob[table[tilde_s + 1, 2].astype(int)] = (rho - np.sum(table[:tilde_s + 1, 3])) / table[tilde_s + 1, 3]
            #     if tilde_s + 2 < S: # todo: not sure about this
            #         dose_prob[table[tilde_s + 2:, 2].astype(int)] = 0

            # Determine whether to skip patient or not
            # Use dose
            # if dose_prob[curr_s] >= np.random.rand(): 
            if True:
                Ii = np.argmax(q_hat * dose_set)
                I[t] = Ii
                X[t] = np.random.rand() <= q_true[curr_s, I[t]]
                Y[t] = np.random.rand() <= p_true[curr_s, I[t]]

                q_bar[I[t]] = (q_bar[I[t]] * n_choose_all[I[t]] + X[t]) / (n_choose_all[I[t]] + 1)
                p_hat[I[t]] = (p_hat[I[t]] * n_choose_all[I[t]] + Y[t]) / (n_choose_all[I[t]] + 1)
                n_choose[curr_s, I[t]] += 1
                n_choose_all[I[t]] += 1

                Z[t] = 1
                q_a[I[t]] += X[t]
                q_b[I[t]] += 1 - X[t]

                for k in range(K):
                    q_hat[k] = get_ucb(q_bar[k], c, n_choose_all[k], np.sum(n_choose_all))
                
                q_ei[I[t]] = get_expected_improvement(q_a[I[t]], q_b[I[t]])
                if current_budget > 0:
                    cum_eff[current_budget] = cum_eff[current_budget - 1] + X[t]
                    cum_tox[current_budget] = cum_tox[current_budget - 1] + Y[t]
                current_budget += 1
                
            # # Skip patient
            # else: 
            #     X[t] = -1
            #     Y[t] = -1
            #     Z[t] = 0

        # If a dose has been allocated, update alpha params
        if I[t] != -1: 
            ak_hat[I[t]] = np.log(p_hat[I[t]]) / \
                                  np.log((np.tanh(dose_labels[I[t]]) + 1.) / 2.)
            if ak_hat[I[t]] > a_max:
                ak_hat[I[t]] = a_max
        
        t += 1

    rec, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin = finalize_results(T, t, K, S, X, H, n_choose_all, ak_hat, dose_labels,
                     tox_thre, eff_thre, q_mse, q_bar, p_true, opt_ind, uses_context=False)
    p_hat = np.tile(p_hat, (S, 1))
    return rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat


def finalize_results(T, t, K, S, X, H, n_choose, ak_hat, dose_labels,
                     tox_thre, eff_thre, q_mse, q_bar, p_true, opt_ind,
                     uses_context=True):
    # Recommendation and observe results
    rec = np.zeros((S, K+1))
    cum_s = np.zeros((S, T))
    rec_err = np.zeros(S)
    typeI = np.zeros(S)
    typeII = np.zeros(S)
    a_hat_fin = np.zeros(S)
    p_out = np.zeros((S, K))


    for s in range(S):
        if uses_context:
            I_est = np.argmax(n_choose[s, :])
            a_hat_fin[s] = ak_hat[s, I_est]
            p_out[s, :] = get_toxicity(dose_labels[s, :], a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar[s, :] * (p_out[s, :] <= tox_thre)
        else:
            I_est = np.argmax(n_choose)
            a_hat_fin[s] = ak_hat[I_est]
            p_out[s, :] = get_toxicity(dose_labels, a_hat_fin[s])
            # Dose with max empirical efficacy also below toxicity threshold
            q_below_tox_thre = q_bar * (p_out[s, :] <= tox_thre)
        max_dose_val = np.max(q_below_tox_thre)
        max_dose_idx = np.argmax(q_below_tox_thre)

        # If the recommended dose is above efficacy threshold, dose rec is correct
        if max_dose_val >= eff_thre:
            rec[s, max_dose_idx] = 1
        else:
            rec[s, K] = 1
            max_dose_idx = K

        if max_dose_idx != opt_ind[s]:
            rec_err[s] = 1
        
        for i in range(K):
            if p_true[s, i] <= tox_thre and p_out[s, i] > tox_thre:
                typeI[s] += 1
            else:
                if p_true[s, i] > tox_thre and p_out[s, i] <= tox_thre:
                    typeII[s] += 1
                    
    if t < T:
        q_mse[:, :, t:] = np.tile(np.expand_dims(q_mse[:, :, t-1], axis=2), (1, 1, T-t))  

    for tau in range(t):
        if X[tau] > -1:
            cum_s[H[tau], tau:]+= 1

    typeI = typeI / K
    typeII = typeII / K

    return rec, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin

# def run_C3T_budget_with_gradient():
#     pass