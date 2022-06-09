from audioop import avg
import math
import ssl
import numpy as np
from scipy.stats import beta 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import gen_patients, initialize_dose_label, get_toxicity
from c3t_budget import run_C3T_Budget, run_C3T_Budget_all, run_C3T, run_C3T_with_gradient_temp, run_C3T_Budget_all_with_gradient


np.random.seed(0)
class C3T_Metrics:
    def __init__(self, S, K, B, T, reps):
        self.rec = np.zeros((S, K+1))
        self.cum_eff = np.zeros(B)
        self.cum_tox = np.zeros(B)
        self.cum_s = np.zeros((S, T, reps))
        self.typeI = np.zeros(S)
        self.typeII = np.zeros(S)
        self.rec_err = np.zeros(S)
        self.a_hat_fin = np.zeros((S, reps))
        self.p_hat = np.zeros((S, K, reps))
        self.q_mse_reps = np.zeros((S, K, T, reps))

def plot_num_patients(B, T, subgroup_idx, subgroup_label, p_rec, cum_s):
    # out_arr = cum_s[subgroup_idx, :, :]
    # frame = pd.DataFrame(out_arr)
    # frame = frame.reset_index()
    # frame = frame.rename(columns={'index': 'timepoint'})
    # frame = pd.melt(frame, id_vars='timepoint', var_name='trial', value_name='num_patients')
    # sns.lineplot(data=frame, x='timepoint', y='num_patients')
    plt.plot(np.mean(cum_s[subgroup_idx, :, :], axis=1), '-r', label= 'C3T-Budget', linewidth=1.5)
    plt.plot(np.arange(B), np.mean(p_rec[subgroup_idx, :B, :], axis=1), '--k', label= '$\pi_s$', linewidth=1.5)
    plt.plot(np.arange(B, T), np.mean(p_rec[subgroup_idx, B:T, :], axis=1), '--k', linewidth=1.5)
    plt.title(subgroup_label)
    plt.ylim(0, B)
    plt.xlim(0, T)
    plt.ylabel('Number of patients')
    plt.legend()

def plot_mse(T, q_mse_reps, avg_over_doses=False, plot_one_dose=None):
    '''
    Plot MSE averaged across dose arms
    '''
    # Mean over trials
    q_mse_means = np.mean(q_mse_reps, axis=2)

    if avg_over_doses:
        q_mse_means = np.mean(q_mse_means, axis=0)
        plt.plot(q_mse_means)
    else:
        frame = pd.DataFrame(q_mse_means)
        frame = frame.reset_index().rename(columns={'index': 'dose'})
        frame = pd.melt(frame, id_vars='dose', var_name='timepoint', value_name='q_mse')
        frame['dose'] = frame['dose'].astype(str)
        if plot_one_dose is not None:
            frame = frame[frame['dose'] == plot_one_dose]
        sns.lineplot(data=frame, x='timepoint', y='q_mse', hue='dose')

    plt.ylim(0, 0.2)
    plt.xlim(0, T)
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.legend()

def plot_dose_skeleton(dose_labels, dose_skeleton):
    plt.plot(dose_labels, dose_skeleton, '--ko')
    plt.ylim(0, 1)
    plt.xlabel('Dose labels')
    plt.ylabel('Toxicity')
    plt.legend()

def plot_dose_toxicity_curve(dose_labels, p_true, a_hat_fin, p_empirical):
    num_reps = a_hat_fin.shape[0]
    num_doses = dose_labels.shape[0]
    model_toxicities = np.array([get_toxicity(dose_labels, a_hat_fin[i]) for i in range(num_reps)]).flatten()
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

    # frame = pd.DataFrame({'dose_labels': dose_labels, 'true': p_true,
    #                       'model': get_toxicity(dose_labels, a_hat_fin),
    #                       'empirical': p_empirical})
    # frame = pd.melt(frame, id_vars='dose_labels', var_name='toxicity', value_name='toxicity_value')
    sns.lineplot(data=frame, x='dose_labels', y='toxicity_value', hue='toxicity', style='toxicity', markers=True)
    plt.xlim(-5, 0)
    plt.ylim(0, 1)
    plt.xlabel('Dose labels')
    plt.ylabel('Toxicity')

def plot_dose_toxicity_curve_with_skeleton(dose_labels, p_true, a_hat_fin, p_empirical, dose_skeleton_labels, dose_skeleton):
    num_reps = a_hat_fin.shape[0]
    num_doses = dose_labels.shape[0]
    model_toxicities = np.array([get_toxicity(dose_skeleton_labels, a_hat_fin[i]) for i in range(num_reps)]).flatten()
    frame = pd.DataFrame({'trial': np.repeat(np.arange(num_reps), num_doses),
                          'dose_labels': np.tile(dose_skeleton_labels, num_reps), 
                          'model': model_toxicities,
                          'empirical': p_empirical.flatten('F')})
    frame = pd.melt(frame, id_vars=['trial', 'dose_labels'], var_name='toxicity', value_name='toxicity_value')

    # frame = pd.DataFrame({'dose_labels': dose_skeleton_labels, 'dose_skeleton': dose_skeleton,
    #                        'model': get_toxicity(dose_skeleton_labels, a_hat_fin),
    #                        'empirical': p_empirical})
    # frame = pd.melt(frame, id_vars='dose_labels', var_name='toxicity', value_name='toxicity_value')
    frame2 = pd.DataFrame({'trial': np.repeat(0, num_doses),
                           'dose_labels': dose_labels,
                           'toxicity': np.repeat('true', num_doses),
                           'toxicity_value': p_true})
    frame = pd.concat([frame, frame2])
    sns.lineplot(data=frame, x='dose_labels', y='toxicity_value', hue='toxicity', style='toxicity', markers=True)
    plt.xlim(-5, 0)
    plt.ylim(0, 0.8)
    plt.xlabel('Dose labels')
    plt.ylabel('Toxicity')


def main():
    reps = 100 # num of simulated trials
    K = 6
    B = 400
    T = 1200
    S = 3
    arr_rate = [5, 4, 3]
    tox_thre = 0.35 # toxicity threshold
    eff_thre = 0.2 # efficacy threshold
    a0 = 0.5

    # p and q true value
    # toxicity
    p_true = np.array([[0.01, 0.01, 0.05, 0.15, 0.20, 0.45],
                      [0.01, 0.05, 0.15, 0.20, 0.45, 0.60],
                      [0.01, 0.05, 0.15, 0.20, 0.45, 0.60]])
    # efficacy
    q_true = np.array([[0.01, 0.02, 0.05, 0.10, 0.10, 0.10],
                      [0.10, 0.20, 0.30, 0.50, 0.60, 0.65],
                      [0.20, 0.50, 0.60, 0.80, 0.84, 0.85]])
    
    dose_skeleton = np.mean(p_true, axis=0)
    dose_skeleton_labels = initialize_dose_label(dose_skeleton, a0)

    # Initialize actual dose levels
    dose_labels = np.zeros((S, K))
    for s in range(S):
        dose_labels[s, :] = initialize_dose_label(p_true[s, :], a0)
    
    # optimal doses
    opt = np.array([6, 3, 3])

    p_rec = np.zeros((S, T, reps))
    out_metrics = C3T_Metrics(S, K, B, T, reps)

    for i in range(reps):
        print(f"Trial: {i}")
        # patients arrival generation
        pats = gen_patients(T, arr_rate)
        for tau in range(T):
            p_rec[pats[tau], tau:, i] += 1
        
        # C3T-Budget

        # rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat = run_C3T_Budget(T, B, S, K, pats, arr_rate,
        #                                                                         tox_thre, eff_thre, p_true,
        #                                                                         q_true, opt, dose_labels, no_skip=False)
        
        # rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat = run_C3T(T, B, S, K, pats, arr_rate,
        #                                                                         tox_thre, eff_thre, p_true,
        #                                                                         q_true, opt, dose_labels)
        # rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat = run_C3T_with_gradient_temp(T, B, S, K, pats, arr_rate,
        #                                                                         tox_thre, eff_thre, p_true,
        #                                                                         q_true, opt, dose_labels)
        # rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat = run_C3T_Budget_all(T, B, S, K, pats, arr_rate,
        #                                                                     tox_thre, eff_thre, p_true,
        #                                                                     q_true, opt, dose_skeleton_labels)
        rec, cum_eff, cum_tox, cum_s, typeI, typeII, q_mse, rec_err, a_hat_fin, p_hat = run_C3T_Budget_all_with_gradient(T, B, S, K, pats, arr_rate,
                                                                             tox_thre, eff_thre, p_true,
                                                                             q_true, opt, dose_skeleton_labels)

        out_metrics.rec[:, :] = np.squeeze(out_metrics.rec[:, :]) + rec
        out_metrics.cum_eff[:] = out_metrics.cum_eff[:] + cum_eff
        out_metrics.cum_tox[:] = out_metrics.cum_tox[:] + cum_tox
        out_metrics.cum_s[:, :, i] = np.squeeze(out_metrics.cum_s[:, :, i]) + cum_s
        out_metrics.typeI[:] = out_metrics.typeI[:] + typeI
        out_metrics.typeII[:] = out_metrics.typeII[:] + typeII
        out_metrics.rec_err[:] = out_metrics.rec_err[:] + rec_err
        out_metrics.a_hat_fin[:, i] = a_hat_fin
        out_metrics.p_hat[:, :, i] = p_hat
        out_metrics.q_mse_reps[:, :, :, i] = q_mse

    a_hat_fin_mean = np.mean(out_metrics.a_hat_fin, axis=1)
    p_hat_fin_mean = np.mean(out_metrics.p_hat, axis=2)

    # Print results
    print("== Recommended dose error rates ==============")
    print(f"Algorithm |  SG1  |  SG2  |  SG3  | Total |")
    print(f"C3T-Budget | {out_metrics.rec_err[0] / reps} | {out_metrics.rec_err[1] / reps} | {out_metrics.rec_err[2] / reps} | {np.mean(out_metrics.rec_err) / reps}")

    typeI = np.mean(out_metrics.typeI / reps)
    typeII = np.mean(out_metrics.typeII / reps)

    print("== Safe dose estimation error rates ===========")
    print("Algorithm |  Type-I  |  Type-II |  Total   |")
    print(f"C3T-Budget | {typeI} | {typeII} | {(typeI + typeII) / 2}")

    efficacy = np.max(out_metrics.cum_eff) / B / reps
    toxicity = np.max(out_metrics.cum_tox) / B / reps
    print("== Efficacy and toxicity per patient =======")
    print("Algorithm |   Efficacy   |   Toxicity   |")
    print(f"C3T_Budget | {efficacy} | {toxicity}")

    plt.figure(figsize=(10, 8))
    sns.set_theme()

    # Subgroup plots
    plt.subplot(331)
    plot_num_patients(B, T, 0, 'Subgroup 1', p_rec, out_metrics.cum_s)

    plt.subplot(332)
    plot_num_patients(B, T, 1, 'Subgroup 2', p_rec, out_metrics.cum_s)

    plt.subplot(333)
    plot_num_patients(B, T, 2, 'Subgroup 3', p_rec, out_metrics.cum_s)

    plt.subplot(334)
    plot_mse(T, out_metrics.q_mse_reps[0, :, :, :], avg_over_doses=True)

    plt.subplot(335)
    plot_mse(T, out_metrics.q_mse_reps[1, :, :, :], avg_over_doses=False, plot_one_dose=str(3))

    plt.subplot(336)
    plot_mse(T, out_metrics.q_mse_reps[2, :, :, :], avg_over_doses=False, plot_one_dose=str(3))

    # plt.subplot(337)
    # subgroup_index = 0
    # plot_dose_toxicity_curve(dose_labels[subgroup_index], p_true[subgroup_index],
    #                          out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])

    # plt.subplot(338)
    # subgroup_index = 1
    # plot_dose_toxicity_curve(dose_labels[subgroup_index], p_true[subgroup_index],
    #                          out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])

    # plt.subplot(339)
    # subgroup_index = 2
    # plot_dose_toxicity_curve(dose_labels[subgroup_index], p_true[subgroup_index],
    #                          out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])

    # Plots for combined (non-contextual) model
    plt.subplot(337)
    subgroup_index = 0
    plot_dose_toxicity_curve_with_skeleton(dose_labels[subgroup_index], p_true[subgroup_index],
                             out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :], 
                             dose_skeleton_labels, dose_skeleton)

    plt.subplot(338)
    subgroup_index = 1
    plot_dose_toxicity_curve_with_skeleton(dose_labels[subgroup_index], p_true[subgroup_index],
                             out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :], 
                             dose_skeleton_labels, dose_skeleton)

    plt.subplot(339)
    subgroup_index = 2
    plot_dose_toxicity_curve_with_skeleton(dose_labels[subgroup_index], p_true[subgroup_index],
                             out_metrics.a_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :], 
                             dose_skeleton_labels, dose_skeleton)


    plt.tight_layout()
    plt.show()

main()
# print(get_expected_improvement(1, 3))
# print(alpha_func(0.1, 6, 0.35, 15))
# print(get_ucb(0.5, 0.5, 7, 11))
# print(gen_patients(10, [5, 4, 3]))