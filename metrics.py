import numpy as np


class ExperimentMetrics:
    def __init__(self, S, K, T, reps, trial_metrics):
        self.total_cum_eff = np.zeros((T, reps))
        self.total_cum_tox = np.zeros((T, reps))
        self.cum_eff = np.zeros((S, T, reps))
        self.cum_tox = np.zeros((S, T, reps))

        self.p_hat = np.zeros((S, K, reps))
        self.q_mse_reps = np.zeros((S, K, T, reps))
        self.total_eff_regret = np.zeros((T, reps))
        self.eff_regret = np.zeros((S, T, reps))
        self.total_tox_regret = np.zeros((T, reps))
        self.tox_regret = np.zeros((S, T, reps))
        self.safety_violations = np.zeros((S, reps))
        self.regret = np.zeros((S, T, reps))

        self.rec = np.zeros((S, K+1))
        self.rec_err = np.zeros((S, reps))
        self.typeI = np.zeros(S)
        self.typeII = np.zeros(S)
        self.a_hat_fin = np.zeros((S, reps))
        self.b_hat_fin = np.zeros((S, reps))
        
        self.pats_count = np.zeros((S, reps))
        self.dose_err_by_person = np.zeros((S, reps))
        self.cum_eff_by_person = np.zeros((S, reps))
        self.cum_tox_by_person = np.zeros((S, reps))
        self.eff_regret_by_person = np.zeros((S, reps))
        self.regret_by_person = np.zeros((S, reps))

        self.incorporate_metrics(trial_metrics)

    def incorporate_metrics(self, trial_metrics):
        for idx, trial_metric_obj in enumerate(trial_metrics):
            self.total_cum_eff[:, idx] = trial_metric_obj.total_cum_eff
            self.total_cum_tox[:, idx] = trial_metric_obj.total_cum_tox
            self.cum_eff[:, :, idx] = trial_metric_obj.cum_eff
            self.cum_tox[:, :, idx] = trial_metric_obj.cum_tox

            self.p_hat[:, :, idx] = trial_metric_obj.p_hat
            self.q_mse_reps[:, :, :, idx] = trial_metric_obj.q_mse
            self.total_eff_regret[:, idx] = trial_metric_obj.total_eff_regret
            self.eff_regret[:, :, idx] = trial_metric_obj.eff_regret
            self.total_tox_regret[:, idx] = trial_metric_obj.total_tox_regret
            self.tox_regret[:, :, idx] = trial_metric_obj.tox_regret
            self.safety_violations[:, idx] = trial_metric_obj.safety_violations
            self.regret[:, :, idx] = trial_metric_obj.regret

            self.rec[:, :] = np.squeeze(self.rec[:, :]) + trial_metric_obj.rec
            self.rec_err[:, idx] = trial_metric_obj.rec_err
            self.typeI = self.typeI + trial_metric_obj.typeI
            self.typeII = self.typeII + trial_metric_obj.typeII
            self.a_hat_fin[:, idx] = trial_metric_obj.a_hat_fin
            self.b_hat_fin[:, idx] = trial_metric_obj.b_hat_fin

            self.pats_count[:, idx] = trial_metric_obj.pats_count
            self.dose_err_by_person[:, idx] = trial_metric_obj.dose_err_by_person
            self.cum_eff_by_person[:, idx] = trial_metric_obj.cum_eff_by_person
            self.cum_tox_by_person[:, idx] = trial_metric_obj.cum_tox_by_person
            self.eff_regret_by_person[:, idx] = trial_metric_obj.eff_regret_by_person
            self.regret_by_person[:, idx] = trial_metric_obj.regret_by_person


class TrialMetrics:
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
        self.regret = np.zeros((S, T))

        self.rec = np.zeros((S, K+1))
        self.rec_err = np.zeros(S)
        self.typeI = np.zeros(S)
        self.typeII = np.zeros(S)
        self.a_hat_fin = np.zeros(S)
        self.b_hat_fin = np.zeros(S)

        self.pats_count = np.zeros(S)
        self.dose_err_by_person = np.zeros(S)
        self.cum_eff_by_person = np.zeros(S)
        self.cum_tox_by_person = np.zeros(S)
        self.eff_regret_by_person = np.zeros(S)
        self.regret_by_person = np.zeros(S)
    
    def update_initial_metrics(self, num_subgroups, curr_s, regret, curr_eff, curr_tox, curr_eff_regret, curr_tox_regret):
        self.total_cum_eff[0] = curr_eff
        self.total_cum_tox[0] = curr_tox
        self.total_eff_regret[0] = curr_eff_regret
        self.total_tox_regret[0] = curr_tox_regret

        self.cum_eff[curr_s, 0] = curr_eff
        self.cum_tox[curr_s, 0] = curr_tox
        self.eff_regret[curr_s, 0] = curr_eff_regret
        self.tox_regret[curr_s, 0] = curr_tox_regret
        self.regret[curr_s, 0] = regret

        for group_idx in range(num_subgroups):
            if group_idx != curr_s:
                self.eff_regret[group_idx, 0] = 0
                self.cum_eff[group_idx, 0] = 0
                self.cum_tox[group_idx, 0] = 0
                self.tox_regret[group_idx, 0] = 0
                self.regret[group_idx, 0] = 0

    def update_metrics(self, timestep, num_subgroups, curr_s, regret, curr_eff, curr_tox, curr_eff_regret, curr_tox_regret):
        self.total_cum_eff[timestep] = self.total_cum_eff[timestep - 1] + curr_eff
        self.total_cum_tox[timestep] = self.total_cum_tox[timestep - 1] + curr_tox
        self.total_eff_regret[timestep] = self.total_eff_regret[timestep - 1] + curr_eff_regret
        self.total_tox_regret[timestep] = self.total_tox_regret[timestep - 1] + curr_tox_regret
        
        self.cum_eff[curr_s, timestep] = self.cum_eff[curr_s, timestep - 1] + curr_eff
        self.cum_tox[curr_s, timestep] = self.cum_tox[curr_s, timestep - 1] + curr_tox
        self.eff_regret[curr_s, timestep] = self.eff_regret[curr_s, timestep - 1] + curr_eff_regret
        self.tox_regret[curr_s, timestep] = self.tox_regret[curr_s, timestep - 1] + curr_tox_regret
        self.regret[curr_s, timestep] = self.regret[curr_s, timestep - 1] + regret
        
        for group_idx in range(num_subgroups):
            if group_idx != curr_s:
                self.eff_regret[group_idx, timestep] = self.eff_regret[group_idx, timestep - 1]
                self.cum_eff[group_idx, timestep] = self.cum_eff[group_idx, timestep - 1] 
                self.cum_tox[group_idx, timestep] = self.cum_tox[group_idx, timestep - 1]
                self.eff_regret[group_idx, timestep] = self.eff_regret[group_idx, timestep - 1]
                self.regret[group_idx, timestep] = self.regret[group_idx, timestep - 1]
