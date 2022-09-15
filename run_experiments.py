import math
import numpy as np
from scipy.stats import beta 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import ExperimentMetrics
from dose_finding_models import TwoParamSharedModel, TanhModel, TwoParamModel, TwoParamAllSharedModel, OGTanhModel


np.random.seed(0)
class ExperimentRunner:
    def __init__(self, reps, num_doses, num_patients, num_subgroups, arr_rate, tox_thre, eff_thre, p_true, q_true, opt, learning_rate):
        self.reps = reps
        self.num_doses = num_doses
        self.num_patients = num_patients
        self.num_subgroups = num_subgroups
        self.arr_rate = arr_rate
        self.tox_thre = tox_thre
        self.eff_thre = eff_thre
        self.p_true = p_true
        self.q_true = q_true
        self.opt = opt
        self.learning_rate = learning_rate

    def gen_patients(self):
        '''
        Generates all patients for an experiment of length T
        '''
        # Arrival proportion of each subgroup. If arrive_rate = [5, 4, 3],
        # arrive_dist = [5/12, 4/12, 3/12]
        arrive_sum = sum(self.arr_rate)
        arrive_dist = [rate/arrive_sum for rate in self.arr_rate]
        arrive_dist.insert(0, 0)

        # [0, 5/12, 9/12, 12/12]
        arrive_dist_bins = np.cumsum(arrive_dist)

        # Random numbers between 0 and 1 in an array of shape (1, T)
        patients_gen = np.random.rand(self.num_patients)
        patients = np.digitize(patients_gen, arrive_dist_bins) - 1
        return patients


    def print_results(self, a_hat_fin_mean, out_metrics):
        num_patients = out_metrics.total_cum_eff.shape[0]
        efficacy = out_metrics.total_cum_eff[-1, :].mean()
        toxicity = out_metrics.total_cum_tox[-1, :].mean()

        metrics_frame = pd.DataFrame({'subgroup': list(np.arange(3)) + ['overall'],
                                      'dose error by person': list(out_metrics.regret_by_person.mean(axis=1)) + [out_metrics.regret[:, -1, :].sum(axis=0).mean() / num_patients], # incorrect dose assignments
                                      'efficacy regret': list(out_metrics.eff_regret[:, -1, :].mean(axis=1)) + [out_metrics.eff_regret[:, -1, :].mean()],
                                      'efficacy regret by person': list(out_metrics.eff_regret_by_person.mean(axis=1)) + [out_metrics.eff_regret[:, -1, :].mean() / num_patients],
                                      'efficacy by person': list(out_metrics.cum_eff_by_person.mean(axis=1)) + [efficacy / num_patients],
                                      'toxicity by person': list(out_metrics.cum_tox_by_person.mean(axis=1)) + [toxicity / num_patients],
                                      'final dose error': list(out_metrics.rec_err.mean(axis=1)) + [out_metrics.rec_err.mean()]}).T
        print(metrics_frame)

    def make_plots(self, dose_labels, out_metrics):
        plt.figure(figsize=(10, 8))
        sns.set_theme()

        # Subgroup plots
        # Dose toxicity for contextual model
        plt.subplot(331)
        subgroup_index = 0
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], out_metrics.a_hat_fin[subgroup_index, :],
                                        out_metrics.b_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(332)
        subgroup_index = 1
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], out_metrics.a_hat_fin[subgroup_index, :],
                                        out_metrics.b_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(333)
        subgroup_index = 2
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], out_metrics.a_hat_fin[subgroup_index, :],
                                        out_metrics.b_hat_fin[subgroup_index, :], out_metrics.p_hat[subgroup_index, :, :])
        

        plt.subplot(334)
        self.plot_over_time(self.reps, self.num_subgroups, self.num_patients, out_metrics.total_eff_regret, out_metrics.eff_regret, 'Regret')

        plt.subplot(335)
        self.plot_over_time(self.reps, self.num_subgroups, self.num_patients, out_metrics.total_cum_eff, out_metrics.cum_eff, 'Efficacy')

        plt.subplot(336)
        self.plot_over_time(self.reps, self.num_subgroups, self.num_patients, out_metrics.total_cum_tox, out_metrics.cum_tox, 'Toxicity')

        plt.subplot(337)
        self.plot_outcome(out_metrics.regret_by_person, 'Dose Error')

        # Efficacy and toxicity by patient
        plt.subplot(338)
        self.plot_efficacy_means(out_metrics.total_cum_eff, out_metrics.cum_eff, out_metrics.total_cum_tox,
                            out_metrics.cum_tox, out_metrics.pats_count, self.num_patients)

        plt.subplot(339)
        self.plot_outcome(out_metrics.safety_violations, 'Safety Constraint Violations')

        plt.tight_layout()
        plt.show()

    def plot_over_time(self, num_reps, num_groups, total_time, total_eff_regret, eff_regret, value_name):
        # row = time, column = trial
        # subgroups, time, trial
        regret_frame = pd.DataFrame(total_eff_regret)
        regret_frame = regret_frame.reset_index()
        # index = time
        regret_frame = pd.melt(regret_frame, id_vars=['index'], var_name='trial', value_name=value_name)
        regret_frame['group'] = np.repeat('all', num_reps * total_time)
        group_frames = []
        for group in range(num_groups):
            group_arr = eff_regret[group, :, :]
            group_frame = pd.DataFrame(group_arr)
            group_frame = group_frame.reset_index()
            group_frame = pd.melt(group_frame, id_vars=['index'], var_name='trial', value_name=value_name)
            group_frame['group'] = np.repeat(str(group), num_reps * total_time)
            group_frames.append(group_frame)
        regret_frame = pd.concat([regret_frame] + group_frames)
        sns.lineplot(data=regret_frame, x='index', y=value_name, hue='group')
        plt.xlabel('Time')
        plt.ylabel(value_name)

    def plot_outcome(self, outcome_vals, value_name):
        '''
        Recomended dose error plot - seaborn boxplot
        x-axis: subgroup
        y-axis: dose error
        '''
        frame = pd.DataFrame(outcome_vals)
        mean_frame = frame.mean(axis=0) 
        frame = frame.transpose()
        frame['all'] = mean_frame
        frame = frame.reset_index()
        frame = pd.melt(frame, id_vars=['index'], var_name='group', value_name=value_name)
        sns.pointplot(x='group', y=value_name, data=frame, join=False, kind='point')
        plt.ylim(0, 1.0)
        plt.xlabel('Subgroup')
        plt.ylabel(value_name)

    def plot_efficacy_means(self, total_cum_eff, cum_eff, total_cum_tox, cum_tox, pats_count, T):
        final_total_eff = total_cum_eff[-1, :] / T
        final_group_eff = cum_eff[:, -1, :]
        frame_dict = {'trial': np.arange(len(final_total_eff)),
                    'all': final_total_eff}
        for group in range(len(final_group_eff)):
            group_eff = final_group_eff[group, :]
            group_count = pats_count[group, :]
            frame_dict[str(group)] = group_eff / group_count
        frame = pd.DataFrame(frame_dict)
        frame = pd.melt(frame, id_vars=['trial'], var_name='group', value_name='Metric per Person')
        frame['metric'] = 'efficacy'

        final_total_tox = total_cum_tox[-1, :] / T
        final_group_tox = cum_tox[:, -1, :]
        tox_frame_dict = {'trial': np.arange(len(final_total_tox)),
                        'all': final_total_tox}
        for group in range(len(final_group_tox)):
            group_tox = final_group_tox[group, :]
            group_count = pats_count[group, :]
            tox_frame_dict[str(group)] = group_tox / group_count
        tox_frame = pd.DataFrame(tox_frame_dict)
        tox_frame = pd.melt(tox_frame, id_vars=['trial'], var_name='group', value_name='Metric per Person')
        tox_frame['metric'] = 'toxicity'

        final_frame = pd.concat([frame, tox_frame])

        sns.pointplot(x='group', y='Metric per Person', hue='metric', data=final_frame, join=False, kind='point')
        plt.xlabel('Subgroup')
        plt.ylabel('Metric per Person')

    def run_two_param(self, a0, b0):
        dose_skeleton = np.mean(self.p_true, axis=0)
        dose_skeleton_labels = TanhModel.initialize_dose_label(dose_skeleton, a0)

        p_rec = np.zeros((self.num_subgroups, self.num_patients, self.reps))
        metrics_objects = []

        for i in range(self.reps):
            print(f"Trial: {i}")
            # patients arrival generation
            pats = self.gen_patients()
            for tau in range(self.num_patients):
                p_rec[pats[tau], tau:, i] += 1
            dose_labels = np.zeros((self.num_subgroups, self.num_doses))
                
            model = TwoParamSharedModel(self.num_patients, self.num_subgroups, self.num_doses, pats, self.learning_rate, a0, b0)
            for s in range(self.num_subgroups):
                #dose_labels[s, :] = TwoParamSharedModel.initialize_dose_label(p_true[s, :], a0, b0)
                dose_labels[s, :] = TwoParamSharedModel.initialize_dose_label(dose_skeleton, a0, b0)
        
            run_metrics = model.run_model(self.tox_thre, self.eff_thre, self.p_true, self.q_true, self.opt, dose_labels)
            metrics_objects.append(run_metrics)
        
        exp_metrics = ExperimentMetrics(self.num_subgroups, self.num_doses, self.num_patients, self.reps, metrics_objects)

        a_hat_fin_mean = np.mean(exp_metrics.a_hat_fin, axis=1)
        p_hat_fin_mean = np.mean(exp_metrics.p_hat, axis=2)

        self.print_results(a_hat_fin_mean, exp_metrics)
        self.make_plots(dose_labels, exp_metrics)
    
    def run_one_param(self, model_type, a0):
        dose_skeleton = np.mean(self.p_true, axis=0)
        dose_skeleton_labels = TanhModel.initialize_dose_label(dose_skeleton, a0)

        p_rec = np.zeros((self.num_subgroups, self.num_patients, self.reps))
        metrics_objects = []

        for i in range(self.reps):
            print(f"Trial: {i}")
            # patients arrival generation
            pats = self.gen_patients()
            for tau in range(self.num_patients):
                p_rec[pats[tau], tau:, i] += 1
            dose_labels = np.zeros((self.num_subgroups, self.num_doses))
                
            model = model_type(self.num_patients, self.num_subgroups, self.num_doses, pats, self.learning_rate, a0)
            for s in range(self.num_subgroups):
                #dose_labels[s, :] = TwoParamSharedModel.initialize_dose_label(p_true[s, :], a0, b0)
                dose_labels[s, :] = TanhModel.initialize_dose_label(dose_skeleton, a0)
        
            run_metrics = model.run_model(self.tox_thre, self.eff_thre, self.p_true, self.q_true, self.opt, dose_labels)
            metrics_objects.append(run_metrics)
        
        exp_metrics = ExperimentMetrics(self.num_subgroups, self.num_doses, self.num_patients, self.reps, metrics_objects)

        a_hat_fin_mean = np.mean(exp_metrics.a_hat_fin, axis=1)
        p_hat_fin_mean = np.mean(exp_metrics.p_hat, axis=2)

        self.print_results(a_hat_fin_mean, exp_metrics)
        self.make_plots(dose_labels, exp_metrics)


def main():
    reps = 100 # num of simulated trials
    num_doses = 6
    num_patients = 30
    num_subgroups = 3
    arr_rate = [5, 4, 3]
    tox_thre = 0.35 # toxicity threshold
    eff_thre = 0.2 # efficacy threshold
    learning_rate = 0.01

    # p and q true value
    # toxicity
    p_true = np.array([[0.01, 0.01, 0.05, 0.15, 0.20, 0.45],
                    [0.01, 0.05, 0.15, 0.20, 0.45, 0.60],
                    [0.01, 0.05, 0.15, 0.20, 0.45, 0.60]])
    # efficacy
    q_true = np.array([[0.01, 0.02, 0.05, 0.10, 0.10, 0.10],
                    [0.10, 0.20, 0.30, 0.50, 0.60, 0.65],
                    [0.20, 0.50, 0.60, 0.80, 0.84, 0.85]])
    # optimal doses
    opt = np.array([6, 3, 3])
            
    runner = ExperimentRunner(reps, num_doses, num_patients, num_subgroups, arr_rate, tox_thre, eff_thre, p_true, q_true, opt, learning_rate)

    # a0 = 0.1
    # b0 = 0.1
    # runner.run_two_param(a0, b0)

    a0 = 0.5
    runner.run_one_param(OGTanhModel, a0)

if __name__ == "__main__":
    main()