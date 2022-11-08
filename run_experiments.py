import math
import numpy as np
from scipy.stats import beta 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import ExperimentMetrics
from dose_finding_models import TwoParamSharedModel, TanhModel, TwoParamModel, TwoParamAllSharedModel, OGTanhModel
from data_generation import TrialPopulationScenarios, DoseFindingScenarios


np.random.seed(0)
class ExperimentRunner:
    def __init__(self, reps, patient_scenario, dose_scenario, num_patients, learning_rate):
        self.reps = reps
        self.patient_scenario = patient_scenario
        self.dose_scenario = dose_scenario
        self.num_patients = num_patients
        self.learning_rate = learning_rate

    def print_results(self, out_metrics):
        num_patients = out_metrics.total_cum_eff.shape[0]
        efficacy = out_metrics.total_cum_eff[-1, :].mean()
        toxicity = out_metrics.total_cum_tox[-1, :].mean()
        metrics_frame = pd.DataFrame({'subgroup': list(np.arange(self.patient_scenario.num_subgroups)) + ['overall'],
                                      'dose error by person': list(out_metrics.regret_by_person.mean(axis=1)) + [out_metrics.regret[:, -1, :].sum(axis=0).mean() / num_patients], # incorrect dose assignments
                                      'efficacy regret': list(out_metrics.eff_regret[:, -1, :].mean(axis=1)) + [out_metrics.eff_regret[:, -1, :].mean()],
                                      'efficacy regret by person': list(out_metrics.eff_regret_by_person.mean(axis=1)) + [out_metrics.eff_regret[:, -1, :].mean() / num_patients],
                                      'efficacy by person': list(out_metrics.cum_eff_by_person.mean(axis=1)) + [efficacy / num_patients],
                                      'toxicity by person': list(out_metrics.cum_tox_by_person.mean(axis=1)) + [toxicity / num_patients],
                                      'final dose error': list(out_metrics.rec_err.mean(axis=1)) + [out_metrics.rec_err.mean()]}).T
        print(metrics_frame)
        print(f"a: {out_metrics.a_hat_fin.mean(axis=1)}")
        print(f"b: {out_metrics.b_hat_fin.mean(axis=1)}")

    def make_plots(self, dose_labels, out_metrics):
        plt.subplot(334)
        self.plot_over_time(self.reps, self.patient_scenario.num_subgroups, self.num_patients, out_metrics.total_eff_regret, out_metrics.eff_regret, 'Regret')

        plt.subplot(335)
        self.plot_over_time(self.reps, self.patient_scenario.num_subgroups, self.num_patients, out_metrics.total_cum_eff, out_metrics.cum_eff, 'Efficacy')

        plt.subplot(336)
        self.plot_over_time(self.reps, self.patient_scenario.num_subgroups, self.num_patients, out_metrics.total_cum_tox, out_metrics.cum_tox, 'Toxicity')

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
        sns.pointplot(x='group', y=value_name, data=frame, join=False)
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

        sns.pointplot(x='group', y='Metric per Person', hue='metric', data=final_frame, join=False)
        plt.xlabel('Subgroup')
        plt.ylabel('Metric per Person')

    def run_two_param(self, model_type, a0, b0):
        dose_skeleton = np.mean(self.p_true, axis=0)
        dose_skeleton_labels = TanhModel.initialize_dose_label(dose_skeleton, a0)

        p_rec = np.zeros((self.num_subgroups, self.num_patients, self.reps))
        metrics_objects = []

        for i in range(self.reps):
            print(f"Trial: {i}")
            # patients arrival generation
            pats = self.patient_scenario.generate_samples(self.num_patients)
            for tau in range(self.num_patients):
                p_rec[pats[tau], tau:, i] += 1
            dose_labels = np.zeros((self.num_subgroups, self.num_doses))
                
            model = model_type(self.num_patients, self.num_subgroups, self.num_doses, pats, self.learning_rate, a0, b0)
            for s in range(self.num_subgroups):
                #dose_labels[s, :] = TwoParamSharedModel.initialize_dose_label(p_true[s, :], a0, b0)
                dose_labels[s, :] = TwoParamSharedModel.initialize_dose_label(dose_skeleton, a0, b0)
        
            run_metrics = model.run_model(self.tox_thre, self.eff_thre, self.p_true, self.q_true, self.opt, dose_labels)
            metrics_objects.append(run_metrics)
        
        exp_metrics = ExperimentMetrics(self.num_subgroups, self.num_doses, self.num_patients, self.reps, metrics_objects)

        p_hat_fin_mean = np.mean(exp_metrics.p_hat, axis=2)

        self.print_results(exp_metrics)
        plt.figure(figsize=(10, 8))
        sns.set_theme()

        # Subgroup plots
        # Dose toxicity for contextual model
        plt.subplot(331)
        subgroup_index = 0
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                        exp_metrics.b_hat_fin[subgroup_index, :], exp_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(332)
        subgroup_index = 1
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                        exp_metrics.b_hat_fin[subgroup_index, :], exp_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(333)
        subgroup_index = 2
        TwoParamSharedModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.p_true[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                        exp_metrics.b_hat_fin[subgroup_index, :], exp_metrics.p_hat[subgroup_index, :, :])
        

        self.make_plots(dose_labels, exp_metrics)
    
    def run_one_param(self, model_type, a0):
        metrics_objects = []

        for i in range(self.reps):
            print(f"Trial: {i}")
            pats = self.patient_scenario.generate_samples(self.num_patients)
            dose_labels = np.zeros((self.patient_scenario.num_subgroups, self.dose_scenario.num_doses))
                
            for s in range(self.patient_scenario.num_subgroups):
                dose_labels[s, :] = self.dose_scenario.dose_labels

            model = model_type(self.num_patients, self.patient_scenario.num_subgroups, self.dose_scenario.num_doses, pats, self.learning_rate, a0)
            run_metrics = model.run_model(self.dose_scenario, dose_labels)
            metrics_objects.append(run_metrics)
        
        exp_metrics = ExperimentMetrics(self.patient_scenario.num_subgroups, self.dose_scenario.num_doses,
                                        self.num_patients, self.reps, metrics_objects)
        p_hat_fin_mean = np.mean(exp_metrics.p_hat, axis=2)

        self.print_results(exp_metrics)
        plt.figure(figsize=(10, 8))
        sns.set_theme()

        # Subgroup plots
        # Dose toxicity for contextual model
        plt.subplot(331)
        subgroup_index = 0
        TanhModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.dose_scenario.toxicity_probs[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                           exp_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(332)
        subgroup_index = 1
        TanhModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.dose_scenario.toxicity_probs[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                          exp_metrics.p_hat[subgroup_index, :, :])
        
        plt.subplot(333)
        subgroup_index = 2
        TanhModel.plot_dose_toxicity_curve(dose_labels[subgroup_index], self.dose_scenario.toxicity_probs[subgroup_index], exp_metrics.a_hat_fin[subgroup_index, :],
                                           exp_metrics.p_hat[subgroup_index, :, :])
        

        self.make_plots(dose_labels, exp_metrics)


def main():
    reps = 100 # num of simulated trials
    num_patients = 50
    learning_rate = 0.01
    patient_scenario = TrialPopulationScenarios.lee_trial_population()
    dose_scenario = DoseFindingScenarios.lee_synthetic_example()
            
    runner = ExperimentRunner(reps, patient_scenario, dose_scenario, num_patients, learning_rate)

    a0 = 1 / np.e
    runner.run_one_param(OGTanhModel, a0)

def main2():
    reps = 100 # num of simulated trials
    num_patients = 50
    learning_rate = 0.01
    patient_scenario = TrialPopulationScenarios.homogenous_population()
    dose_scenario = DoseFindingScenarios.oquigley_model_example()
            
    runner = ExperimentRunner(reps, patient_scenario, dose_scenario, num_patients, learning_rate)

    a0 = 1 / np.e
    runner.run_one_param(OGTanhModel, a0)

if __name__ == "__main__":
    main2()