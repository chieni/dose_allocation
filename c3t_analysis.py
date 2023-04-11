import pandas as pd
import numpy as np

# filepath = "results/c3t_scenarios_jitter9"
# num_scenarios = 18
# frames = []
# for scenario in range(1, num_scenarios+1):
#     frame = pd.read_csv(f"{filepath}/scenario{scenario}/metrics_fram.csv", index_col=0).T
#     frames.append(frame)

# metrics = ['safety violations', 'utility', 'thall_utility', 'dose error by person',
#            'efficacy by person', 'toxicity by person', 'final dose error']
# for metric in metrics:
#     metric_frame = pd.DataFrame(index = frames[0]['subgroup'].values)
#     for scenario in range(1, num_scenarios+1):
#         metric_frame[f"scenario{scenario}"] = frames[scenario-1][metric].values

#     metric_frame.to_csv(f"{filepath}/{metric}.csv")


# filepath = "results/c3t_num_samples_jitter2"
# test_sample_nums = np.arange(51, 1000, 9)

# frames = []
# for num_samples in test_sample_nums:
#     frame = pd.read_csv(f"{filepath}/num_samples{num_samples}/metrics_fram.csv", index_col=0).T
#     frames.append(frame)

# metrics = ['safety violations', 'utility', 'thall_utility', 'dose error by person',
#            'efficacy by person', 'toxicity by person', 'final dose error']
# for metric in metrics:
#     metric_frame = pd.DataFrame(index = frames[0]['subgroup'].values)
#     for idx, num_samples in enumerate(test_sample_nums):
#         metric_frame[num_samples] = frames[idx][metric].values

#     metric_frame.to_csv(f"{filepath}/{metric}.csv")


filepath = "results/c3t_ratios1000_3"
patient_ratios = np.arange(0.1, 1.0, 0.05)
frames = []
for patient_ratio in patient_ratios:
    frame = pd.read_csv(f"{filepath}/ratio{patient_ratio}/metrics_fram.csv", index_col=0).T
    frames.append(frame)

metrics = ['safety violations', 'utility', 'thall_utility', 'dose error by person',
           'efficacy by person', 'toxicity by person', 'final dose error']
for metric in metrics:
    metric_frame = pd.DataFrame(index = frames[0]['subgroup'].values)
    for idx, patient_ratio in enumerate(patient_ratios):
        metric_frame[round(patient_ratio, 2)] = frames[idx][metric].values

    metric_frame.to_csv(f"{filepath}/{metric}.csv")