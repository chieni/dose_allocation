import pandas as pd

filepath = "results/c3t_more6"
num_scenarios = 18
frames = []
for scenario in range(1, num_scenarios+1):
    frame = pd.read_csv(f"{filepath}/scenario{scenario}/metrics_fram.csv", index_col=0).T
    frames.append(frame)

metrics = ['safety violations', 'utility', 'thall_utility', 'dose error by person',
           'efficacy by person', 'toxicity by person', 'final dose error']
for metric in metrics:
    metric_frame = pd.DataFrame(index = frames[0]['subgroup'].values)
    for scenario in range(1, num_scenarios+1):
        metric_frame[f"scenario{scenario}"] = frames[scenario-1][metric].values

    metric_frame.to_csv(f"{filepath}/{metric}.csv")

