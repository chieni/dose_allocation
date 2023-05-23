import pandas as pd
import numpy as np


folder = "results/gp_scenarios_early_stop/scenario16"
num_trials = 100
violations = []
violations0 = []
violations1 = []

for idx in range(num_trials):
    filename = f"{folder}/trial{idx}/timestep_metrics.csv"
    df = pd.read_csv(filename)
    df_sg1 = df[df['subgroup_idx'] == 0]
    df_sg2 = df[df['subgroup_idx'] == 1]
    

    violations.append(df.shape[0]/51)
    violations0.append(df_sg1.shape[0]/25.5)
    violations1.append(df_sg2.shape[0]/25.5)

print(np.mean(violations))
print(np.mean(violations0))
print(np.mean(violations1))