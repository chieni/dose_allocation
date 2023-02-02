import paramiko
import os
import pandas as pd
import numpy as np


def copy_files(out_folder_name, folder_name, num_trials, num_subgroups):
    servers = ['172.174.178.62', '20.55.111.55','20.55.111.101','172.174.233.187','172.174.234.5',
            '172.174.234.65','172.174.234.185', '172.174.234.240', '172.174.233.180','172.174.235.241',
            '172.174.234.17', '172.174.234.16','172.174.233.34','172.174.233.135','4.236.170.64','20.55.26.95',
            '4.236.170.149','4.236.170.103']
    folders = [folder_name for idx in range(len(servers))]

    for idx, (server_name, folder_name) in enumerate(zip(servers, folders)):
        # Create an SSH client
        client = paramiko.SSHClient()

        # Automatically add the server's host key (this is insecure, you should use ssh-keys instead)
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the server
        client.connect(hostname=server_name, username='ic390', password='B!!')

        # Create an SFTP client
        sftp = client.open_sftp()

        remote_files = sftp.listdir(f"dose_allocation/results/{folder_name}")
        matching_files = [file for file in remote_files if file.endswith(".csv") or file.endswith(".png")]

        # Download the matching files
        for file in matching_files:
            remote_file = f"dose_allocation/results/{folder_name}/{file}"
            path = f"results/{out_folder_name}/scenario{idx+1}"
            if not os.path.exists(path):
                os.makedirs(path)
            local_file = f"{path}/{file}"
            sftp.get(remote_file, local_file)
        
        # Download trial files
        for trial in range(num_trials):
            local_path = f"results/{out_folder_name}/scenario{idx+1}/trial{trial}"
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            remote_path = f"dose_allocation/results/{folder_name}/trial{trial}"
            for subgroup_idx in range(num_subgroups):
                remote_file = f"{remote_path}/{subgroup_idx}_predictions.csv"
                local_file = f"{local_path}/{subgroup_idx}_predictions.csv"
                sftp.get(remote_file, local_file)
            remote_file = f"{remote_path}/timestep_metrics.csv"
            local_file = f"{local_path}/timestep_metrics.csv"
            sftp.get(remote_file, local_file)

        sftp.close()
        client.close()

def combine_files(filepath):
    filepath = f"results/{filepath}"
    num_scenarios = 18
    frames = []
    for scenario in range(1, num_scenarios+1):
        frame = pd.read_csv(f"{filepath}/scenario{scenario}/final_metric_means.csv", index_col=0)
        frames.append(frame)
    metrics = ['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']
    for metric in metrics:
        metric_frame = pd.DataFrame(index=frames[0].index)
        for scenario in range(1, num_scenarios+1):
            metric_frame[f"scenario{scenario}"] = frames[scenario-1][metric].values
        metric_frame.to_csv(f"{filepath}/{metric}.csv")

copy_files('fourteenth_pass', 'exp18', 100, 2)
combine_files('fourteenth_pass')