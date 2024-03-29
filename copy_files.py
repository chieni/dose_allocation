import paramiko
import os
import pandas as pd
import numpy as np


password = ''
username = ''
def copy_files(servers, folder_name, num_trials, num_subgroups):
    folders = [folder_name for idx in range(len(servers))]

    patient_ratios = np.arange(0.1, 1.0, 0.05)
    test_sample_nums = np.arange(60, 240, 9)

    for idx, (server_name, folder_name) in enumerate(zip(servers, folders)):
        # Create an SSH client
        client = paramiko.SSHClient()

        # Automatically add the server's host key (this is insecure, you should use ssh-keys instead)
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the server
        client.connect(hostname=server_name, username=username, password=password)

        # Create an SFTP client
        sftp = client.open_sftp()

        try:
            remote_files = sftp.listdir(f"dose_allocation/results/{folder_name}")
            matching_files = [file for file in remote_files if file.endswith(".csv") or file.endswith(".png")]
        except:
            continue

        # Download the matching files
        for file in matching_files:
            remote_file = f"dose_allocation/results/{folder_name}/{file}"
            path = f"results/{folder_name}/scenario{idx+1}"
            #path = f"results/{folder_name}/ratio{round(patient_ratios[idx], 2)}"
            ##path = f"results/{folder_name}/num_samples{test_sample_nums[idx]}"
            if not os.path.exists(path):
                os.makedirs(path)
            local_file = f"{path}/{file}"
            try:
                sftp.get(remote_file, local_file)
            except:
                continue
        
       # Download trial files
        # for trial in range(num_trials):
        #     local_path = f"results/{folder_name}/scenario{idx+1}/trial{trial}"
        #     #local_path = f"results/{folder_name}/ratio{round(patient_ratios[idx], 2)}/trial{trial}"
        #     if not os.path.exists(local_path):
        #         os.makedirs(local_path)
        #     remote_path = f"dose_allocation/results/{folder_name}/trial{trial}"
        #     for subgroup_idx in range(num_subgroups):
        #         remote_file = f"{remote_path}/{subgroup_idx}_predictions.csv"
        #         local_file = f"{local_path}/{subgroup_idx}_predictions.csv"
        #         try:
        #             sftp.get(remote_file, local_file)
        #         except:
        #             continue
        #     remote_file = f"{remote_path}/timestep_metrics.csv"
        #     local_file = f"{local_path}/timestep_metrics.csv"
        #     try:
        #         sftp.get(remote_file, local_file)
        #     except:
        #         continue

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

def combine_files_continuous(filepath):
    filepath = f"results/{filepath}"
    num_scenarios = 4
    frames = []
    for scenario in range(1, num_scenarios+1):
        frame = pd.read_csv(f"{filepath}/scenario{scenario}/final_metric_means.csv", index_col=0)
        frames.append(frame)
    metrics = ['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error',
               'final_dose_error', 'final_dose_diff', 'final_dose_diff_abs']
    for metric in metrics:
        metric_frame = pd.DataFrame(index=frames[0].index)
        for scenario in range(1, num_scenarios+1):
            metric_frame[f"scenario{scenario}"] = frames[scenario-1][metric].values
        metric_frame.to_csv(f"{filepath}/{metric}.csv")

def combine_files_sample_sizes(filepath):
    filepath = f"results/{filepath}"
    test_sample_nums = np.arange(51, 538, 9)
    frames = []
    for num_samples in test_sample_nums:
        frame = pd.read_csv(f"{filepath}/num_samples{num_samples}/final_metric_means.csv", index_col=0)
        frames.append(frame)
    metrics = ['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']
    for metric in metrics:
        metric_frame = pd.DataFrame(index=frames[0].index)
        for idx, num_samples in enumerate(test_sample_nums):
            metric_frame[num_samples] = frames[idx][metric].values
        metric_frame.to_csv(f"{filepath}/{metric}.csv")

def combine_files_ratios(filepath):
    filepath = f"results/{filepath}"
    patient_ratios = np.arange(0.1, 0.95, 0.05)
    frames = []
    for num_samples in patient_ratios:
        frame = pd.read_csv(f"{filepath}/ratio{round(num_samples, 2)}/final_metric_means.csv", index_col=0)
        frames.append(frame)
    metrics = ['tox_outcome', 'eff_outcome', 'utility', 'safety_violations', 'dose_error', 'final_dose_error']
    for metric in metrics:
        metric_frame = pd.DataFrame(index=frames[0].index)
        for idx, patient_ratio in enumerate(patient_ratios):
            metric_frame[round(patient_ratio, 2)] = frames[idx][metric].values
        metric_frame.to_csv(f"{filepath}/{metric}.csv")

def combine_files_crm(filepath):
    filepath = f"results/{filepath}"
    num_scenarios = 18
    patient_ratios = np.arange(0.1, 0.9, 0.05)
    frames = []
    # for scenario in range(1, num_scenarios+1):
    #     frame = pd.read_csv(f"{filepath}/scenario{scenario}/overall_metrics.csv", index_col=0)
    #     frames.append(frame)
    for num_samples in patient_ratios:
        frame = pd.read_csv(f"{filepath}/ratio{round(num_samples, 2)}/overall_metrics.csv", index_col=0)
        frames.append(frame)
    metrics = ['tox_outcome', 'eff_outcome', 'utilities', 'safety_violations', 'dose_error', 'final_dose_error']
    for metric in metrics:
        metric_frame = pd.DataFrame(index=frames[0].index)
        # for scenario in range(1, num_scenarios+1):
        #     metric_frame[f"scenario{scenario}"] = frames[scenario-1][metric].values
        for idx, patient_ratio in enumerate(patient_ratios):
            metric_frame[round(patient_ratio, 2)] = frames[idx][metric].values
        metric_frame.to_csv(f"{filepath}/{metric}.csv")


servers = ['172.174.178.62', '20.55.111.55','20.55.111.101','172.174.233.187','172.174.234.5',
        '172.174.234.65','172.174.234.185', '172.174.234.240', '172.174.233.180','172.174.235.241',
        '172.174.234.17', '172.174.234.16','172.174.233.34','172.174.233.135','4.236.170.64','20.55.26.95',
        '4.236.170.149','4.236.170.103', '172.174.224.64']
servers2 = ['172.174.224.64', '20.42.87.118', '172.174.180.168', '172.174.180.184', '20.119.91.21',
            '172.174.208.22', '172.174.212.95', '172.174.212.107', '172.174.212.108', '172.174.212.139',
            '172.174.212.157', '172.174.212.161', '172.174.212.136', '172.174.212.171', '172.174.212.174',
            '172.174.212.176', '172.174.212.183', '172.174.212.248']

copy_files(servers, 'gp_scenarios_mtd', 100, 2)
combine_files('gp_scenarios_mtd')
#combine_files_ratios('gp_ratios_small_separate')
#combine_files_sample_sizes('gp_sample_size')
# combine_files_crm('crm_ratios2')
# combine_files_continuous('gp_continuous_scenarios')