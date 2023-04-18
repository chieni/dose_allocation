import numpy as np
import paramiko


def launch_experiment(hostname, exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials):
    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Add the remote server's SSH key to the local SSH client
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname=hostname, username='ic390', password='B!scuit3310!')

    # Run a command 
    exp_command = get_command(exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials)
    print(exp_command)
    stdin, stdout, stderr = ssh.exec_command(f"/bin/bash -lc 'conda activate azureml_py38 \n cd dose_allocation \n git stash \n git pull \n nohup {exp_command} >/dev/null 2>&1' ")
    count = 0
    for line in iter(stdout.readline, ""):
        print(line, end="")
        count += 1
        if count > 1:
            break

    ssh.close()

def kill_experiment(hostname):
    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Add the remote server's SSH key to the local SSH client
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname=hostname, username='ic390', password='B!scuit3310!')

    stdin, stdout, stderr = ssh.exec_command(f"/bin/bash -lc 'pkill python' ")
    count = 0
    for line in iter(stderr.readline, ""):
        print(line, end="")
        count += 1
        if count > 1:
            break

    ssh.close()

def launch_continuous_experiment(hostname, exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials):
    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Add the remote server's SSH key to the local SSH client
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname=hostname, username='ic390', password='B!scuit3310!')

    # Run a command 
    exp_command = get_continuous_command(exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials)
    print(exp_command)
    stdin, stdout, stderr = ssh.exec_command(f"/bin/bash -lc 'conda activate azureml_py38 \n cd dose_allocation \n git stash \n git pull \n nohup {exp_command} >/dev/null 2>&1' ")
    count = 0
    for line in iter(stdout.readline, ""):
        print(line, end="")
        count += 1
        if count > 1:
            break

    ssh.close()

def launch_crm_experiment(hostname, exp_name, scenario_num, num_samples, group_ratio, num_trials):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username='ic390', password='B!scuit3310!')
    crm_command = get_crm_command(exp_name, scenario_num, num_samples, group_ratio, num_trials)
    print(crm_command)
    stdin, stdout, stderr = ssh.exec_command(f"/bin/bash -lc 'conda activate azureml_py38 \n cd dose_allocation \n git stash \n git pull \n nohup {crm_command} >/dev/null 2>&1' ")
    count = 0
    for line in iter(stdout.readline, ""):
        print(line, end="")
        count += 1
        if count > 1:
            break
    ssh.close()


def get_command(exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials):
    command = f"python new_experiments_clean.py --filepath results/{exp_name} \
               --scenario {scenario_num} --beta_param 0.2 --num_samples {num_samples} --sampling_timesteps {sampling_timesteps} \
               --tox_lengthscale 4 --eff_lengthscale 2  --tox_mean -0.3 \
               --eff_mean -0.1 --learning_rate 0.0075 --num_latents 3 \
               --set_lmc --use_thall --use_lcb_init --group_ratio {group_ratio} --num_trials {num_trials}" 

    return command

def get_continuous_command(exp_name, scenario_num, num_samples, sampling_timesteps, group_ratio, num_trials):
    command = f"python continuous_experiments.py --filepath results/{exp_name} \
               --scenario {scenario_num} --beta_param 0.2 --num_samples {num_samples} --sampling_timesteps {sampling_timesteps} \
               --tox_lengthscale 4 --eff_lengthscale 2  --tox_mean -0.3 \
               --eff_mean -0.1 --learning_rate 0.0075 --num_latents 3 \
               --set_lmc --use_thall --use_lcb_init --group_ratio {group_ratio} --num_trials {num_trials}" 

    return command

def get_crm_command(exp_name, scenario_num, num_samples, group_ratio, num_trials):
    command = f"python crm.py --filepath results/{exp_name} \
                --scenario {scenario_num} --num_samples {num_samples} --group_ratio {group_ratio} --num_trials {num_trials} --add_jitter" 

    return command


servers = ['172.174.178.62', '20.55.111.55','20.55.111.101','172.174.233.187','172.174.234.5',
        '172.174.234.65','172.174.234.185', '172.174.234.240', '172.174.233.180','172.174.235.241',
        '172.174.234.17', '172.174.234.16','172.174.233.34','172.174.233.135','4.236.170.64','20.55.26.95',
        '4.236.170.149','4.236.170.103', '172.174.224.64']

# launch_experiment('172.174.178.62', 'exp24', 9, 51, 18)

# For all scenarios
# num_samples = 51
# num_trials = 100
# sampling_timesteps = 18
# patient_ratio = 0.5
# for idx, server in enumerate(servers):
#     scenario_idx =  idx + 1
#     launch_experiment(server, 'gp_scenarios2', scenario_idx, num_samples, sampling_timesteps,
#                       patient_ratio, num_trials)

# For all scenarios
# num_samples = 51
# num_trials = 100
# sampling_timesteps = 18
# patient_ratio = 0.5
# for idx, server in enumerate(servers):
#     scenario_idx =  idx + 1
#     launch_continuous_experiment(server, 'gp_continuous_scenarios', scenario_idx, num_samples, sampling_timesteps,
#                       patient_ratio, num_trials)


# test_sample_nums = np.arange(375, 546, 9)
# scenario_idx = 9 # scenario 9
# for idx, num_samples in enumerate(test_sample_nums):
#     server = servers[idx]
#     #sampling_timesteps = 18 + (idx * 3)
#     sampling_timesteps = int((18/51) * num_samples)
#     print(num_samples, sampling_timesteps)
#     launch_experiment(server, 'exp_sample_size4', scenario_idx, num_samples, sampling_timesteps)

# For all scenarios CRM
# num_samples = 51
# num_trials = 100
# for idx, server in enumerate(servers):
#     scenario_idx = idx + 1
#     launch_crm_experiment(server, 'crm_scenarios3', scenario_idx, num_samples, num_trials)

# 5, 6, 9, 13, 15, 18, 19
# python crm.py --filepath results/crm_scenarios4 --scenario 14 --num_samples 51 --num_trials 100 --add_jitter
# scenarios_to_try = [6, 19]
# num_samples = 51
# num_trials = 100
# for idx, server in enumerate(servers):
#     scenario_idx = scenarios_to_try[idx % len(scenarios_to_try)]
#     launch_crm_experiment(server, 'crm_scenarios5', scenario_idx, num_samples, num_trials)

# patient_ratios = np.arange(0.1, 1.0, 0.05)
# scenario_idx = 11
# num_samples = 201
# num_trials = 100
# sampling_timesteps = 69
# for idx, patient_ratio in enumerate(patient_ratios):
#     server = servers[idx]
#     launch_experiment(server, 'gp_ratios_exp3', scenario_idx, num_samples, sampling_timesteps, patient_ratio, num_trials)

patient_ratios = np.arange(0.1, 1.0, 0.05)
# scenario_idx = 11
# num_samples = 201
# num_trials = 100
# for idx, patient_ratio in enumerate(patient_ratios):
#     server = servers[idx]
#     launch_crm_experiment(server, 'crm_ratios2', scenario_idx, num_samples, patient_ratio, num_trials)

for idx, patient_ratio in enumerate(patient_ratios):
    server = servers[idx]
    kill_experiment(server)