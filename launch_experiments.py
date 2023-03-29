import numpy as np
import paramiko


def launch_experiment(hostname, exp_name, scenario_num, num_samples, sampling_timesteps):
    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Add the remote server's SSH key to the local SSH client
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname=hostname, username='ic390', password='B!scuit3310!')

    # Run a command 
    exp_command = get_command(exp_name, scenario_num, num_samples, sampling_timesteps)
    print(exp_command)
    stdin, stdout, stderr = ssh.exec_command(f"/bin/bash -lc 'conda activate azureml_py38 \n cd dose_allocation \n git stash \n git pull \n nohup {exp_command} >/dev/null 2>&1' ")
    # >/dev/null 2>&1
    # count = 0
    # for line in iter(stderr.readline, ""):
    #     print(line, end="")
    #     count += 1
    #     if count > 30:
    #         break

    ssh.close()

def get_command(exp_num, scenario_num, num_samples, sampling_timesteps):
    command = f"python new_experiments.py --filepath results/{exp_num} \
               --scenario {scenario_num} --beta_param 0.2 --num_samples {num_samples} --sampling_timesteps {sampling_timesteps} \
               --tox_lengthscale 4 --eff_lengthscale 2  --tox_mean -0.3 \
               --eff_mean -0.1 --learning_rate 0.0075 --num_latents 3 \
               --set_lmc --use_thall --use_lcb_init" 

    return command

def get_crm_command(exp_num, scenario_num, num_samples, sampling_timesteps):
    command = f"python new_experiments.py --filepath results/{exp_num} \
               --scenario {scenario_num} --beta_param 0.2 --num_samples {num_samples} --sampling_timesteps {sampling_timesteps} \
               --tox_lengthscale 4 --eff_lengthscale 2  --tox_mean -0.3 \
               --eff_mean -0.1 --learning_rate 0.0075 --num_latents 3 \
               --set_lmc --use_thall --use_lcb_init" 

    return command


servers = ['172.174.178.62', '20.55.111.55','20.55.111.101','172.174.233.187','172.174.234.5',
        '172.174.234.65','172.174.234.185', '172.174.234.240', '172.174.233.180','172.174.235.241',
        '172.174.234.17', '172.174.234.16','172.174.233.34','172.174.233.135','4.236.170.64','20.55.26.95',
        '4.236.170.149','4.236.170.103', '172.174.224.64']


# launch_experiment('172.174.178.62', 'exp24', 9, 51, 18)

# For all scenarios
# for idx, server in enumerate(servers):
#     scenario_idx =  idx + 1
#     launch_experiment(server, 'exp23', scenario_idx)

# For one scenario
# num_samples = 51
# sampling_timesteps = 18 

# test_sample_nums = np.arange(51, 223, 9)

test_sample_nums = np.arange(375, 546, 9)
scenario_idx = 9 # scenario 9
for idx, num_samples in enumerate(test_sample_nums):
    server = servers[idx]
    #sampling_timesteps = 18 + (idx * 3)
    sampling_timesteps = int((18/51) * num_samples)
    print(num_samples, sampling_timesteps)
    launch_experiment(server, 'exp_sample_size4', scenario_idx, num_samples, sampling_timesteps)
