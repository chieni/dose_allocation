import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def dose_error_plot(c3t_folder_name, gp_filename):
    c3t_filename = f"results/{c3t_folder_name}/final dose error.csv"

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='dose_error')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='dose_error')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set()
    plt.figure(figsize=(12, 4))
    frame = frame[frame['index'] != 'overall']
    sns.pointplot(data=frame, x='scenario', y='dose_error', hue='method', join=False)
    plt.show()

def safety_plot(c3t_folder_name, gp_filename):
    c3t_filename = f"results/{c3t_folder_name}/safety violations.csv"

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='safety_violations')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='safety_violations')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set()
    plt.figure(figsize=(12, 4))
    frame = frame[frame['index'] != 'overall']
    sns.pointplot(data=frame, x='scenario', y='safety_violations', hue='method', join=False)
    plt.show()

def tox_plot(c3t_folder_name, gp_filename):
    c3t_filename = f"results/{c3t_folder_name}/toxicity by person.csv"

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='toxicity')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='toxicity')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set()
    plt.figure(figsize=(12, 4))
    frame = frame[frame['index'] != 'overall']
    sns.pointplot(data=frame, x='scenario', y='toxicity', hue='method', join=False)
    plt.show()


def eff_plot(c3t_folder_name, gp_filename):
    c3t_filename = f"results/{c3t_folder_name}/efficacy by person.csv"

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='efficacy')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='efficacy')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set()
    plt.figure(figsize=(12, 4))
    frame = frame[frame['index'] != 'overall']
    sns.pointplot(data=frame, x='scenario', y='efficacy', hue='method', join=False)
    plt.show()

def utility_plot(c3t_folder_name, gp_filename):
    c3t_filename = f"results/{c3t_folder_name}/thall_utility.csv"

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='utility')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='utility')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set()
    plt.figure(figsize=(12, 4))
    frame = frame[frame['index'] != 'overall']
    sns.pointplot(data=frame, x='scenario', y='utility', hue='method', join=False)
    plt.show()

def tox_eff_plot(c3t_folder_name, gp_tox_filename, gp_eff_filename):
    c3t_tox_filename = f"results/{c3t_folder_name}/toxicity by person.csv"
    c3t_eff_filename = f"results/{c3t_folder_name}/efficacy by person.csv"

    c3t_tox_frame = pd.read_csv(c3t_tox_filename, index_col=0)
    c3t_eff_frame = pd.read_csv(c3t_eff_filename, index_col=0)
    gp_tox_frame = pd.read_csv(gp_tox_filename, index_col=0)
    gp_eff_frame = pd.read_csv(gp_eff_filename, index_col=0)

    c3t_tox_melt = pd.melt(c3t_tox_frame.reset_index(), id_vars='index', var_name='scenario', value_name='tox')
    c3t_tox_melt['method'] = 'c3t'

    c3t_eff_melt = pd.melt(c3t_eff_frame.reset_index(), id_vars='index', var_name='scenario', value_name='eff')
    c3t_eff_melt['method'] = 'c3t'
    c3t_tox_melt['eff'] = c3t_eff_melt['eff']

    gp_tox_melt = pd.melt(gp_tox_frame.reset_index(), id_vars='index', var_name='scenario', value_name='tox')
    gp_tox_melt['method'] = 'gp'

    gp_eff_melt = pd.melt(gp_eff_frame.reset_index(), id_vars='index', var_name='scenario', value_name='eff')
    gp_eff_melt['method'] = 'gp'
    gp_tox_melt['eff'] = gp_eff_melt['eff']

    frame = pd.concat([c3t_tox_melt, gp_tox_melt])

    sns.set()
    frame = frame[frame['index'] != 'overall']
    frame['index'] = frame['index'].apply(lambda val: int(float(val)))
    sns.scatterplot(data=frame, x='eff', y='tox', hue='method', style='index')
    plt.show()

folder_name = "eleventh_pass"
dose_filename = f"results/{folder_name}/final_dose_error.csv"
thall_filename = f"results/{folder_name}/thall_final_dose_error.csv"
safety_filename = f"results/{folder_name}/safety_violations.csv"
tox_filename = f"results/{folder_name}/tox_outcome.csv"
eff_filename = f"results/{folder_name}/eff_outcome.csv"
utility_filename = f"results/{folder_name}/test_thall_utility.csv"

c3t_folder_name = "c3t_more6"

#dose_error_plot(c3t_folder_name, thall_filename)
#safety_plot(c3t_folder_name, safety_filename)
#utility_plot(c3t_folder_name, utility_filename)
#tox_plot(c3t_folder_name, tox_filename)
#eff_plot(c3t_folder_name, eff_filename)
tox_eff_plot(c3t_folder_name, tox_filename, eff_filename)