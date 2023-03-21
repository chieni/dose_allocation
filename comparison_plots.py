import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np


def dose_error_plot(three_filename, c3t_filename, gp_filename, out_filename):
    three_frame = pd.read_csv(three_filename, index_col=0)
    three_melt = pd.melt(three_frame.reset_index(), id_vars='index', var_name='scenario', value_name='dose_error')
    three_melt['method'] = '3+3'

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='dose_error')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='dose_error')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt, three_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    # sns.set()
    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 8))
    frame = frame[frame['index'] != 'overall']
    frame['index'] = frame['index'].apply(lambda val: int(float(val)))

    fig = sns.pointplot(data=frame, x='dose_error', y='scenario', hue='method', capsize=0.4, errwidth=2.0, scale=0.9, join=False)
    # fig = sns.scatterplot(data=frame, x='scenario', y='dose_error', hue='method', style='index')
    # fig.set(xlabel=None, ylabel=None, xlim=(-0.5, 17.5), ylim=(-0.1, 1.0))
    fig.set(xlabel=None, ylabel=None, xlim=(-0.1, 1.1), ylim=(-0.5, 17.5))
    plt.legend([],[], frameon=False)
    plt.tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=500)

def safety_plot(three_filename, c3t_filename, gp_filename, out_filename):
    three_frame = pd.read_csv(three_filename, index_col=0)
    three_melt = pd.melt(three_frame.reset_index(), id_vars='index', var_name='scenario', value_name='safety_violations')
    three_melt['method'] = '3+3'

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='safety_violations')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='safety_violations')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt, three_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    # sns.set()
    # plt.figure(figsize=(8, 4))
    # fig = sns.pointplot(data=frame, x='scenario', y='safety_violations', hue='method', join=False)
    # plt.show()

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 8))
    frame = frame[frame['index'] != 'overall']
    fig = sns.pointplot(data=frame, x='safety_violations', y='scenario', hue='method', capsize=0.4, errwidth=2.0, scale=0.9, join=False)
    fig.set(xlabel=None, ylabel=None, xlim=(-0.1, 1.1), ylim=(-0.5, 17.5))

    # fig = sns.pointplot(data=frame, x='scenario', y='safety_violations', hue='method', join=False)
    # fig.set(xlabel=None, ylabel=None, xlim=(-0.5, 17.5), ylim=(-0.1, 1.0))
    plt.legend([],[], frameon=False)
    plt.tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=300)

def tox_plot(c3t_folder_name, gp_filename, folder_name):
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
    plt.savefig(f"{folder_name}/tox_comparison_plot.png", dpi=300)


def eff_plot(c3t_folder_name, gp_filename, folder_name):
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
    plt.savefig(f"{folder_name}/eff_comparison_plot.png", dpi=300)

def utility_plot(three_filename, c3t_filename, gp_filename, out_filename):
    three_frame = pd.read_csv(three_filename, index_col=0)
    three_melt = pd.melt(three_frame.reset_index(), id_vars='index', var_name='scenario', value_name='utility')
    three_melt['method'] = '3+3'

    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='scenario', value_name='utility')
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='scenario', value_name='utility')
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt, three_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 8))
    frame = frame[frame['index'] != 'overall']
    # fig = sns.pointplot(data=frame, x='scenario', y='utility', hue='method', join=False)
    # fig.set(xlabel=None, ylabel=None, xlim=(-0.5, 17.5), ylim=(-0.1, 1.0))
    fig = sns.pointplot(data=frame, x='utility', y='scenario', hue='method', capsize=0.4, errwidth=2.0, scale=0.9, join=False)
    fig.set(xlabel=None, ylabel=None, xlim=(-0.5, 0.5), ylim=(-0.5, 17.5))

    plt.legend([],[], frameon=False)
    plt.tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=300)

def tox_eff_plot(c3t_folder_name, gp_tox_filename, gp_eff_filename, folder_name):
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
    plt.savefig(f"{folder_name}/tox_eff_comparison_plot.png", dpi=300)


def tox_eff_diff_plot(c3t_folder_name, gp_tox_filename, gp_eff_filename, folder_name):
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

    #frame = pd.concat([c3t_tox_melt, gp_tox_melt])
    gp_tox_melt['tox_diff'] = c3t_tox_melt['tox'] - gp_tox_melt['tox']
    gp_tox_melt['eff_diff'] = gp_tox_melt['eff'] - c3t_tox_melt['eff']

    sns.set()
    plt.figure(figsize=(8, 8))
    gp_tox_melt = gp_tox_melt[gp_tox_melt['index'] != 'overall']
    gp_tox_melt['index'] = gp_tox_melt['index'].apply(lambda val: int(float(val)))
    sns.scatterplot(data=gp_tox_melt, x='eff_diff', y='tox_diff', style='index')
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/tox_eff_diff_comparison_plot.png", dpi=300)


def util_with_thall(c3t_folder_name, gp_filename, folder_name):
    pass



def sample_size_plot(c3t_filename, gp_filename):
    sns.set()
    c3t_frame = pd.read_csv(c3t_filename, index_col=0)
    c3t_frame.columns = np.arange(51, 205, 9)
    c3t_melt = pd.melt(c3t_frame.reset_index(), id_vars='index', var_name='sample_size', value_name='dose_error')
    c3t_melt.rename({'index': 'subgroup', 'scenario': 'sample_size'})
    c3t_melt['method'] = 'c3t'

    gp_frame = pd.read_csv(gp_filename, index_col=0)
    gp_frame.columns = np.arange(51, 205, 9)
    gp_melt = pd.melt(gp_frame.reset_index(), id_vars='index', var_name='sample_size', value_name='dose_error')
    gp_melt.rename({'index': 'subgroup', 'scenario': 'sample_size'})
    gp_melt['method'] = 'gp'

    frame = pd.concat([c3t_melt, gp_melt])
    frame['index'] = frame['index'].apply(lambda val: str(int(float(val))) if val != 'overall' else 'overall')
    sns.lineplot(data=frame, x='sample_size', y='dose_error', style='index', hue='method')
    plt.show()



folder_name = "nineteenth_pass"
dose_filename = f"results/{folder_name}/final_dose_error.csv"
thall_filename = f"results/{folder_name}/thall_final_dose_error_retrain.csv"
safety_filename = f"results/{folder_name}/safety_violations.csv"
tox_filename = f"results/{folder_name}/tox_outcome.csv"
eff_filename = f"results/{folder_name}/eff_outcome.csv"
utility_filename = f"results/{folder_name}/thall_utilities.csv"

c3t_folder_name = "c3t_more10"

# safety_plot(f"results/{c3t_folder_name}/safety violations.csv", safety_filename, f"results/{folder_name}")
# dose_error_plot(f"results/{c3t_folder_name}/final dose error.csv", dose_filename, f"results/{folder_name}")
# tox_plot(c3t_folder_name, tox_filename, f"results/{folder_name}")
# eff_plot(c3t_folder_name, eff_filename, f"results/{folder_name}")
# utility_plot(f"results/{c3t_folder_name}/thall_utility.csv", utility_filename, f"results/{folder_name}")
# tox_eff_plot(c3t_folder_name, tox_filename, eff_filename, f"results/{folder_name}")
# tox_eff_diff_plot(c3t_folder_name, tox_filename, eff_filename, f"results/{folder_name}")

#sample_size_plot("results/c3t_num_sample2/final dose error.csv", "results/num_samples_exp2/final_dose_error.csv")
#sample_size_plot("results/c3t_num_sample2/utility.csv", "results/num_samples_exp2/utility.csv")


# dose_error_plot(f"results/threeplusexp/final_ose_error.csv", f"results/{c3t_folder_name}/final dose error.csv",
#                 dose_filename, f"results/{folder_name}/all_dose_error_comparison_plot.png")
# safety_plot(f"results/threeplusexp/safety.csv", f"results/{c3t_folder_name}/safety violations.csv",
#             safety_filename, f"results/{folder_name}/all_safety_comparison_plot.png")
# utility_plot(f"results/threeplusexp/utility.csv", f"results/{c3t_folder_name}/thall_utility.csv",
#              utility_filename, f"results/{folder_name}/all_utility_comparison_plot.png")