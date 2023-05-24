import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


from data_generation import DoseFindingScenarios, DoseFindingScenario



def plot_true_subgroup_curves_paper_fig(out_filename, scenario):
    sns.set_style('white')
    _, axs = plt.subplots(1, 2, figsize=(9, 4))
    # axs[0].set_title(f"Toxicity")
    # axs[1].set_title(f"Efficacy")
    # axs[2].set_title(f"Thall Utility")
    utils = DoseFindingScenario.calculate_dose_utility_thall(scenario.toxicity_probs, scenario.efficacy_probs,
                                                                scenario.toxicity_threshold, scenario.efficacy_threshold,
                                                                scenario.p_param)
    linestyles = ['solid', 'dashdot']
    colors = ['crimson', 'mediumblue']
    alphas = [0.9, 0.6]
    for idx in range(scenario.num_subgroups):
        axs[0].plot(scenario.dose_labels, scenario.toxicity_probs[idx, :], marker='', linewidth=2.5, linestyle='solid', color=colors[idx], label=f"Toxicity {idx}", alpha=alphas[idx])
        axs[0].plot(scenario.dose_labels, scenario.efficacy_probs[idx, :], marker='', linewidth=2.5, linestyle='dashdot', color=colors[idx], label=f"Efficacy {idx}", alpha=alphas[idx])
        axs[1].plot(scenario.dose_labels, utils[idx], marker='', linewidth=2.5, color=colors[idx],linestyle='solid', label=f"Subgroup {idx}")
        best_dose_idx = np.argmax(utils[idx])
        best_dose = scenario.dose_labels[best_dose_idx]
        print(idx, best_dose)

        axs[0].plot(best_dose, scenario.toxicity_probs[idx, best_dose_idx], marker='o', color='green')
        axs[0].plot(best_dose,scenario.efficacy_probs[idx, best_dose_idx], marker='o', color='green')

        axs[1].plot(best_dose, utils[idx, best_dose_idx], marker='o', color='green', label='Optimal Dose')

    axs[0].plot(scenario.dose_labels, np.repeat(scenario.toxicity_threshold, len(scenario.dose_labels)), color="rosybrown",
                linestyle="solid", label='Threshold', alpha=0.5)
    axs[0].plot(scenario.dose_labels, np.repeat(scenario.efficacy_threshold, len(scenario.dose_labels)), color="rosybrown",
                linestyle="dashdot", label='Threshold', alpha=0.5)
    axs[1].plot([0, 40], [0, 0], marker='', linewidth=1, linestyle='solid', color='gray', alpha=0.5)
    axs[0].set_ylim([0, 1.1])
    axs[1].set_ylim([-0.5, 0.5])
    # axs[0].set_xlabel('Dose')
    # axs[0].set_ylabel('Response Probability')
    # axs[1].set_xlabel('Dose')
    # axs[1].set_ylabel('Utility')
    axs[1].legend()

    
    handles, labels = axs[0].get_legend_handles_labels()
    myHandle = [Line2D([], [], color='rebeccapurple', linestyle='solid'),
                Line2D([], [],color='crimson',  linestyle='dashdot', alpha=0.7),
                Line2D([], [], color='mediumblue', linestyle='dashdot', alpha=0.7),
                Line2D([], [], color='rosybrown', linestyle='solid'),
                Line2D([], [], color='rosybrown', linestyle='dashdot')] 
    axs[0].legend(handles=myHandle, labels=[ 'Toxicity: All','Efficacy: Subgroup 0', 'Efficacy: Subgroup 1',
                                            'Max Toxicity Threshold', 'Min Efficacy Threshold'])
    handles, labels = axs[1].get_legend_handles_labels()
    myHandle = [Line2D([], [],color='crimson',  linestyle='solid', alpha=0.7),
                Line2D([], [], color='mediumblue', linestyle='solid', alpha=0.7),
                Line2D([], [], marker="o", color='green', markersize="5", linestyle='None')] 
    axs[1].legend(handles=myHandle, labels=['Subgroup 0', 'Subgroup 1', 'Optimal Dose'])
    # plt.tight_layout()
    # plt.savefig(out_filename, dpi=500)
    axs[0].tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    axs[1].tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=500)
        
    
def plot_utility_tradeoff(out_foldername, scenario):
    p_param = scenario.p_param
    tox_thre = scenario.toxicity_threshold
    eff_thre = scenario.efficacy_threshold
    midpoint = scenario.midpoint # 0.3, 0.4
    print(midpoint)
    
    x_values = np.arange(eff_thre, 1.0, 0.01)
    y_values = tox_thre * (1 - ((x_values - 1.)/(eff_thre - 1.))**p_param)**(1./p_param)

    # smaller_midpoint = (0.4, 0.25)
    smaller_midpoint = (0.5, 0.2)
    smaller_p = DoseFindingScenario.calculate_utility_param(tox_thre, eff_thre, smaller_midpoint)
    print(smaller_p)
    smaller_y_values = tox_thre * (1 - ((x_values - 1.)/(eff_thre - 1.))**smaller_p)**(1./smaller_p)

    sns.set_style('whitegrid')
    plt.plot(x_values, y_values, color='darkgreen', linewidth=2.5, label=f"p = {round(p_param, 1)}")
    plt.plot(x_values, smaller_y_values, color='darkgreen', linewidth=2.5, linestyle='dashed', label=f"p = {round(smaller_p, 1)}")
    plt.plot(midpoint[0], midpoint[1], marker='o', markersize='7', color='black', label=f"Midpoint: {midpoint}")
    plt.plot(smaller_midpoint[0], smaller_midpoint[1], marker='X', markersize='7', color='black', label=f"Midpoint: {smaller_midpoint}")
    plt.xlabel('Probability of Efficacy')
    plt.ylabel('Probability of Toxicity')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f"{out_foldername}/util_tradeoff_curve.png", dpi=500)
    plt.close()

    colors = ['royalblue', 'darkorange']
    best_doses = []
    best_doses_smaller = []
    for idx in range(scenario.num_subgroups):
        utils = DoseFindingScenario.calculate_dose_utility_thall(scenario.toxicity_probs[idx, :], scenario.efficacy_probs[idx, :],
                                                                 tox_thre, eff_thre,
                                                                 scenario.p_param)
        utils_smaller = DoseFindingScenario.calculate_dose_utility_thall(scenario.toxicity_probs[idx, :], scenario.efficacy_probs[idx, :],
                                                                        tox_thre, eff_thre,
                                                                        smaller_p)
        plt.plot(scenario.dose_labels, utils, color=colors[idx], linewidth=2.5, label=f"Subgroup {idx}, p = {round(p_param, 1)}")
        plt.plot(scenario.dose_labels, utils_smaller, color=colors[idx], linewidth=2.5, linestyle='dashed', label=f"Subgroup {idx}, p = {round(smaller_p, 1)}")
        best_dose = np.argmax(utils)
        best_doses.append(best_dose)
        best_dose_smaller = np.argmax(utils_smaller)
        best_doses_smaller.append(best_dose_smaller)
        plt.plot(scenario.dose_labels[best_dose], utils[best_dose], marker='o', color=colors[idx], markersize=10)
        plt.plot(scenario.dose_labels[best_dose_smaller], utils_smaller[best_dose_smaller], marker='X', color=colors[idx], markersize=10)

    plt.xlabel("Dose Value")
    plt.ylabel('Utility')
    plt.legend()
    plt.savefig(f"{out_foldername}/subgroup_utilities_curve.png", dpi=500)
    plt.close()

    print(best_doses)
    print(best_doses_smaller)
    for idx in range(scenario.num_subgroups):
        plt.plot(scenario.dose_labels, scenario.toxicity_probs[idx, :], color=colors[idx], linewidth=2.5, label=f"Subgroup {idx}")
        opt_idx = best_doses[idx]
        plt.plot(scenario.dose_labels[opt_idx], scenario.toxicity_probs[idx, :][opt_idx], color=colors[idx], marker='o', markersize=10)

        opt_idx = best_doses_smaller[idx]
        plt.plot(scenario.dose_labels[opt_idx], scenario.toxicity_probs[idx, :][opt_idx], color=colors[idx], marker='X', markersize=10)
    
    plt.plot(scenario.dose_labels, np.repeat(tox_thre, len(scenario.dose_labels)), linewidth=2.5, color="dimgray",
             linestyle="dotted", label='Toxicity Threshold', alpha=0.8)

    plt.xlabel("Dose Value")
    plt.ylabel('Probability of Toxicity')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f"{out_foldername}/subgroup_toxicities_curve.png", dpi=500)
    plt.close()

    for idx in range(scenario.num_subgroups):
        plt.plot(scenario.dose_labels, scenario.efficacy_probs[idx, :], color=colors[idx], linewidth=2.5, label=f"Subgroup {idx}")
        opt_idx = best_doses[idx]
        plt.plot(scenario.dose_labels[opt_idx], scenario.efficacy_probs[idx, :][opt_idx], color=colors[idx], marker='o', markersize=10)
        opt_idx = best_doses_smaller[idx]
        plt.plot(scenario.dose_labels[opt_idx], scenario.efficacy_probs[idx, :][opt_idx], color=colors[idx], marker='X', markersize=10)
    plt.plot(scenario.dose_labels, np.repeat(eff_thre, len(scenario.dose_labels)), linewidth=2.5, color="dimgray",
                        linestyle="dotted", label='Efficacy Threshold', alpha=0.8)
    plt.xlabel("Dose Value")
    plt.ylabel('Probability of Efficacy')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f"{out_foldername}/subgroup_efficacies_curve.png", dpi=500)
    plt.close()



def plot_synthetic_subgroup_curves(out_foldername, scenario, scenario_idx):
    sns.set_style('white')
    utils = DoseFindingScenario.calculate_dose_utility_thall(scenario.toxicity_probs, scenario.efficacy_probs,
                                                             scenario.toxicity_threshold, scenario.efficacy_threshold,
                                                             scenario.p_param)
    # marker_val = 'o'
    marker_val =''
    colors = ['royalblue', 'darkorange']
    opt_colors = ['blue', 'chocolate']
    linestyles = ['solid', 'dashdot']
    markers = ['X', 'P']
    linewidth = 3
    markersize = 14
    line_markersize = 8
    for idx in range(scenario.num_subgroups):
        plt.plot(scenario.dose_labels, scenario.toxicity_probs[idx, :], marker=marker_val, linewidth=linewidth, linestyle=linestyles[idx], markersize=line_markersize,
                 color=colors[idx], label=f"Subgroup {idx}")
        best_dose_idx = scenario.optimal_doses[idx]
        if best_dose_idx < len(scenario.dose_labels):
            plt.plot(scenario.dose_labels[best_dose_idx], scenario.toxicity_probs[idx, best_dose_idx], linewidth=linewidth, marker=markers[idx], color=opt_colors[idx], markersize=markersize, label=f"Subgroup {idx}: optimal dose")
    plt.plot(scenario.dose_labels, np.repeat(scenario.toxicity_threshold, len(scenario.dose_labels)), linewidth=linewidth, color="dimgray",
                        linestyle="dotted", label='Toxicity Threshold')
    
    plt.ylim(0, 1.1)
    plt.xlabel('Dose')
    plt.ylabel('Probability of Toxicity')
    plt.tight_layout()
    plt.savefig(f"{out_foldername}/scenario{scenario_idx}_toxicity.png", dpi=500)
    plt.close()

    for idx in range(scenario.num_subgroups):
        plt.plot(scenario.dose_labels, scenario.efficacy_probs[idx, :], marker=marker_val, linewidth=linewidth, linestyle=linestyles[idx], markersize=line_markersize,
                 color=colors[idx],label=f"Subgroup {idx}")
        best_dose_idx = scenario.optimal_doses[idx]
        if best_dose_idx < len(scenario.dose_labels):
            plt.plot(scenario.dose_labels[best_dose_idx],scenario.efficacy_probs[idx, best_dose_idx], linewidth=linewidth, marker=markers[idx], color=opt_colors[idx], markersize=markersize, label=f"Subgroup {idx}: optimal dose")
    plt.plot(scenario.dose_labels, np.repeat(scenario.efficacy_threshold, len(scenario.dose_labels)), linewidth=linewidth, color="dimgray",
                        linestyle="dotted",label='Efficacy Threshold')

    plt.ylim(0, 1.1)
    plt.xlabel('Dose')
    plt.ylabel('Probability of Efficacy')
    plt.tight_layout()
    plt.savefig(f"{out_foldername}/scenario{scenario_idx}_efficacy.png", dpi=500)
    plt.close()

    for idx in range(scenario.num_subgroups):
        plt.plot(scenario.dose_labels, utils[idx], marker=marker_val, linewidth=linewidth, linestyle=linestyles[idx], color=colors[idx], markersize=line_markersize,
                 label=f"Subgroup {idx}")
        best_dose_idx = scenario.optimal_doses[idx]
        if best_dose_idx < len(scenario.dose_labels):
            plt.plot(scenario.dose_labels[best_dose_idx], utils[idx, best_dose_idx], linewidth=linewidth, marker=markers[idx], color=opt_colors[idx], markersize=markersize, label=f"Subgroup {idx}: optimal dose")
    plt.xlabel('Dose')
    plt.ylabel('Utility')
    plt.tight_layout()

    myHandle = [Line2D([], [], color='royalblue', linestyle='solid'),
                Line2D([], [],color='darkorange',  linestyle='dashdot'),
                Line2D([], [], color='blue', marker='X', linestyle='None'),
                Line2D([], [], color='chocolate', marker='P', linestyle='None'),
                Line2D([], [], color='dimgray', linestyle='dotted')] 
    plt.legend(handles=myHandle, labels=['Subgroup 0', 'Subgroup 1', 'Subgroup 0: Optimal dose',
                                        'Subgroup 1: Optimal dose', 'Threshold'])
    
    plt.savefig(f"{out_foldername}/scenario{scenario_idx}_utility.png", dpi=500)
    plt.close()


def opt_scenarios_plot(out_foldername, ei_filename, ucb_filename, util_filename, metric):
    ei_frame = pd.read_csv(ei_filename, index_col=0)
    ei_melt = pd.melt(ei_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    ei_melt['method'] = 'EI'

    ucb_frame = pd.read_csv(ucb_filename, index_col=0)
    ucb_melt = pd.melt(ucb_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    ucb_melt['method'] = 'UCB'

    util_frame = pd.read_csv(util_filename, index_col=0)
    util_melt = pd.melt(util_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    util_melt['method'] = 'Utility'

    frame = pd.concat([ei_melt, ucb_melt, util_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set_style("whitegrid")
    plt.figure(figsize=(4.5, 8))
    frame = frame[frame['index'] != 'overall']
    frame['index'] = frame['index'].apply(lambda val: int(float(val)))

    fig = sns.pointplot(data=frame, x=metric, y='scenario', hue='method', markers=['*','P', '.'],
                        capsize=0.4, errwidth=1.5, scale=0.9, join=False)

    #fig.set(xlabel=None, ylabel=None, xlim=(-0.1, 1.1), ylim=(-0.5, 17.5))
    fig.set(xlabel=None, ylabel=None, xlim=(-0.75, 0.75), ylim=(-0.5, 17.5))
    # plt.legend()
    # plt.legend([],[], frameon=False)
    plt.tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(f"{out_foldername}/{metric}_opt_comparison.png", bbox_inches="tight", pad_inches=0, dpi=500)

def components_scenarios_plot(out_foldername, og_filename, mtd_filename, no_exp_filename,
                              unconstrained_filename, sep_filename, one_filename, metric):
    og_frame = pd.read_csv(og_filename, index_col=0)
    og_melt = pd.melt(og_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    og_melt['method'] = 'SAFE-T'

    mtd_frame = pd.read_csv(mtd_filename, index_col=0)
    mtd_melt = pd.melt(mtd_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    mtd_melt['method'] = 'MTD'

    no_exp_frame = pd.read_csv(no_exp_filename, index_col=0)
    no_exp_melt = pd.melt(no_exp_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    no_exp_melt['method'] = 'No-Exp'

    unconstrained_frame = pd.read_csv(unconstrained_filename, index_col=0)
    unconstrained_melt = pd.melt(unconstrained_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    unconstrained_melt['method'] = 'Unconstrained'

    sep_frame = pd.read_csv(sep_filename, index_col=0)
    sep_melt = pd.melt(sep_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    sep_melt['method'] = 'Sep'

    one_frame = pd.read_csv(one_filename, index_col=0)
    one_melt = pd.melt(one_frame.reset_index(), id_vars='index', var_name='scenario', value_name=metric)
    one_melt['method'] = 'One'

    #frame = pd.concat([og_melt, sep_melt, one_melt])
    #frame = pd.concat([og_melt, no_exp_melt, unconstrained_melt])
    frame = pd.concat([og_melt, mtd_melt])
    frame['scenario'] = frame['scenario'].apply(lambda val: val[8:])

    sns.set_style("whitegrid")
    plt.figure(figsize=(4.5, 8))
    frame = frame[frame['index'] != 'overall']
    frame['index'] = frame['index'].apply(lambda val: int(float(val)))

    # palette = sns.color_palette()
    # print(palette)
    # newp = [palette[0], palette[2], palette[3]]
    # palette = sns.color_palette("colorblind")
    palette = sns.color_palette()
    fig = sns.pointplot(data=frame, x=metric, y='scenario', hue='method', 
                        capsize=0.4, errwidth=1.5, scale=0.9, join=False)

    fig.set(xlabel=None, ylabel=None, xlim=(-0.1, 1.1), ylim=(-0.5, 17.5))
    if metric == "utility":
        fig.set(xlabel=None, ylabel=None, xlim=(-0.75, 0.75), ylim=(-0.5, 17.5))
    # plt.legend()
    # plt.legend([],[], frameon=False)
    plt.tick_params(
        axis='both',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        left=False,
        labelleft=False
    )
    plt.savefig(f"{out_foldername}/{metric}_components_mtd.png", bbox_inches="tight", pad_inches=0, dpi=500)
    plt.close()

def ratios_plot(out_foldername, og_filename, sep_filename, one_filename, metric):
    test_ratios = np.arange(0.15, 0.9, 0.05)
    test_ratios = np.array([round(ratio, 2) for ratio in test_ratios]).astype(str)
    sns.set_style("white")

    og_frame = pd.read_csv(og_filename, index_col=0)
    og_frame.columns = og_frame.columns
    og_frame = og_frame[test_ratios]
    og_frame.columns = test_ratios
    
    og_melt = pd.melt(og_frame.reset_index(), id_vars='index', var_name='subgroup_ratio', value_name=metric)
    og_melt.rename({'index': 'subgroup', 'scenario': 'subgroup_ratio'})
    og_melt['method'] = 'SAFE-T'

    sep_frame = pd.read_csv(sep_filename, index_col=0)
    sep_frame.columns = sep_frame.columns
    sep_frame = sep_frame[test_ratios]
    sep_frame.columns = test_ratios
    
    sep_melt = pd.melt(sep_frame.reset_index(), id_vars='index', var_name='subgroup_ratio', value_name=metric)
    sep_melt.rename({'index': 'subgroup', 'scenario': 'subgroup_ratio'})
    sep_melt['method'] = 'Sep'

    one_frame = pd.read_csv(one_filename, index_col=0)
    one_frame.columns = one_frame.columns
    one_frame = one_frame[test_ratios]
    one_frame.columns = test_ratios
    
    one_melt = pd.melt(one_frame.reset_index(), id_vars='index', var_name='subgroup_ratio', value_name=metric)
    one_melt.rename({'index': 'subgroup', 'scenario': 'subgroup_ratio'})
    one_melt['method'] = 'One'

    # frame = pd.concat([og_melt, sep_melt, one_melt])
    frame = pd.concat([og_melt, sep_melt])
    frame['index'] = frame['index'].apply(lambda val: str(int(float(val))) if val != 'overall' else 'overall')
    frame = frame.reset_index()
    frame = frame[frame['index'] != 'overall']

    sns.lineplot(data=frame, x='subgroup_ratio', y=metric, style='index', hue='method', linewidth=2)
    plt.legend([],[], frameon=False)
    plt.savefig(f"{out_foldername}/ratios_component_model_{metric}.png", dpi=500)
    plt.close()



scenarios = {
    1: DoseFindingScenarios.paper_example_1(),
    2: DoseFindingScenarios.paper_example_2(),
    3: DoseFindingScenarios.paper_example_3(),
    4: DoseFindingScenarios.paper_example_4(),
    5: DoseFindingScenarios.paper_example_5(),
    6: DoseFindingScenarios.paper_example_6(),
    7: DoseFindingScenarios.paper_example_7(),
    8: DoseFindingScenarios.paper_example_8(),
    9: DoseFindingScenarios.paper_example_9(),
    10: DoseFindingScenarios.paper_example_10(),
    11: DoseFindingScenarios.paper_example_11(),
    12: DoseFindingScenarios.paper_example_12(),
    13: DoseFindingScenarios.paper_example_13(),
    14: DoseFindingScenarios.paper_example_14(),
    15: DoseFindingScenarios.paper_example_15(),
    16: DoseFindingScenarios.paper_example_16(),
    17: DoseFindingScenarios.paper_example_17(),
    18: DoseFindingScenarios.paper_example_18(),
    19: DoseFindingScenarios.paper_example_19()
}


continuous_scenarios = {
        1: DoseFindingScenarios.continuous_subgroups_example_1(),
        2: DoseFindingScenarios.continuous_subgroups_example_2(),
        3: DoseFindingScenarios.continuous_subgroups_example_3(),
        4: DoseFindingScenarios.continuous_subgroups_example_4()
}
#scenario = DoseFindingScenarios.continuous_subgroups_example_2()
# plot_utility_tradeoff("results/appendix_figs", scenario)

# for idx, scenario in scenarios.items():
#     plot_synthetic_subgroup_curves("results/appendix_figs", scenario, idx)

plot_synthetic_subgroup_curves("results/appendix_figs", scenarios[8], 8)

# for idx, scenario in continuous_scenarios.items():
#     plot_synthetic_subgroup_curves("results/appendix_figs/continuous", scenario, idx)

# scenario.plot_true_subgroup_curves_paper_fig("results/comparison_plots/fig1_scen1.png")

# ei_filename = "results/gp_scenarios4/final_dose_error.csv"
# ucb_filename = "results/gp_scenarios_ucb/final_dose_error.csv"
# util_filename = "results/gp_scenarios_util/final_dose_error.csv"
# opt_scenarios_plot("results/appendix_figs/optimization", ei_filename, ucb_filename, util_filename, 'dose_error')

# ei_filename = "results/gp_scenarios4/safety_violations.csv"
# ucb_filename = "results/gp_scenarios_ucb/safety_violations.csv"
# util_filename = "results/gp_scenarios_util/safety_violations.csv"
# opt_scenarios_plot("results/appendix_figs/optimization", ei_filename, ucb_filename, util_filename, 'safety_violations')

# ei_filename = "results/gp_scenarios4/utility.csv"
# ucb_filename = "results/gp_scenarios_ucb/utility.csv"
# util_filename = "results/gp_scenarios_util/utility.csv"
# opt_scenarios_plot("results/appendix_figs/optimization", ei_filename, ucb_filename, util_filename, 'utility')


# metrics = ["final_dose_error", "safety_violations", "utility"]
# for metric in metrics:
#     og_filename = f"results/gp_scenarios4/{metric}.csv"
#     mtd_filename = f"results/gp_scenarios_mtd/{metric}.csv" # TODO: replace with real one
#     no_exp_filename = f"results/gp_scenarios_no_expansion/{metric}.csv"
#     unconstrained_filename = f"results/gp_scenarios_unconstrained/{metric}.csv"
#     sep_filename = f"results/gp_scenarios_separate/{metric}.csv"
#     one_filename = f"results/gp_scenarios_one_model/{metric}.csv"
#     components_scenarios_plot("results/appendix_figs/components", og_filename, mtd_filename, no_exp_filename,
#                                 unconstrained_filename, sep_filename, one_filename, metric)



# for metric in metrics:
#     og_filename = f"results/gp_ratios_small/{metric}.csv"
#     sep_filename = f"results/gp_ratios_small_separate/{metric}.csv"
#     one_filename = f"results/gp_ratios_small_separate/{metric}.csv"
#     ratios_plot("results/appendix_figs/components", og_filename, sep_filename, one_filename, metric)