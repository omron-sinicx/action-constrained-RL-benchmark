from stable_baselines3.common.results_plotter import load_results, ts2xy, window_func
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import mannwhitneyu
import bootstrapped.bootstrap as bs
from bootstrapped import stats_functions
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def loadEvaluations(filename):
    npz = np.load(filename)
    return npz['timesteps'], npz['results']

def loadEvaluationsN(stem, n):
    rows = []
    timesteps = None
    for i in range(1, n+1):
        filename = f"{stem}-{i}/evaluations.npz"
        timesteps, results = loadEvaluations(filename)
        rows.append(np.mean(results, axis=1))
    L=min([len(row) for row in rows])
    for i in range(len(rows)):
        rows[i] = rows[i][:L]
    timesteps=timesteps[:L]
    matrix = np.vstack(rows)
    return (timesteps, matrix)


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    fig = plt.figure(title, figsize = (6,4))

    N=10
    R = []
    labels = []
    cmap=plt.get_cmap('tab20')
    index = -1
    for prefix, label in [
            ["../logs/main/half_cheetah/stupid_proj", "DPro"], 
            ["../logs/main/half_cheetah/stupid_proj-P", "DPro+"], 
            ["../logs/main/half_cheetah/wrapper-DDPG", "DPre"], 
            ["../logs/main/half_cheetah/wrapper-P-DDPG", "DPre+"], 
                          ["../logs/main/half_cheetah/OptSq", "DOpt"], 
                          ["../logs/main/half_cheetah/OptSqP", "DOpt+"], 
                          ["../logs/main/half_cheetah/NFWPO100", "NFW"],
                          ["../logs/main/half_cheetah/NFWPOOriginal100", "NFW*"],
            ["../logs/main/half_cheetah/alpha-DDPG", "DAlpha"],
                          ["../logs/main/half_cheetah/shrinkage-DDPG", "DRad"],
                          ["../logs/main/half_cheetah/wrapper-TrueSAC", "SPre"], 
                          ["../logs/main/half_cheetah/wraper-P-TrueSAC", "SPre+"], 
                          ["../logs/main/half_cheetah/alpha-TrueSAC", "SAlpha"],
                          ["../logs/main/half_cheetah/shrinkage-TrueSAC", "SRad"],
   ]:
        index+=1
        if label == "DOpt" or label == "DOpt+":
            continue
        x, y = loadEvaluationsN(prefix, N)
        #means = [bs.bootstrap(y[:,i], stat_func=stats_functions.mean) for i in range(len(x))]
        #plt.errorbar(x, [mean.value for mean in means], [[mean.lower_bound for mean in means], [mean.upper_bound for mean in means]], label = label, capsize = 2, elinewidth = 1)
        #plt.errorbar(x, np.mean(y, axis=0), np.std(y, axis=0)/np.sqrt(N), label = label, capsize = 2, color=cmap(index))
        plt.plot(x, np.mean(y, axis=0), label = label, color=cmap(index))
        #R.append(np.sum(y, axis=1))
        labels.append(label)

    plt.xlabel('Number of timesteps')
    plt.ylabel('Rewards')
    plt.ticklabel_format(scilimits=[-1,1])
    plt.ylim([-1000,9000])
    #plt.title(title)
    plt.legend(ncol=2, loc='lower right')
    plt.subplots_adjust(left=0.11,right=0.95,top=0.95,bottom=0.13)
    plt.savefig("../../../miura/figures/half_cheetah.pdf")
    plt.show()
        
    
plot_results("./", "Learning Curve Half_Cheetah")
