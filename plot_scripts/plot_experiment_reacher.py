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
    colors=[0,2,4,6,3,5,8,10,12,14,16,18,1]
    for label in ["DPro", "DPro+","DPre", "DPre+", "DOpt", 
                  "DOpt", "NFW", "DAlpha", "DRad", "SPre", "SPre+", "SAlpha", "SRad",
    ]:
        prefix = "re-exp/logs-R-L2/" + label
        index+=1
        x, y = loadEvaluationsN(prefix, N)
        #means = [bs.bootstrap(y[:,i], stat_func=stats_functions.mean) for i in range(len(x))]
        #plt.errorbar(x, [mean.value for mean in means], [[mean.lower_bound for mean in means], [mean.upper_bound for mean in means]], label = label, capsize = 2, elinewidth = 1)
        #plt.errorbar(x, np.mean(y, axis=0), np.std(y, axis=0)/np.sqrt(N), label = label, capsize = 2, color=cmap(index))
        plt.plot(x, np.mean(y, axis=0), label = label, color=cmap(colors[index]))
        #R.append(np.sum(y, axis=1))
        labels.append(label)

    plt.xlabel('Number of timesteps')
    plt.ylabel('Rewards')
    plt.ylim([-10.,25.0])
    #plt.title(title)
    plt.legend(ncol=2, loc='lower right')
    plt.subplots_adjust(left=0.11,right=0.95,top=0.95,bottom=0.13)
    plt.savefig("../../../miura/figures/reacher.pdf")
    plt.show()
        
    
plot_results("./", "Learning Curve Reacher")
