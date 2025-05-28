import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

__doc__ = f"""
Recording the gradient noise conducted by sgd under AlexNet of CIFAR10 dataset need a long time to train the model
for hundred thousands of epochs of forward and backward. It is difficult to run it again fast, so we do the experiments
on the Server, download the 100,000 backward data and save them in the files.
You can run it directly to see the effect of noise. 
The training code we used see 'https://github.com/umutsimsekli/sgd_tail_index/blob/master/models.py'
Reference: 
Nguyen T H, Simsekli U, Gurbuzbalaban M, et al. 
First exit time analysis of stochastic gradient descent under heavy-tailed gradient noise[J]. 
Advances in neural information processing systems, 2019, 32.
"""

# LaTeX + Times New Roman 配置
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "axes.labelweight": 'bold',
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# figure format
def sci_notation(x, _):
    if x == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    base = x / 10**exponent
    return r"${:.1f} \times 10^{{{}}}$".format(base, exponent)

formatter = FuncFormatter(sci_notation)

def plot_first_order():
    dataframe = pd.read_csv('experiment_data/sgd_noise_moment1.csv', sep=',', encoding='utf-8')

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bins = np.linspace(0, 900, 200)

    axs[0].hist(dataframe['real_noise'], bins=bins, color='#e08214', log=True)
    axs[0].set_title("Real Gradient Noise", fontsize=14, weight='bold')
    axs[0].set_xlabel("Real Gradient Noise", fontsize=16, weight='bold')
    axs[0].set_ylabel("Count", fontsize=16, weight='bold')

    axs[1].hist(dataframe['gaussian_noise'], bins=bins, color='#8073ac', log=True)
    axs[1].set_title("Gaussian Noise", fontsize=14, weight='bold')
    axs[1].set_xlabel("Gaussian Noise", fontsize=16, weight='bold')

    axs[2].hist(dataframe['SalphaS_noise'], bins=bins, color='#e08214', log=True)
    axs[2].set_title(r"S$\alpha$S Noise", fontsize=14, weight='bold')
    axs[2].set_xlabel(r"S$\alpha$S Noise", fontsize=16, weight='bold')

    for ax in axs:
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.savefig("experiment_result_figure/sgd_noise_moment1.pdf")
    plt.show()


def plot_second_order():
    # figure setting
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bins = np.linspace(0, 9e5, 300)

    titles = [
        r"\textbf{\rmfamily Real Second Moment}",
        r"\textbf{\rmfamily Gaussian Noise}",
        r"$S\alpha S$\textbf{ Noise}",
    ]
    xlabels = [
        r"\textbf{Real Second Moment}",
        r"\textbf{Gaussian Noise}",
        r"$S\alpha S$ \textbf{Noise}",
    ]
    colors = ['#e08214', '#8073ac', '#e08214']
    dataframe = pd.read_csv("experiment_data/sgd_noise_moment2.csv", sep=',', encoding='utf-8')
    datasets = []
    for column in dataframe.columns:
        datasets.append(dataframe[column].to_numpy())

    # main loop
    for i in range(3):
        axs[i].hist(datasets[i], bins=bins, color=colors[i], log=True)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(xlabels[i])
        axs[i].xaxis.set_major_formatter(formatter)
        axs[i].yaxis.set_major_formatter(formatter)

    #set Y label
    axs[0].set_ylabel(r"\textbf{\rmfamily Count}")
    plt.tight_layout()
    plt.savefig("experiment_result_figure/sgd_noise_moment2.pdf")
    plt.show()

if __name__ == "__main__":
    plot_first_order()
    plot_second_order()

