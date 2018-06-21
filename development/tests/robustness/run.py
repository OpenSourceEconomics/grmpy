from grmpy.estimate.estimate import estimate
from os.path import join
from shutil import move
import os
import numpy as np
import matplotlib.pyplot as plt
from grmpy.estimate.estimate_auxiliary import calculate_mte
import pandas as pd

def plot_est_mte(rslt, data_frame):
    name = 'comparison'
    quantiles = np.arange(0.01, 1., 0.005).tolist()
    mte = calculate_mte(rslt, data_frame, quantiles)

    ax = plt.figure().add_subplot(111)

    ax.set_ylabel(r"$B^{MTE}$")
    ax.set_xlabel("$u_S$")
    ax.plot(quantiles, mte, label='MTE')
    ax.set_ylim([-0.5, 0.7])

    plt.legend()

    plt.tight_layout()
    plt.savefig('MTE-replication-fig.png')
    plt.show()

if __name__ == '__main__':
    directory = os.path.dirname(os.path.realpath(__file__))
    target = os.path.split(os.path.split(os.path.split(directory)[0])[0])[0] + '/docs/figures'
    filename = 'MTE-replication-fig.png'
    rslt = estimate('replication.grmpy.ini')
    data = pd.read_pickle('aer-replication-mock.pkl')
    plot_est_mte(rslt, data)
    move(join(directory, filename), join(target, filename))

