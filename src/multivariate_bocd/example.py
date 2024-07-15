import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LogNorm

from mbocd import BOCD, MultivariateGaussianWishartModel


def plot(bocd, data):

    changepoints = bocd.changepoints
    run_probs = bocd.run_length_probabilities()

    plt.rcParams['pdf.fonttype'] = 42

    num_variables = data.shape[1]
    t = list(range(0, len(data)))

    fig, axs = plt.subplots(num_variables + 1, 1, figsize=(10, num_variables))
    for i in range(num_variables):

        plt.subplot(num_variables + 1, 1, i+1)

        y = data[:, i]
        plt.plot(t, y, lw=3, c='sienna')
        if i + 1 < num_variables:
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ylim = plt.ylim()
        plt.vlines(x=changepoints, ymin=ylim[0], ymax=ylim[1], colors='gray', label='Changepoint', lw=1)
        plt.tick_params(axis='both', labelsize=12)
        if i == 0:
            plt.legend(fontsize=18)

    # Run Length Probabilities
    plt.subplot(num_variables + 1, 1, num_variables + 1)
    ymax = np.argmax(np.argmax(run_probs != 0, axis=1)) - 20
    run_probs = run_probs[ymax:, :]
    plt.imshow(run_probs, aspect='auto', cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1))
    axs[-1].set_xlim(axs[0].get_xlim())
    yticks = list(np.arange(run_probs.shape[0] - (run_probs.shape[0] % 10), -1, -80))
    axs[-1].set_yticks(yticks)
    axs[-1].set_yticklabels(yticks[::-1])
    plt.tick_params(axis='both', labelsize=12)
    plt.ylabel(f'Run Length ($r_t$)', fontsize=18)

    plt.xlabel('Timestep', rotation=0, fontsize=18)
    plt.suptitle('Variables through Time', fontsize=20)
    fig.align_ylabels(axs)
    plt.subplots_adjust(top=0.95, bottom=0.045, left=0.09, right=0.975, hspace=0.12, wspace=0.185)
    plt.show()


if __name__ == '__main__':

    # *** Load Data ***
    data_path = './data/time_filtered_ppangles_wamv_2024-01-23-09-51-25.csv'
    data = pd.read_csv(data_path).iloc[:1300]    # We truncate for speed, but `.iloc` should be removed.
    data = data.to_numpy()

    # Preprocess data if necessary. In our case, we scale it to make learning easier.
    data = 100 * data

    # *** Get the Prior ***

    # We use the initial measurements to define our prior. However, if you
    # have other prior information you can also use it. You also don't need
    # to initialize the prior from data, if the prior parameters are known
    # a `MultivariateGaussianWishartModel` can be directly initialized.

    prior_num_timesteps = 150
    prior_data = data[:prior_num_timesteps]
    inference_data = data[prior_num_timesteps:]

    prior = MultivariateGaussianWishartModel.init_from_data(
        data=prior_data, reset_prior_on_changepoint=True
    )

    # *** Multivariate BOCD ***
    lambda_hp = 250    # Lambda hyperparameter

    bocd = BOCD(model=prior, hazard=1 / lambda_hp)

    # *** Process Data Online ***
    for obs in inference_data:
        bocd.add_observation(obs)

    # *** Plot Results
    plot(bocd=bocd, data=inference_data)
