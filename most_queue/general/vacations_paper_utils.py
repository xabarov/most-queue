import json
import math

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def print_table(experiments_stat):
    """
    Prints a table comparing the results of the Takahashi-Takami method and simulation modeling for different queueing systems.
    :param experiments_stat: A list of dictionaries containing the results of the experiments.
    :type experiments_stat: list[dict]
    :return: None
    """
    for stat in experiments_stat:
        print("\nComparison of calculation results by the Takahashi-Takami method and simulation modeling.\n"
              "M/H2/{0:^2d} with H2-warm-up, H2-cooling and H2-cooling delay"
              "Utilization factor: {1:^1.2f}".format(stat["n"], stat["ro"]))
        print(f'Service time coefficient of variation {stat["coev_service"]:0.3f}')
        print(f'Warm-up time coefficient of variation {stat["coev_warm"]:0.3f}')
        print(f'Cooling time coefficient of variation {stat["coev_cold"]:0.3f}')
        print(f'Cooling delay coefficient of variation {stat["coev_cold_delay"]:0.3f}')
        print("Number of iterations of the Takahashi-Takami algorithm: {0:^4d}".format(stat["tt_num_of_iter"]))
        print(
            f"Probability of being in a warm-up state\n\tИМ: {stat['sim_warm_prob']:0.3f}\n\tЧисл: {stat['tt_warm_prob']:0.3f}")
        print(
            f"Probability of being in a cooling state\n\tИМ: {stat['sim_cold_prob']:0.3f}\n\tЧисл: {stat['tt_cold_prob']:0.3f}")
        print(
            f"Probability of being in a cooling delay state\n\tИМ: {stat['sim_cold_delay_prob']:0.3f}\n\tЧисл: {stat['tt_cold_delay_prob']:0.3f}")

        print("Running time of the Takahashi-Takami algorithm: {0:^5.3f} c".format(stat['tt_time']))
        print("Simulation time: {0:^5.3f} c".format(stat['sim_time']))
        print("{0:^25s}".format("First 10 probabilities of QS states"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Numerical", "Sim"))
        print("-" * 32)
        for i in range(11):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, stat["p_tt"][i], stat["p_sim"][i]))

        print("\n")
        print("{0:^25s}".format("Initial waiting times in the QS"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Numerical", "Sim"))
        print("-" * 32)
        for i in range(3):
            calc_mom = stat["w_tt"][i].real if isinstance(stat["w_tt"][i], complex) else stat["w_tt"][i]
            sim_mom =  stat["w_sim"][i].real if isinstance(stat["w_sim"][i], complex) else stat["w_sim"][i]
            print(f"{i+1:^4d}|{calc_mom:^15.3g}|{sim_mom:^15.3g}")


def dump_stat(experiments_stat, save_name='run_stat.json'):
    """
    Dump the results of experiments to a JSON file.
    :param experiments_stat: A list of dictionaries containing the results of the experiments.
    :param save_name: The name of the file to save the results to. Defaults to 'run_stat.json'.
    """
    with open(save_name, 'w') as f:
        json.dump(experiments_stat, f)


def load_stat(stat_name):
    """
    Load the results of experiments from a JSON file.
    :param stat_name: The name of the file to load the results from.
    :return: A list of dictionaries containing the results of the experiments.
    """
    with open(stat_name, 'r') as f:
        return json.load(f)


def calc_moments_by_mean_and_coev(mean, coev):
    """
    Calculate the initial three moments (mean, variance, and skewness) based on the mean and coefficient of variation.
    :param mean: The mean value of the distribution.
    :param coev: The coefficient of variation (standard deviation divided by the mean).
    :return: A list containing the calculated moments [mean, variance, skewness].
    """
    b = [0.0] * 3
    alpha = 1 / (coev ** 2)
    b[0] = mean
    b[1] = math.pow(b[0], 2) * (math.pow(coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    return b


def make_plot(experiments_stat, w_moments_num=0,
              param_name="ro", mode='error'):
    """
    Build plot for wait times initial moments
    :param experiments_stat: list of experiment statistics
    :param w_moments_num: number of moment to plot (0 - mean, 1 - variance, 2 - skewness)
    :param param_name: name of parameter to use for x-axis
    :param mode: 'error' or 'mass' - whether to plot errors or mass values
    :return: figure and axis objects
    """
    fig, ax = plt.subplots()
    w_sim_mass = []
    w_tt_mass = []
    xs = []
    if mode == 'error':
        errors = []

    for stat in experiments_stat:
        if mode == 'error':
            w_sim = stat["w_sim"][w_moments_num]
            w_tt = stat["w_tt"][w_moments_num]
            errors.append(100 * (w_sim - w_tt) / w_tt)
        else:
            w_sim_mass.append(stat["w_sim"][w_moments_num])
            w_tt_mass.append(stat["w_tt"][w_moments_num])

        if param_name != 'delay_mean':
            xs.append(stat[param_name])
        else:
            xs.append(stat["b_d"][0])

    if mode == 'error':
        ax.plot(xs, errors)
        ax.set_ylabel(r"$\varepsilon$, %")
    else:
        ax.plot(xs, w_sim_mass, label="Sim")
        ax.plot(xs, w_tt_mass, label="Calc")
        ax.set_ylabel(r"$\omega_{1}$")
        plt.legend()

    if param_name == 'ro':
        ax.set_xlabel(r"$\rho$")
    elif param_name == 'coev':
        ax.set_xlabel(r"$\nu$")
    elif param_name == 'n':
        ax.set_xlabel("n")
    elif param_name == "delay_mean":
        ax.set_xlabel("t, с")

    plt.show()