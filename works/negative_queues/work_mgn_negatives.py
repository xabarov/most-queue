"""
Collect results for the QS M/G/n queue with negative jobs and RCS discipline.
"""
import json
import math
import os
from dataclasses import asdict

import numpy as np
import yaml

from most_queue.general.plots import DependsType, plot_sim_vs_calc_moments
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc
from most_queue.theory.negative.structs import (
    DependsOnChannelsResults,
    DependsOnJSONEncoder,
    DependsOnUtilizationResults,
    DependsOnVariationResults,
)


def calc_moments_by_mean_and_coev(mean, coev):
    """
    Calculate the E[X^k] for k=0,1,2
    for a distribution with given mean and coefficient of variation.
    :param mean: The mean value of the distribution.
    :param coev: The coefficient of variation (standard deviation divided by the mean).
    :return: A list containing the calculated moments
    """
    b = [0.0] * 3
    alpha = 1 / (coev ** 2)
    b[0] = mean
    b[1] = math.pow(b[0], 2) * (math.pow(coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    return b


def collect_calc_results(qp: dict, n: int, b: list[float], discipline:
                         NegativeServiceType, max_p: int = 100):
    """
    Collects calculation results for a given number of channels.
    :param qp: Dictionary containing the parameters for the simulation.
    """

    if discipline == NegativeServiceType.RCS:
        queue_calc = MGnNegativeRCSCalc(
            n, float(qp['arrival_rate']['positive']),
            float(qp['arrival_rate']['negative']),
            b, verbose=False, accuracy=float(qp['accuracy']))

    else:
        # disasters
        queue_calc = MGnNegativeDisasterCalc(
            n, float(qp['arrival_rate']['positive']),
            float(qp['arrival_rate']['negative']),
            b, verbose=False, accuracy=float(qp['accuracy']))

    queue_calc.run()

    return queue_calc.get_results(max_p=max_p)


def collect_sim_results(qp: dict, b: list[float], n: int, discipline: NegativeServiceType,
                        max_p: int = 100):
    """
    Collects simulation results for a given number of channels.
    :param qp: Dictionary containing the parameters for the simulation.
    """
    # Run simulation
    queue_sim = QsSimNegatives(n, discipline)

    queue_sim.set_negative_sources(float(qp['arrival_rate']['negative']), 'M')
    queue_sim.set_positive_sources(float(qp['arrival_rate']['positive']), 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(int(qp['num_of_jobs']))

    return queue_sim.get_results(max_p=max_p)


def run_depends_on_channels(qp: dict, discipline: NegativeServiceType,
                            max_p: int = 100) -> DependsOnChannelsResults:
    """
    Collect results for M/G/n queue with negative jobs 
    and RCS discipline depends on number of channels.
    :param qp: Dictionary with queue parameters.
    :return: DependsOnChannelsResults object with calculated and simulated moments.
    """
    channels = list(range(1, qp['channels']['max'] + 1))

    calc_results = []
    sim_results = []

    for n in channels:
        print(f'Channels: {n}')

        service_mean = n*qp['utilization']['base'] / \
            qp['arrival_rate']['positive']

        b = calc_moments_by_mean_and_coev(
            service_mean, qp['service']['cv']['base'])

        # collect results
        calc_results.append(collect_calc_results(
            qp=qp, b=b, n=n, discipline=discipline, max_p=max_p))
        sim_results.append(collect_sim_results(
            qp=qp, b=b, n=n, discipline=discipline, max_p=max_p))

    return DependsOnChannelsResults(calc=calc_results, sim=sim_results,
                                    channels=channels,
                                    utilization_factor=qp['utilization']['base'],
                                    service_time_variation_coef=qp['service']['cv']['base'])


def run_depends_on_varience(qp: dict, discipline: NegativeServiceType,
                            max_p: int = 100) -> DependsOnVariationResults:
    """
    Collects simulation and calculation results for a given set of parameters,
      varying the service time variation coefficient
      and returns them as a DependsOnVariationResults object.
      :param qp: Dictionary containing the parameters for the simulation.
      :return: A DependsOnVariationResults object containing the simulation and calculation results.
    """
    service_time_variation_coefs = np.linspace(
        qp['service']['cv']['min'], qp['service']['cv']['max'], qp['service']['cv']['num_points'])
    calc_results = []
    sim_results = []

    for coef in service_time_variation_coefs:
        print(f'Service time variation coefficient: {coef:0.2f}')
        service_mean = qp['channels']['base'] * \
            qp['utilization']['base']/qp['arrival_rate']['positive']

        b = calc_moments_by_mean_and_coev(service_mean, coef)
        # collect results
        calc_results.append(collect_calc_results(n=qp['channels']['base'],
            qp=qp, b=b, discipline=discipline, max_p=max_p))
        sim_results.append(collect_sim_results(n=qp['channels']['base'],
            qp=qp, b=b, discipline=discipline, max_p=max_p))

    return DependsOnVariationResults(
        service_time_variation_coef=service_time_variation_coefs.tolist(),
        calc=calc_results,
        sim=sim_results,
        channels=qp['channels']['base'],
        utilization_factor=qp['utilization']['base']
    )


def run_depends_on_utilization(qp: dict, discipline: NegativeServiceType,
                               max_p: int = 100) -> DependsOnUtilizationResults:
    """
    Collects simulation and calculation results for a given range of utilization factors.
    :param qp: Dictionary containing the parameters for the simulation.
    :return: A DependsOnUtilizationResults object containing the collected results.
    :rtype: DependsOnUtilizationResults
    """
    utilization_factors = np.linspace(
        qp['utilization']['min'], qp['utilization']['max'], qp['utilization']['num_points'])
    sim_results = []
    calc_results = []
    for rho in utilization_factors:
        print(f"Utilization factor: {rho:0.2f}")
        service_mean = qp['channels']['base'] * \
            rho/qp['arrival_rate']['positive']

        b = calc_moments_by_mean_and_coev(
            service_mean, qp['service']['cv']['base'])

        # collect results
        calc_results.append(collect_calc_results(n=qp['channels']['base'],
            qp=qp, b=b, discipline=discipline, max_p=max_p))
        sim_results.append(collect_sim_results(n=qp['channels']['base'],
            qp=qp, b=b, discipline=discipline, max_p=max_p))

    return DependsOnUtilizationResults(
        utilization_factor=utilization_factors.tolist(),
        calc=calc_results,
        sim=sim_results, channels=qp['channels']['base'],
        service_time_variation_coef=qp['service']['cv']['base'])


def read_parameters_from_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return the content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f_yaml:
        return yaml.safe_load(f_yaml)


if __name__ == "__main__":

    base_qp = read_parameters_from_yaml(
        "works/negative_queues/base_parameters.yaml")

    L_NEG = base_qp['arrival_rate']['negative']
    UTILIZATION_FACTOR = base_qp['utilization']['base']
    CHANNELS_NUM = base_qp['channels']['base']
    SERVICE_TIME_COEF_VARIANCE = base_qp['service']['cv']['base']

    APPENDIX = f"n_{CHANNELS_NUM}_"
    APPENDIX += "_".join([f"{k}_{v:.2f}" for k, v in [('u', UTILIZATION_FACTOR),
                         ('b', SERVICE_TIME_COEF_VARIANCE), ('l_neg', L_NEG)]])

    IS_RCS = True  # or False for queue with disasters

    CUR_DIR = os.path.dirname(__file__)

    if IS_RCS:
        queue_discipline = NegativeServiceType.RCS
        EXP_DIR_NAME = f'results/rcs/{APPENDIX}'
    else:
        queue_discipline = NegativeServiceType.DISASTER
        EXP_DIR_NAME = f'results/disaster/{APPENDIX}'

    EXP_DIR_NAME = os.path.join(CUR_DIR, EXP_DIR_NAME)

    if not os.path.exists(EXP_DIR_NAME):
        os.makedirs(EXP_DIR_NAME)

    results_channels = run_depends_on_channels(base_qp, queue_discipline)

    results_coefs = run_depends_on_varience(base_qp, queue_discipline)

    results_utilization = run_depends_on_utilization(
        base_qp, queue_discipline)

    xs_channels = results_channels.channels
    xs_coefs = results_coefs.service_time_variation_coef
    xs_utilization = results_utilization.utilization_factor

    SAVE_PATH_CHANNELS = f"{EXP_DIR_NAME}/channels"
    SAVE_PATH_COEFS = f"{EXP_DIR_NAME}/coefs"
    SAVE_PATH_UTILIZATION_FACTOR = f"{EXP_DIR_NAME}/utilization"

    for results, save_path, xs, depends_on in zip([results_channels,
                                                   results_coefs,
                                                   results_utilization],
                                                  [SAVE_PATH_CHANNELS,
                                                      SAVE_PATH_COEFS,
                                                      SAVE_PATH_UTILIZATION_FACTOR],
                                                  [xs_channels, xs_coefs,
                                                      xs_utilization],
                                                  [DependsType.CHANNELS_NUMBER,
                                                   DependsType.COEFFICIENT_OF_VARIATION,
                                                   DependsType.UTILIZATION_FACTOR]):

        v_sim_ave = [r.v[0] for r in results.sim]
        v_sim_broken_ave = [r.v_broken[0] for r in results.sim]
        v_sim_served_ave = [r.v_served[0] for r in results.sim]
        v_calc_broken_ave = [r.v_broken[0] for r in results.calc]
        v_calc_served_ave = [r.v_served[0] for r in results.calc]
        v_calc_ave = [r.v[0] for r in results.calc]

        w_sim_ave = [r.w[0] for r in results.sim]
        w_calc_ave = [r.w[0] for r in results.calc]

        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(results), f, cls=DependsOnJSONEncoder)

        plot_params = [
            {"sim_results": v_sim_ave, "calc_results": v_calc_ave,
                "save_path": os.path.join(save_path, "v_ave.png"),
             "is_errors": False, "is_waiting_time": False},
            {"sim_results": w_sim_ave, "calc_results": w_calc_ave,
                "save_path": os.path.join(save_path, "w_ave.png"),
             "is_errors": False, "is_waiting_time": True},
            {"sim_results": v_sim_broken_ave, "calc_results": v_calc_broken_ave,
                "save_path": os.path.join(save_path, "v_broken_ave.png"),
             "is_errors": False, "is_waiting_time": False},
            {"sim_results": v_sim_served_ave, "calc_results": v_calc_served_ave,
                "save_path": os.path.join(save_path, "v_served_ave.png"),
             "is_errors": False, "is_waiting_time": False},
            {"sim_results": v_sim_ave, "calc_results": v_calc_ave,
                "save_path": os.path.join(save_path, "v_ave_err.png"),
             "is_errors": True, "is_waiting_time": False},
            {"sim_results": v_sim_served_ave, "calc_results": v_calc_served_ave,
                "save_path": os.path.join(save_path, "v_served_ave_err.png"),
             "is_errors": True, "is_waiting_time": False},
            {"sim_results": v_sim_broken_ave, "calc_results": v_calc_broken_ave,
                "save_path": os.path.join(save_path, "v_broken_ave_err.png"),
             "is_errors": True, "is_waiting_time": False},
            {"sim_results": w_sim_ave, "calc_results": w_calc_ave,
                "save_path": os.path.join(save_path, "w_ave_err.png"),
             "is_errors": True, "is_waiting_time": True}
        ]

        for params in plot_params:
            plot_sim_vs_calc_moments(
                xs=xs,
                sim_results=params["sim_results"],
                calc_results=params["calc_results"],
                depends_on=depends_on,
                save_path=params["save_path"],
                is_errors=params["is_errors"],
                is_waiting_time=params['is_waiting_time']
            )
