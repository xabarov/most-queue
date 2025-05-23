"""
Collect results for the QS M/G/n queue with disasters.
"""
import json
import math
from dataclasses import asdict

import numpy as np

from most_queue.general.plots import DependsType, plot_sim_vs_calc_moments
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.structs import (
    DependsOnChannelsResults,
    DependsOnJSONEncoder,
    DependsOnUtilizationResults,
    DependsOnVariationResults,
)


def calc_service_moments(channels: int, utilization_factor: float,
                         service_time_variation_coef: float,
                         l_pos: float):
    """
    Gamma service time moments calculation.
    """
    b1 = channels * utilization_factor / l_pos  # average service time

    b = [0.0] * 3
    alpha = 1 / (service_time_variation_coef ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * \
        (math.pow(service_time_variation_coef, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    return b


def collect_calc_results(channels: int, b: list[float],
                         l_pos: float, l_neg: float, max_p: int = 100, accuracy=1e-8):
    """
    Collects calculation results for a given number of channels.
    :param channels: Number of channels.
    :param b: Service time moments.
    :param l_pos: Positive arrivals rate.
    :param l_neg: Negative arrivals rate.
    """
    queue_calc = MGnNegativeDisasterCalc(
        channels, l_pos, l_neg, b, verbose=False, accuracy=accuracy)

    queue_calc.run()

    return queue_calc.get_results(max_p=max_p)


def collect_sim_results(channels: int, b: list[float],
                        l_pos: float, l_neg: float, num_of_jobs: int,
                        max_p: int = 100):
    """
    Collects simulation results for a given number of channels.
    :param channels: Number of channels in the system.
    :param b: List of service time initial moments.
    :param l_pos: Rate parameter for positive arrivals.
    :param l_neg: Rate parameter for negative arrivals.
    :param num_of_jobs: Number of jobs to simulate.
    """
    # Run simulation
    queue_sim = QsSimNegatives(
        channels, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(num_of_jobs)

    queue_sim.run(num_of_jobs)

    return queue_sim.get_results(max_p=max_p)


def run_depends_on_channels(l_pos: float, l_neg: float, channels_max: int = 10,
                            utilization_factor: float = 0.7,
                            service_time_variation_coef: float = 1.57,
                            num_of_jobs=300000, accuracy=1e-8) -> DependsOnChannelsResults:
    """
    Collect results for M/G/n queue with disasters depends on number of channels.
    :param l_pos: rate of positive jobs arrival.
    :param l_neg: rate of negative jobs arrival.
    :param channels_max: max number of channels to test.
    :param utilization_factor: utilization factor for the queue.
    :param service_time_variation_coef: coefficient of variation of service time.
    :param num_of_jobs: number of jobs in simulation.
    :return: DependsOnChannelsResults object with calculated and simulated moments.
    """
    channels = list(range(1, channels_max + 1))

    calc_results = []
    sim_results = []

    for n in channels:
        print(f'Channels: {n}')

        b = calc_service_moments(channels=n,
                                 utilization_factor=utilization_factor,
                                 service_time_variation_coef=service_time_variation_coef,
                                 l_pos=l_pos)

        # collect results
        calc_results.append(collect_calc_results(
            channels=n, b=b, l_pos=l_pos, l_neg=l_neg, accuracy=accuracy))
        sim_results.append(collect_sim_results(channels=n, l_pos=l_pos, l_neg=l_neg, b=b,
                                               num_of_jobs=num_of_jobs))

    return DependsOnChannelsResults(calc=calc_results, sim=sim_results,
                                    channels=channels,
                                    utilization_factor=utilization_factor,
                                    service_time_variation_coef=service_time_variation_coef)


def run_depends_on_varience(l_pos: float, l_neg: float, channels: int = 3,
                            utilization_factor: float = 0.7,
                            service_var_coef_min: float = 0.3,
                            service_var_coef_max: float = 3.0,
                            service_var_coef_step: float = 0.1,
                            num_of_jobs=300000, accuracy=1e-8) -> DependsOnVariationResults:
    """
    Collects simulation and calculation results for a given set of parameters,
      varying the service time variation coefficient
      and returns them as a DependsOnVariationResults object.
      :param l_pos: Positive arrival rate.
      :param l_neg: Negative arrival rate.
      :param channels: Number of channels in the system.
      :param utilization_factor: Utilization factor of the system.
      :param service_var_coef_min: Minimum value of the service time variation coefficient.
      :param service_var_coef_max: Maximum value of the service time variation coefficient.
      :param service_var_coef_step: Step size for the service time variation coefficient.
      :param num_of_jobs: Number of jobs to simulate.
      :return: A DependsOnVariationResults object containing the simulation and calculation results.
    """
    service_time_variation_coefs = np.arange(service_var_coef_min,
                                             service_var_coef_max + service_var_coef_step,
                                             service_var_coef_step)

    calc_results = []
    sim_results = []

    for coef in service_time_variation_coefs:
        print(f'Service time variation coefficient: {coef:0.2f}')
        b = calc_service_moments(channels=channels,
                                 utilization_factor=utilization_factor,
                                 service_time_variation_coef=coef,
                                 l_pos=l_pos)

        # collect results
        calc_results.append(collect_calc_results(
            channels=channels, b=b, l_pos=l_pos, l_neg=l_neg, accuracy=accuracy))
        sim_results.append(collect_sim_results(channels=channels, l_pos=l_pos, l_neg=l_neg, b=b,
                                               num_of_jobs=num_of_jobs))

    return DependsOnVariationResults(
        service_time_variation_coef=service_time_variation_coefs.tolist(),
        calc=calc_results,
        sim=sim_results,
        channels=channels,
        utilization_factor=utilization_factor
    )


def run_depends_on_utilization(l_pos: float, l_neg: float, channels: int = 3,
                               service_time_variation_coef: float = 1.2,
                               utilization_factor_min: float = 0.1,
                               utilization_factors_max: float = 0.9,
                               utilization_factor_step: float = 0.05,
                               num_of_jobs=300000, accuracy=1e-8) -> DependsOnUtilizationResults:
    """
    Collects simulation and calculation results for a given range of utilization factors.
    :param l_pos: Positive arrival rate parameter.
    :type l_pos: float
    :param l_neg: Negative arrival rate parameter.
    :type l_neg: float
    :param channels: Number of service channels.
    :type channels: int
    :param service_time_variation_coef: Coefficient of variation for the service time.
    :type service_time_variation_coef: float
    :param utilization_factor_min: Minimum utilization factor to consider.
    :type utilization_factor_min: float
    :param utilization_factors_max: Maximum utilization factor to consider.
    :type utilization_factors_max: float
    :param utilization_factor_step: Step size for the range of utilization factors.
    :type utilization_factor_step: float
    :param num_of_jobs: Number of jobs to simulate.
    :type num_of_jobs: int
    :return: A DependsOnUtilizationResults object containing the collected results.
    :rtype: DependsOnUtilizationResults
    """
    utilization_factors = np.arange(utilization_factor_min,
                                    utilization_factors_max + utilization_factor_step,
                                    utilization_factor_step)
    sim_results = []
    calc_results = []
    for utilization_factor in utilization_factors:
        print(f"Utilization factor: {utilization_factor:0.2f}")
        b = calc_service_moments(channels=channels,
                                 utilization_factor=utilization_factor,
                                 service_time_variation_coef=service_time_variation_coef,
                                 l_pos=l_pos)

        # collect results
        calc_results.append(collect_calc_results(
            channels=channels, b=b, l_pos=l_pos, l_neg=l_neg, accuracy=accuracy))
        sim_results.append(collect_sim_results(channels=channels, l_pos=l_pos, l_neg=l_neg, b=b,
                                               num_of_jobs=num_of_jobs))

    return DependsOnUtilizationResults(
        utilization_factor=utilization_factors.tolist(),
        calc=calc_results,
        sim=sim_results, channels=channels,
        service_time_variation_coef=service_time_variation_coef)


if __name__ == "__main__":

    import os

    L_POS = 1.0
    L_NEG = 0.5
    NUM_OF_JOBS = 1000000
    ACCURACY = 1e-10
    UTILIZATION_FACTOR = 0.7
    CHANNELS_MAX = 10
    CHANNELS_NUM = 3
    SERVICE_TIME_COEF_VARIANCE = 1.2
    COEF_MAX = 3.0

    APPENDIX = f"n_{CHANNELS_NUM}_"
    APPENDIX += "_".join([f"{k}_{v:.2f}" for k, v in [('u', UTILIZATION_FACTOR),
                         ('b', SERVICE_TIME_COEF_VARIANCE), ('l_neg', L_NEG)]])

    EXP_DIR_NAME = f'works/results/disaster_{APPENDIX}'
    IS_EXISTS = False
    if not os.path.exists(EXP_DIR_NAME):
        os.makedirs(EXP_DIR_NAME)
    else:
        IS_EXISTS = True

    if IS_EXISTS:
        with open(os.path.join(f"{EXP_DIR_NAME}/channels", 'results.json'), 'r', encoding='utf-8') as f:
            results_channels = json.load(f)
        with open(os.path.join(f"{EXP_DIR_NAME}/coefs", 'results.json'), 'r', encoding='utf-8') as f:
            results_coefs = json.load(f)
        with open(os.path.join(f"{EXP_DIR_NAME}/utilization", 'results.json'), 'r', encoding='utf-8') as f:
            results_utilization = json.load(f)

        xs_channels = results_channels['channels']
        xs_coefs = results_coefs['service_time_variation_coef']
        xs_utilization = results_utilization['utilization']
    else:
        results_channels = run_depends_on_channels(
            l_pos=L_POS, l_neg=L_NEG, channels_max=CHANNELS_MAX,
            utilization_factor=UTILIZATION_FACTOR,
            service_time_variation_coef=SERVICE_TIME_COEF_VARIANCE,
            num_of_jobs=NUM_OF_JOBS, accuracy=ACCURACY)

        results_coefs = run_depends_on_varience(
            l_pos=L_POS, l_neg=L_NEG, channels=CHANNELS_NUM, num_of_jobs=NUM_OF_JOBS,
            utilization_factor=UTILIZATION_FACTOR, accuracy=ACCURACY, service_var_coef_max=COEF_MAX)

        results_utilization = run_depends_on_utilization(
            l_pos=L_POS, l_neg=L_NEG, channels=CHANNELS_NUM,
            service_time_variation_coef=SERVICE_TIME_COEF_VARIANCE,
            num_of_jobs=NUM_OF_JOBS, accuracy=ACCURACY)

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

        if IS_EXISTS:
            v_sim_ave = [r['v'][0] for r in results['sim']]
            v_sim_broken_ave = [r['v_broken'][0] for r in results['sim']]
            v_sim_served_ave = [r['v_served'][0] for r in results['sim']]
            v_calc_broken_ave = [r['v_broken'][0] for r in results['calc']]
            v_calc_served_ave = [r['v_served'][0] for r in results['calc']]
            v_calc_ave = [r['v'][0] for r in results['calc']]

            w_sim_ave = [r['w'][0] for r in results['sim']]
            w_calc_ave = [r['w'][0] for r in results['calc']]
        else:
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
