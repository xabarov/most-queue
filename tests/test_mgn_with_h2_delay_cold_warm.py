"""
Run one simulation vs calculation for queueing system.
with H2-warming, H2-cooling and H2-delay of cooling starts.
"""
import math
import os
import time

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mgn_with_h2_delay_cold_warm import \
    MGnH2ServingColdWarmDelay

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])


ARRIVAL_RATE = float(params['arrival']['rate'])
SERVICE_TIME_CV = float(params['service']['cv'])

WARM_UP_MEAN = float(params['warm-up']['mean'])
WARM_UP_CV = float(params['warm-up']['cv'])

COOLING_MEAN = float(params['cooling']['mean'])
COOLING_CV = float(params['cooling']['cv'])

COOLING_DELAY_MEAN = float(params['cooling-delay']['mean'])
COOLING_DELAY_CV = float(params['cooling-delay']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])


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


def run_calculation(arrival_rate: float, b: list[float],
                    b_w: list[float], b_c: list[float], b_d: list[float],
                    num_channels: int):
    """
    Calculation of an M/H2/n queue with H2-warming, H2-cooling and H2-delay 
    of the start of cooling using Takahasi-Takami method.
    Args:
       arrival_rate (float): The arrival rate of the queue.
        b (list): A list containing the E[X^k] k=0, 1, 2. for the service time distribution.
        b_w (list): A list containing the E[X^k] k=0, 1, 2. for the warmup time distribution.
        b_c (list): A list containing the E[X^k] k=0, 1, 2. for the cooling time distribution.
        b_d (list): A list containing the E[X^k] k=0, 1, 2. for the delay time distribution.
        num_of_channels (int): The number of channels in the queue.
    Returns:
        dict: A dictionary containing the statistics of the queue.
    """
    num_start = time.process_time()

    solver = MGnH2ServingColdWarmDelay(
        arrival_rate, b, b_w, b_c, b_d, num_channels)

    solver.run()

    stat = {}
    stat["w"] = solver.get_w()
    stat["process_time"] = time.process_time() - num_start
    stat["p"] = solver.get_p()[:10]
    stat["num_of_iter"] = solver.num_of_iter_

    stat["warmup_prob"] = solver.get_warmup_prob()
    stat["cold_prob"] = solver.get_cold_prob()
    stat["cold_delay_prob"] = solver.get_cold_delay_prob()
    stat['servers_busy_probs'] = solver.get_probs_of_servers_busy()

    return stat


def run_simulation(arrival_rate: float, b: list[float],
                   b_w: list[float], b_c: list[float], b_d: list[float],
                   num_channels: int, num_of_jobs: int = 300_000, ave_num: int = 10):
    """
    Run simulation for an M/H2/n queue with H2-warming, 
    H2-cooling and H2-delay before cooling starts.
    Args:
       arrival_rate (float): The arrival rate of the queue.
        b (list): A list containing the E[X^k] k=0, 1, 2. for the service time distribution.
        b_w (list): A list containing the E[X^k] k=0, 1, 2. for the warmup time distribution.
        b_c (list): A list containing the E[X^k] k=0, 1, 2. for the cooling time distribution.
        b_d (list): A list containing the E[X^k] k=0, 1, 2. for the delay time distribution.
        num_of_channels (int): The number of channels in the queue.
        num_of_jobs (int): The number of jobs to simulate.
    Returns:
        dict: A dictionary containing the statistics of the queue.
    """

    gamma_params = GammaDistribution.get_params(b)
    gamma_params_warm = GammaDistribution.get_params(b_w)
    gamma_params_cold = GammaDistribution.get_params(b_c)
    gamma_params_cold_delay = GammaDistribution.get_params(b_d)

    ws = []
    ps = []
    process_times = []
    warmup_probs = []
    cold_probs = []
    cold_delay_probs = []

    for sim_run_num in range(ave_num):
        print(f"Running simulation {sim_run_num + 1} of {ave_num}")

        im_start = time.process_time()
        sim = VacationQueueingSystemSimulator(num_channels)
        sim.set_sources(arrival_rate, 'M')

        sim.set_servers(gamma_params, 'Gamma')
        sim.set_warm(gamma_params_warm, 'Gamma')
        sim.set_cold(gamma_params_cold, 'Gamma')
        sim.set_cold_delay(gamma_params_cold_delay, 'Gamma')
        sim.run(num_of_jobs)

        ws.append(sim.w)
        process_times.append(time.process_time() - im_start)
        cold_probs.append(sim.get_cold_prob())
        cold_delay_probs.append(sim.get_cold_delay_prob())
        warmup_probs.append(sim.get_warmup_prob())
        ps.append(sim.get_p()[:10])

    # average over all simulations

    stat = {}

    stat["w"] = np.mean(ws, axis=0).tolist()
    stat["process_time"] = np.sum(process_times)
    stat["cold_prob"] = np.mean(cold_probs)
    stat["cold_delay_prob"] = np.mean(cold_delay_probs)
    stat["warmup_prob"] = np.mean(warmup_probs)
    stat["p"] = np.mean(ps, axis=0).tolist()

    return stat


def read_parameters_from_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return the content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_mgn_h2_delay_cold_warm():
    """
    Test the M/G/N queue with H2 delay, cold and warm phases.
    """
    service_time_mean = NUM_OF_CHANNELS*UTILIZATION_FACTOR/ARRIVAL_RATE

    # Calculate initial moments for service time, warm-up time,
    # cool-down time, and delay before cooling starts.
    b_service = calc_moments_by_mean_and_coev(
        service_time_mean, SERVICE_TIME_CV)
    b_warmup = calc_moments_by_mean_and_coev(
        WARM_UP_MEAN, WARM_UP_CV)
    b_cooling = calc_moments_by_mean_and_coev(COOLING_MEAN, COOLING_CV)
    b_delay = calc_moments_by_mean_and_coev(
        COOLING_DELAY_MEAN, COOLING_DELAY_CV)

    num_results = run_calculation(
        arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNELS, b=b_service,
        b_w=b_warmup, b_c=b_cooling, b_d=b_delay
    )
    sim_results = run_simulation(
        arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNELS, b=b_service,
        b_w=b_warmup, b_c=b_cooling, b_d=b_delay, num_of_jobs=NUM_OF_JOBS,
        ave_num=1
    )

    probs_print(p_sim=sim_results["p"], p_num=num_results["p"], size=10)
    times_print(sim_moments=sim_results["w"], calc_moments=num_results["w"])

    # Print the results for the number of busy servers
    print("Probability distribution of number of busy servers:")
    for i, prob in enumerate(num_results['servers_busy_probs']):
        print(f'\t{i}: {prob: 0.4f}')

    assert np.allclose(
        sim_results["w"], num_results["w"], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results["p"][:10], num_results["p"][:10],
                       atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == '__main__':

    test_mgn_h2_delay_cold_warm()
