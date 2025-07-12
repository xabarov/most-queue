"""
Test the simulation model for an M/M/n/r system
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn
from most_queue.theory.fifo.mmnr import MMnrCalc

# Open config.yaml

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])
ARRIVAL_RATE = float(params['arrival']['rate'])
NUM_OF_JOBS = int(params['num_of_jobs'])
BUFFER = int(params['buffer'])
UTILIZATION_FACTOR = float(params['utilization_factor'])

ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])


def test_sim_mmnr():
    """
    Test the simulation model for an M/M/n/r system
    """
    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR *
                         NUM_OF_CHANNELS)  # Service intensity

    # Create simulation instance
    qs = QsSim(NUM_OF_CHANNELS, buffer=BUFFER)

    # Set arrival process parameters and distribution as exponential
    qs.set_sources(ARRIVAL_RATE, 'M')
    # Set service time parameters and distribution as exponential
    qs.set_servers(mu, 'M')

    # Run the simulation
    qs.run(NUM_OF_JOBS)
    # Get initial moments of waiting time. Also can get .v for sojourn times,
    # probabilities of states .get_p(), periods of continuous occupancy .pppz
    w_sim = qs.w

    mmnr = MMnrCalc(ARRIVAL_RATE, mu, NUM_OF_CHANNELS, BUFFER)
    w = mmnr.get_w()
    times_print(w_sim, w)

    # Verify simulation results against theoretical calculations
    assert np.allclose(w_sim, w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_sim_mdn():
    """
    Test the simulation model for a M/D/n
    """
    qs = QsSim(NUM_OF_CHANNELS)

    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR *
                         NUM_OF_CHANNELS)  # Service intensity

    qs.set_sources(ARRIVAL_RATE, 'M')
    # Using same load coefficient as before
    qs.set_servers(1.0 / mu, 'D')

    qs.run(NUM_OF_JOBS)

    mdn = MDn(ARRIVAL_RATE, 1 / mu, NUM_OF_CHANNELS)
    p_num = mdn.calc_p()
    p_sim = qs.get_p()

    probs_print(p_sim=p_sim, p_num=p_num, size=10)

    assert np.allclose(p_sim[:10], p_num[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_sim_mmnr()
    test_sim_mdn()
