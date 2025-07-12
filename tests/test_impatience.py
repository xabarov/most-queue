"""
Test for M/M/1 queue with exponential impatience.
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.impatient import ImpatientQueueSim
from most_queue.theory.impatience.mm1 import MM1Impatience

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)


ARRIVAL_RATE = float(params['arrival']['rate'])
UTILIZATION_FACTOR = float(params['utilization_factor'])

NUM_OF_JOBS = int(params['num_of_jobs'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

NUM_OF_CHANNELS = 1
IMPATIENCE_RATE = 0.2


def test_impatience():
    """
    Test for M/M/1 queue with exponential impatience.
    """
    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)  # service rate

    # Calculate theoretical results
    imp_calc = MM1Impatience(ARRIVAL_RATE, mu, IMPATIENCE_RATE)
    v1 = imp_calc.get_v1()
    p_num = imp_calc.probs

    # Simulate the queue
    qs = ImpatientQueueSim(NUM_OF_CHANNELS)

    qs.set_sources(ARRIVAL_RATE, 'M')
    qs.set_servers(mu, 'M')
    qs.set_impatience(IMPATIENCE_RATE, 'M')

    qs.run(NUM_OF_JOBS)

    v1_sim = qs.v[0]
    p_sim = qs.get_p()

    # Print results

    times_print(v1_sim, v1, is_w=False)
    probs_print(p_sim, p_num)

    assert abs(v1 - v1_sim) < 0.02

    assert np.allclose(p_sim[:10], p_num[:10],
                       rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_impatience()
