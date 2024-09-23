from theory.mg1_warm_calc import *
from sim.qs_sim import QueueingSystemSimulator
from utils.tables import probs_print, times_print

import numpy as np


def test_mg1_warm():
    l = 1
    b1 = 0.5
    b1_warm = 0.4
    coev = 1.3
    b_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*b_params, 4)

    b_warm_params = rd.H2_dist.get_params_by_mean_and_coev(b1_warm, coev)
    b_warm = rd.H2_dist.calc_theory_moments(*b_params, 4)

    qs = QueueingSystemSimulator(1)
    qs.set_servers(b_params, "H")
    qs.set_warm(b_warm_params, "H")
    qs.set_sources(l, "M")
    qs.run(100000)

    v_ch = get_v(l, b, b_warm)
    v_sim = qs.v

    times_print(v_sim, v_ch, False)
    
    assert len(v_sim) == len(v_ch)



