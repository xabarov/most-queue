"""
Tests for the MAP/PH/c QBD calculator (EPIC-007): correlated arrivals,
phase-type service, c servers. Cross-validated against MAP/M/c, MAP/PH/1,
the Takahashi-Takami M/H2/c and simulation.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_waiting_moments, probs_print
from most_queue.random.distributions import H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.matrix.map_mmc import MapMMcCalc
from most_queue.theory.matrix.map_ph1 import MapPh1Calc
from most_queue.theory.matrix.map_phc import MapPhCCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_CHANNELS = int(params["num_of_channels"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

MMPP = MAP.mmpp([2.5 * ARRIVAL_RATE, 0.5 * ARRIVAL_RATE], np.array([[-0.15, 0.15], [0.25, -0.25]]))


def test_map_phc_exp_service_equals_map_mmc():
    """PH service with a single phase (exponential) reduces MAP/PH/c to MAP/M/c."""
    lam = MAP.arrival_rate(MMPP)
    mu = lam / (NUM_OF_CHANNELS * UTILIZATION_FACTOR)

    phc = MapPhCCalc(n=NUM_OF_CHANNELS)
    phc.set_sources(MMPP)
    phc.set_servers(PHDistribution.from_exp(mu))
    r_phc = phc.run()

    mmc = MapMMcCalc(n=NUM_OF_CHANNELS)
    mmc.set_sources(MMPP)
    mmc.set_servers(mu)
    r_mmc = mmc.run()

    assert np.isclose(r_phc.w[0], r_mmc.w[0], rtol=1e-8), ERROR_MSG
    assert np.allclose(r_phc.p[:20], r_mmc.p[:20], atol=1e-9), ERROR_MSG


def test_map_phc_c1_equals_map_ph1():
    """c=1 reduces MAP/PH/c to the single-server MAP/PH/1."""
    h2 = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(0.5, 1.4))

    phc = MapPhCCalc(n=1)
    phc.set_sources(MMPP)
    phc.set_servers(h2)
    r_phc = phc.run()

    ph1 = MapPh1Calc()
    ph1.set_sources(MMPP)
    ph1.set_servers(h2)
    r_ph1 = ph1.run()

    assert np.isclose(r_phc.w[0], r_ph1.w[0], rtol=1e-6), ERROR_MSG
    assert np.allclose(r_phc.p[:15], r_ph1.p[:15], atol=1e-8), ERROR_MSG


def test_map_phc_poisson_equals_takahashi_takami():
    """Poisson arrivals + H2 service reproduce the exact M/H2/c (Takahashi-Takami)."""
    lam, c, cv = ARRIVAL_RATE, NUM_OF_CHANNELS, 1.5
    b_mean = c * UTILIZATION_FACTOR / lam
    h2_real = H2Distribution.get_params_by_mean_and_cv(b_mean, cv)

    phc = MapPhCCalc(n=c)
    phc.set_sources(MAP.poisson(lam))
    phc.set_servers(PHDistribution.from_h2(h2_real))
    r_phc = phc.run()

    tt = MGnCalc(n=c)
    tt.set_sources(l=lam)
    tt.set_servers(h2_real)
    r_tt = tt.run()

    print_waiting_moments([r_phc.w[0], 0, 0], r_tt.w)
    assert np.isclose(r_phc.w[0], r_tt.w[0], rtol=1e-4), ERROR_MSG
    assert np.allclose(r_phc.p[:15], r_tt.p[:15], atol=1e-4), ERROR_MSG


def test_map_phc_vs_sim():
    """MAP(MMPP-2)/PH(H2)/c against simulation with MAP source and PH servers."""
    lam = MAP.arrival_rate(MMPP)
    service_mean = NUM_OF_CHANNELS * UTILIZATION_FACTOR / lam
    h2 = H2Distribution.get_params_by_mean_and_cv(service_mean, 1.3)
    ph = PHDistribution.from_h2(h2)

    calc = MapPhCCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(MMPP)
    calc.set_servers(ph)
    calc_results = calc.run()

    sim = QsSim(NUM_OF_CHANNELS, seed=42)
    sim.set_sources(MMPP, "MAP")
    sim.set_servers(ph, "PH")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    print_waiting_moments(sim_results.w, calc_results.w)

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_map_phc_exp_service_equals_map_mmc()
    test_map_phc_c1_equals_map_ph1()
    test_map_phc_poisson_equals_takahashi_takami()
    test_map_phc_vs_sim()
