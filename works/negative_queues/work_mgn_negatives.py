"""
Collect results for the QS M/G/n queue with negative jobs and RCS discipline.
"""

import argparse
import copy
import json
import os
import runpy
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

from most_queue.io.plots import DependsType, Plotter
from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import DisasterScenario, NegativeServiceType, QsSimNegatives, RcsScenario
from most_queue.structs import (
    DependsOnChannelsResults,
    DependsOnJSONEncoder,
    DependsOnNegativeRateResults,
    DependsOnUtilizationResults,
    DependsOnVariationResults,
)
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc


def collect_calc_results(
    qp: dict,
    n: int,
    b: list[float],
    discipline: NegativeServiceType,
    requeue_on_negative: bool = False,
    resume_on_negative: bool = False,
    repeat_without_resampling: bool = False,
    max_p: int = 100,
):
    """
    Collects calculation results for a given number of channels.
    :param qp: Dictionary containing the parameters for the simulation.
    :param requeue_on_negative: if True, use "repeat service" scenario (requeue); else "remove" / "clear".
    :param resume_on_negative: if True, use "resume service" (sample once; preemptive-resume) scenario.
    :param repeat_without_resampling: if True, use "repeat without resampling" (sample once; progress lost) scenario.
    """
    if discipline == NegativeServiceType.RCS:
        queue_calc = MGnNegativeRCSCalc(
            n=n,
            calc_params=TakahashiTakamiParams(tolerance=float(qp["accuracy"])),
            requeue_on_disaster=requeue_on_negative,
            resume_on_negative=resume_on_negative,
            repeat_without_resampling=repeat_without_resampling,
        )
        queue_calc.set_sources(l_pos=float(qp["arrival_rate"]["positive"]), l_neg=float(qp["arrival_rate"]["negative"]))
        queue_calc.set_servers(b=b)
    else:
        queue_calc = MGnNegativeDisasterCalc(
            n=n,
            calc_params=TakahashiTakamiParams(tolerance=float(qp["accuracy"])),
            requeue_on_disaster=requeue_on_negative,
            resume_on_disaster=resume_on_negative,
            repeat_without_resampling=repeat_without_resampling,
        )
        queue_calc.set_sources(l_pos=float(qp["arrival_rate"]["positive"]), l_neg=float(qp["arrival_rate"]["negative"]))
        queue_calc.set_servers(b=b)

    queue_results = queue_calc.run()
    queue_results.p = queue_results.p[:max_p]
    return queue_results


def collect_sim_results(
    qp: dict,
    b: list[float],
    n: int,
    discipline: NegativeServiceType,
    rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
    max_p: int = 100,
):
    """
    Collects simulation results for a given number of channels.
    :param qp: Dictionary containing the parameters for the simulation.
    :param rcs_scenario: RcsScenario.REMOVE or REQUEUE (used when discipline is RCS).
    :param disaster_scenario: DisasterScenario.CLEAR_SYSTEM or REQUEUE_ALL (used when discipline is DISASTER).
    """
    queue_sim = QsSimNegatives(
        n,
        discipline,
        verbose=False,
        rcs_scenario=rcs_scenario,
        disaster_scenario=disaster_scenario,
    )
    queue_sim.set_negative_sources(float(qp["arrival_rate"]["negative"]), "M")
    queue_sim.set_positive_sources(float(qp["arrival_rate"]["positive"]), "M")
    # Simulator: for CV>1 use H2 (matches theory's H2 structure better),
    # otherwise keep Gamma (stable for CV<=1).
    mean_b = float(b[0])
    var_b = float(b[1] - b[0] ** 2)
    cv_b = float(np.sqrt(max(0.0, var_b)) / mean_b) if mean_b > 0 else 0.0
    if cv_b > 1.0 + 1e-12:
        h2_params = H2Distribution.get_params(b)
        queue_sim.set_servers(h2_params, "H")
    else:
        gamma_params = GammaDistribution.get_params([b[0], b[1]])
        queue_sim.set_servers(gamma_params, "Gamma")

    sim_results = queue_sim.run(int(qp["num_of_jobs"]))
    sim_results.p = sim_results.p[:max_p]
    return sim_results


def run_depends_on_channels(
    qp: dict,
    discipline: NegativeServiceType,
    requeue_on_negative: bool = False,
    resume_on_negative: bool = False,
    repeat_without_resampling: bool = False,
    rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
    max_p: int = 100,
    service_cv: float | None = None,
) -> DependsOnChannelsResults:
    """
    Collect results for M/G/n queue with negative jobs
    and RCS discipline depends on number of channels.
    :param qp: Dictionary with queue parameters.
    :return: DependsOnChannelsResults object with calculated and simulated moments.
    """
    # NOTE: for restart disciplines the effective mean service time can grow very fast with n
    # (we keep rho and lambda_pos fixed, so E[B]=n*rho/lambda_pos increases linearly with n).
    # With a large negative rate this may push the system close to (or beyond) instability for
    # larger n and make sample means explode. For the n-sweep we use a milder fixed negative
    # intensity to keep all scenarios in a stable, interpretable regime.
    qp_run = copy.deepcopy(qp)
    l_pos = float(qp_run["arrival_rate"]["positive"])
    l_neg_base = float(qp_run["arrival_rate"]["negative"])
    qp_run["arrival_rate"]["negative"] = min(l_neg_base, 0.02 * l_pos)

    channels = list(range(1, qp_run["channels"]["max"] + 1))

    calc_results = []
    sim_results = []

    ch_cfg = qp_run.get("channels", {}) or {}
    num_jobs = int(ch_cfg.get("num_of_jobs") or qp_run["num_of_jobs"])
    cv_val = float(service_cv if service_cv is not None else qp_run["service"]["cv"]["base"])

    for n in channels:
        print(f"Channels: {n}")

        service_mean = n * qp_run["utilization"]["base"] / qp_run["arrival_rate"]["positive"]

        b = gamma_moments_by_mean_and_cv(service_mean, cv_val)

        calc_results.append(
            collect_calc_results(
                qp=qp_run,
                b=b,
                n=n,
                discipline=discipline,
                requeue_on_negative=requeue_on_negative,
                resume_on_negative=resume_on_negative,
                repeat_without_resampling=repeat_without_resampling,
                max_p=max_p,
            )
        )
        sim_results.append(
            collect_sim_results(
                qp={**qp_run, "num_of_jobs": num_jobs},
                b=b,
                n=n,
                discipline=discipline,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
                max_p=max_p,
            )
        )

    return DependsOnChannelsResults(
        calc=calc_results,
        sim=sim_results,
        channels=channels,
        utilization_factor=qp_run["utilization"]["base"],
        service_time_variation_coef=cv_val,
    )


def run_depends_on_varience(
    qp: dict,
    discipline: NegativeServiceType,
    requeue_on_negative: bool = False,
    resume_on_negative: bool = False,
    repeat_without_resampling: bool = False,
    rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
    max_p: int = 100,
) -> DependsOnVariationResults:
    """
    Collects simulation and calculation results for a given set of parameters,
      varying the service time variation coefficient
      and returns them as a DependsOnVariationResults object.
    """
    service_time_variation_coefs = np.linspace(
        qp["service"]["cv"]["min"],
        qp["service"]["cv"]["max"],
        qp["service"]["cv"]["num_points"],
    )
    calc_results = []
    sim_results = []

    for coef in service_time_variation_coefs:
        print(f"Service time variation coefficient: {coef:0.2f}")
        service_mean = qp["channels"]["base"] * qp["utilization"]["base"] / qp["arrival_rate"]["positive"]

        b = gamma_moments_by_mean_and_cv(service_mean, coef)
        calc_results.append(
            collect_calc_results(
                n=qp["channels"]["base"],
                qp=qp,
                b=b,
                discipline=discipline,
                requeue_on_negative=requeue_on_negative,
                resume_on_negative=resume_on_negative,
                repeat_without_resampling=repeat_without_resampling,
                max_p=max_p,
            )
        )
        sim_results.append(
            collect_sim_results(
                n=qp["channels"]["base"],
                qp=qp,
                b=b,
                discipline=discipline,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
                max_p=max_p,
            )
        )

    return DependsOnVariationResults(
        service_time_variation_coef=service_time_variation_coefs.tolist(),
        calc=calc_results,
        sim=sim_results,
        channels=qp["channels"]["base"],
        utilization_factor=qp["utilization"]["base"],
    )


def run_depends_on_utilization(
    qp: dict,
    discipline: NegativeServiceType,
    requeue_on_negative: bool = False,
    resume_on_negative: bool = False,
    repeat_without_resampling: bool = False,
    rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
    max_p: int = 100,
) -> DependsOnUtilizationResults:
    """
    Collects simulation and calculation results for a given range of utilization factors.
    """
    utilization_factors = np.linspace(
        qp["utilization"]["min"],
        qp["utilization"]["max"],
        qp["utilization"]["num_points"],
    )
    sim_results = []
    calc_results = []
    for rho in utilization_factors:
        print(f"Utilization factor: {rho:0.2f}")
        service_mean = qp["channels"]["base"] * rho / qp["arrival_rate"]["positive"]

        b = gamma_moments_by_mean_and_cv(service_mean, qp["service"]["cv"]["base"])

        calc_results.append(
            collect_calc_results(
                n=qp["channels"]["base"],
                qp=qp,
                b=b,
                discipline=discipline,
                requeue_on_negative=requeue_on_negative,
                resume_on_negative=resume_on_negative,
                repeat_without_resampling=repeat_without_resampling,
                max_p=max_p,
            )
        )
        sim_results.append(
            collect_sim_results(
                n=qp["channels"]["base"],
                qp=qp,
                b=b,
                discipline=discipline,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
                max_p=max_p,
            )
        )

    return DependsOnUtilizationResults(
        utilization_factor=utilization_factors.tolist(),
        calc=calc_results,
        sim=sim_results,
        channels=qp["channels"]["base"],
        service_time_variation_coef=qp["service"]["cv"]["base"],
    )


def run_depends_on_negative_rate(
    qp: dict,
    discipline: NegativeServiceType,
    requeue_on_negative: bool = False,
    resume_on_negative: bool = False,
    repeat_without_resampling: bool = False,
    rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
    max_p: int = 100,
    service_cv: float | None = None,
) -> DependsOnNegativeRateResults:
    """
    Collects simulation and calculation results for a given range of negative arrival rates (delta).
    """
    neg_cfg = qp.get("negative_rate", {}) or {}
    delta_min = float(neg_cfg.get("min", 0.05))
    delta_max = float(neg_cfg.get("max", float(qp["arrival_rate"]["negative"])))
    num_points = int(neg_cfg.get("num_points", 15))
    num_jobs = int(neg_cfg.get("num_of_jobs", qp["num_of_jobs"]))

    base_qp = dict(qp)
    base_qp["num_of_jobs"] = num_jobs

    calc_results = []
    sim_results = []

    service_mean = base_qp["channels"]["base"] * base_qp["utilization"]["base"] / base_qp["arrival_rate"]["positive"]
    cv_val = float(service_cv if service_cv is not None else base_qp["service"]["cv"]["base"])
    b = gamma_moments_by_mean_and_cv(service_mean, cv_val)

    if repeat_without_resampling:
        # For "repeat without resampling" (fixed service requirement per job),
        # moments (and sometimes even the mean) can explode if the MGF of B at r (or 2r)
        # does not exist. A practical safeguard for the delta-sweep is to cap delta so that:
        #   (i) E[e^{2 r B}] exists in the H2 approximation of B (second moment existence),
        #  (ii) the implied effective utilization stays away from 1.
        #
        # For H2 (mixture of exponentials) condition (i) requires 2r < min(Re(mu_i)).
        # We use a small safety margin to avoid near-singular regimes.
        try:
            h2p = H2Distribution.get_params_clx(b)
            p1 = float(np.real(complex(h2p.p1)))
            mu1 = float(np.real(complex(h2p.mu1)))
            mu2 = float(np.real(complex(h2p.mu2)))
            p1 = float(np.clip(p1, 0.0, 1.0))

            min_mu = min(mu1, mu2)
            if np.isfinite(min_mu) and min_mu > 0.0:
                # (i) moments existence cap
                delta_cap_m2 = 0.95 * (min_mu / 2.0)

                # (ii) stability/interpretability cap via rho_eff < rho_eff_max
                # Mean completion time for "repeat without resampling":
                #   E[T] = (E[e^{rB}] - 1)/r, where r=delta.
                l_pos = float(base_qp["arrival_rate"]["positive"])
                n_base = int(base_qp["channels"]["base"])
                rho_eff_max = float(neg_cfg.get("rho_eff_max", 0.95))

                mean_limit = (rho_eff_max * n_base / l_pos) if l_pos > 0.0 else float("inf")

                def mean_completion_fixed(r: float) -> float:
                    if r <= 1e-14:
                        return float(b[0])
                    # M_B(r) for H2: p1*mu1/(mu1-r) + (1-p1)*mu2/(mu2-r)
                    # Valid for r < min(mu1, mu2).
                    if r >= min_mu:
                        return float("inf")
                    m = p1 * (mu1 / (mu1 - r)) + (1.0 - p1) * (mu2 / (mu2 - r))
                    return (m - 1.0) / r

                # Find the largest r in [0, delta_cap_m2] such that mean_completion_fixed(r) <= mean_limit.
                hi = float(delta_cap_m2)
                if mean_completion_fixed(hi) <= mean_limit:
                    delta_cap_rho = hi
                else:
                    lo = 0.0
                    for _ in range(70):
                        mid = 0.5 * (lo + hi)
                        if mean_completion_fixed(mid) <= mean_limit:
                            lo = mid
                        else:
                            hi = mid
                    delta_cap_rho = float(lo)

                delta_max = min(delta_max, delta_cap_m2, delta_cap_rho)
        except Exception:
            # Fallback: keep user-provided max; the internal beff code will still guard divergence.
            pass

    if delta_max < delta_min:
        # Degenerate case: fall back to a single-point sweep.
        deltas = np.array([delta_min], dtype=float)
    else:
        deltas = np.linspace(delta_min, delta_max, num_points)

    for delta in deltas:
        print(f"Negative arrival rate: {delta:0.3f}")
        # override only the negative rate
        base_qp["arrival_rate"]["negative"] = float(delta)

        calc_results.append(
            collect_calc_results(
                n=base_qp["channels"]["base"],
                qp=base_qp,
                b=b,
                discipline=discipline,
                requeue_on_negative=requeue_on_negative,
                resume_on_negative=resume_on_negative,
                repeat_without_resampling=repeat_without_resampling,
                max_p=max_p,
            )
        )
        sim_results.append(
            collect_sim_results(
                n=base_qp["channels"]["base"],
                qp=base_qp,
                b=b,
                discipline=discipline,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
                max_p=max_p,
            )
        )

    # restore base negative rate
    base_qp["arrival_rate"]["negative"] = float(qp["arrival_rate"]["negative"])

    return DependsOnNegativeRateResults(
        negative_rate=deltas.tolist(),
        calc=calc_results,
        sim=sim_results,
        channels=qp["channels"]["base"],
        utilization_factor=qp["utilization"]["base"],
        service_time_variation_coef=cv_val,
    )


def read_parameters_from_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return the content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f_yaml:
        return yaml.safe_load(f_yaml)


# Scenarios:
# (discipline, subdir, requeue_on_negative, resume_on_negative, repeat_without_resampling, rcs_scenario, disaster_scenario)
SCENARIOS = [
    (NegativeServiceType.RCS, "rcs/remove", False, False, False, RcsScenario.REMOVE, DisasterScenario.CLEAR_SYSTEM),
    (NegativeServiceType.RCS, "rcs/requeue", True, False, False, RcsScenario.REQUEUE, DisasterScenario.CLEAR_SYSTEM),
    (
        NegativeServiceType.RCS,
        "rcs/requeue_fixed",
        True,
        False,
        True,
        RcsScenario.REQUEUE_NO_RESAMPLING,
        DisasterScenario.CLEAR_SYSTEM,
    ),
    (
        NegativeServiceType.DISASTER,
        "disaster/clear",
        False,
        False,
        False,
        RcsScenario.REMOVE,
        DisasterScenario.CLEAR_SYSTEM,
    ),
    (
        NegativeServiceType.DISASTER,
        "disaster/requeue",
        True,
        False,
        False,
        RcsScenario.REMOVE,
        DisasterScenario.REQUEUE_ALL,
    ),
    (
        NegativeServiceType.DISASTER,
        "disaster/requeue_fixed",
        True,
        False,
        True,
        RcsScenario.REMOVE,
        DisasterScenario.REQUEUE_ALL_NO_RESAMPLING,
    ),
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate negative-queues experiment results and figures.")
    parser.add_argument(
        "--depends",
        nargs="+",
        default=["all"],
        choices=["all", "channels", "coefs", "utilization", "negative_rate"],
        help="Which dependency sweeps to run (default: all).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["all"],
        help=("Which scenarios to run by subdir, e.g. 'rcs/remove' 'disaster/requeue_fixed'. " "Default: all."),
    )
    args = parser.parse_args()
    selected = set(args.depends)
    if "all" in selected:
        selected = {"channels", "coefs", "utilization", "negative_rate"}

    scen_sel = set(args.scenarios or ["all"])
    run_all_scenarios = "all" in scen_sel

    base_qp = read_parameters_from_yaml("works/negative_queues/base_parameters.yaml")
    CUR_DIR = os.path.dirname(__file__)

    # Base output: OUTPUT_DIR env or negative_queues_figures under script dir
    output_base = os.environ.get("OUTPUT_DIR")
    if not output_base:
        output_base = os.path.join(CUR_DIR, "negative_queues_figures")

    for discipline, subdir, requeue, resume, repeat_wo_resampling, rcs_scenario, disaster_scenario in SCENARIOS:
        if (not run_all_scenarios) and (subdir not in scen_sel):
            continue
        exp_dir = os.path.join(output_base, subdir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        print(f"Scenario: {subdir}")

        all_results = []
        all_xs = []
        all_depends = []
        save_paths = []

        if "channels" in selected:
            ch_cfg = base_qp.get("channels", {}) or {}
            cvs = ch_cfg.get("cv_values") or [base_qp["service"]["cv"]["base"]]
            channels_results_by_cv: list[tuple[float, DependsOnChannelsResults]] = []
            for cv_val in cvs:
                results_channels = run_depends_on_channels(
                    base_qp,
                    discipline,
                    requeue_on_negative=requeue,
                    resume_on_negative=resume,
                    repeat_without_resampling=repeat_wo_resampling,
                    rcs_scenario=rcs_scenario,
                    disaster_scenario=disaster_scenario,
                    service_cv=float(cv_val),
                )
                channels_results_by_cv.append((float(cv_val), results_channels))
                all_results.append(results_channels)
                all_xs.append(results_channels.channels)
                all_depends.append(DependsType.CHANNELS_NUMBER)
                cv_tag = str(cv_val).replace(".", "_")
                save_paths.append(os.path.join(exp_dir, f"channels_cv{cv_tag}"))

        if "coefs" in selected:
            results_coefs = run_depends_on_varience(
                base_qp,
                discipline,
                requeue_on_negative=requeue,
                resume_on_negative=resume,
                repeat_without_resampling=repeat_wo_resampling,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
            )
            all_results.append(results_coefs)
            all_xs.append(results_coefs.service_time_variation_coef)
            all_depends.append(DependsType.COEFFICIENT_OF_VARIATION)
            save_paths.append(os.path.join(exp_dir, "coefs"))

        if "utilization" in selected:
            results_utilization = run_depends_on_utilization(
                base_qp,
                discipline,
                requeue_on_negative=requeue,
                resume_on_negative=resume,
                repeat_without_resampling=repeat_wo_resampling,
                rcs_scenario=rcs_scenario,
                disaster_scenario=disaster_scenario,
            )
            all_results.append(results_utilization)
            all_xs.append(results_utilization.utilization_factor)
            all_depends.append(DependsType.UTILIZATION_FACTOR)
            save_paths.append(os.path.join(exp_dir, "utilization"))

        if "negative_rate" in selected:
            neg_cfg = base_qp.get("negative_rate", {}) or {}
            cvs = neg_cfg.get("cv_values") or [base_qp["service"]["cv"]["base"]]
            neg_results_by_cv: list[tuple[float, DependsOnNegativeRateResults]] = []
            for cv_val in cvs:
                # For fixed-service restarts the mean can explode quickly; cap the sweep range.
                qp_for_neg = base_qp
                if repeat_wo_resampling:
                    qp_for_neg = copy.deepcopy(base_qp)
                    nc = qp_for_neg.get("negative_rate", {}) or {}
                    max_old = float(nc.get("max", qp_for_neg["arrival_rate"]["negative"]))
                    # Coarse cap; an additional distribution-dependent cap is applied inside
                    # run_depends_on_negative_rate().
                    qp_for_neg["negative_rate"]["max"] = min(max_old, 0.15)
                    qp_for_neg["negative_rate"]["num_points"] = min(int(nc.get("num_points", 20)), 12)
                results_negative_rate = run_depends_on_negative_rate(
                    qp_for_neg,
                    discipline,
                    requeue_on_negative=requeue,
                    resume_on_negative=resume,
                    repeat_without_resampling=repeat_wo_resampling,
                    rcs_scenario=rcs_scenario,
                    disaster_scenario=disaster_scenario,
                    service_cv=float(cv_val),
                )
                neg_results_by_cv.append((float(cv_val), results_negative_rate))
                all_results.append(results_negative_rate)
                all_xs.append(results_negative_rate.negative_rate)
                all_depends.append(DependsType.NEGATIVE_RATE)
                cv_tag = str(cv_val).replace(".", "_")
                save_paths.append(os.path.join(exp_dir, f"negative_rate_cv{cv_tag}"))

        def _std_from_moments(m: list[float]) -> float:
            if not m or len(m) < 2:
                return 0.0
            var = float(m[1]) - float(m[0]) ** 2
            return float(np.sqrt(max(0.0, var)))

        for results, save_path, xs, depends_on in zip(all_results, save_paths, all_xs, all_depends):
            v_sim_ave = [r.v[0] for r in results.sim]
            v_calc_ave = [r.v[0] for r in results.calc]
            w_sim_ave = [r.w[0] for r in results.sim]
            w_calc_ave = [r.w[0] for r in results.calc]

            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "results.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(results), f, cls=DependsOnJSONEncoder)

            if depends_on == DependsType.NEGATIVE_RATE:
                # Extended set of figures for delta-sweeps.
                plotter_v = Plotter(xs=xs, sim_results=v_sim_ave, calc_results=v_calc_ave, depends_on=depends_on)
                plotter_v.plot_sojourn(save_path=os.path.join(save_path, "v_ave.png"))
                plotter_v.plot_errors(save_path=os.path.join(save_path, "v_ave_err.png"))

                plotter_w = Plotter(xs=xs, sim_results=w_sim_ave, calc_results=w_calc_ave, depends_on=depends_on)
                plotter_w.plot_waiting(save_path=os.path.join(save_path, "w_ave.png"))
                plotter_w.plot_errors(save_path=os.path.join(save_path, "w_ave_err.png"))

                sv_sim = [_std_from_moments(r.v) for r in results.sim]
                sv_calc = [_std_from_moments(r.v) for r in results.calc]
                Plotter(xs=xs, sim_results=sv_sim, calc_results=sv_calc, depends_on=depends_on).plot_series(
                    ylabel=r"$\sigma_{\upsilon}$", save_path=os.path.join(save_path, "sigma_v.png")
                )

                sw_sim = [_std_from_moments(r.w) for r in results.sim]
                sw_calc = [_std_from_moments(r.w) for r in results.calc]
                Plotter(xs=xs, sim_results=sw_sim, calc_results=sw_calc, depends_on=depends_on).plot_series(
                    ylabel=r"$\sigma_{\omega}$", save_path=os.path.join(save_path, "sigma_w.png")
                )

                q_sim = [float(getattr(r, "q", 0.0) or 0.0) for r in results.sim]
                q_calc = [float(getattr(r, "q", 0.0) or 0.0) for r in results.calc]
                Plotter(xs=xs, sim_results=q_sim, calc_results=q_calc, depends_on=depends_on).plot_series(
                    ylabel=r"$q$", save_path=os.path.join(save_path, "q.png")
                )
            else:
                # Legacy: only v_ave and v_ave_err
                plotter = Plotter(xs=xs, sim_results=v_sim_ave, calc_results=v_calc_ave, depends_on=depends_on)
                plotter.plot_sojourn(save_path=os.path.join(save_path, "v_ave.png"))
                plotter.plot_errors(save_path=os.path.join(save_path, "v_ave_err.png"))

    # Restore the previous "overlay" figure style used in the paper:
    # build CV-comparison plots from the stored results.json.
    try:
        if "channels" in selected:
            runpy.run_path(os.path.join(CUR_DIR, "plot_channels_cv_compare.py"), run_name="__main__")
        if "negative_rate" in selected:
            runpy.run_path(os.path.join(CUR_DIR, "plot_negative_rate_cv_compare.py"), run_name="__main__")
    except Exception as e:
        print(f"WARNING: failed to build overlay CV-compare figures: {e}")
