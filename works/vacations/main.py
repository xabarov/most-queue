import os

from channels import run_channels
from cooling import run_cool_ave, run_cool_cv
from cooling_delay import run_cool_delay_average, run_cool_delay_cv
from service import run_service_cv
from utilization import run_utilization
from utils import (
    create_new_experiment_dir,
    read_parameters_from_yaml,
    save_parameters_as_yaml,
)
from warmup import run_warmup_ave, run_warmup_cv


def run_all(qp: dict):
    """
    Run all experiments  based on the given queue parameters.
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(cur_dir, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_path = create_new_experiment_dir(results_folder)
    save_parameters_as_yaml(qp, results_path)

    run_channels(qp, save_path=results_path)
    run_service_cv(qp, save_path=results_path)
    run_utilization(qp, save_path=results_path)

    cool_delay_path = os.path.join(results_path, 'cooling_delay')
    if not os.path.exists(cool_delay_path):
        os.makedirs(cool_delay_path)

    run_cool_delay_average(qp, save_path=cool_delay_path)
    run_cool_delay_cv(qp, save_path=cool_delay_path)

    cooling_path = os.path.join(results_path, 'cooling')
    if not os.path.exists(cooling_path):
        os.makedirs(cooling_path)

    run_cool_ave(qp, save_path=cooling_path)
    run_cool_cv(qp, save_path=cooling_path)

    warmup_path = os.path.join(results_path, 'warmup')
    if not os.path.exists(warmup_path):
        os.makedirs(warmup_path)

    run_warmup_ave(qp, save_path=warmup_path)
    run_warmup_cv(qp, save_path=warmup_path)


if __name__ == "__main__":

    # read parameters from yaml file with path base_parameters.yaml

    base_qp = read_parameters_from_yaml('works/vacations/base_parameters.yaml')

    run_all(base_qp)
