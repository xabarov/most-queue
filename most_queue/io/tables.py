"""
Printing tables for simulation and calculation results.
"""

import math

from colorama import Fore, Style, init

from most_queue.theory.utils.moments import convert_raw_to_central

init()


def print_raw_moments(
    sim_moments: list[float] | float,
    calc_moments: list[float] | float,
    header=None,
    sim_header="Sim",
    calc_header="Num",
):
    """
    Prints the moments of waiting or sojourn time in the system.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments (list): List of calculated moments.
        header (str): Header for the table.
        sim_header (str): Header for simulated moments.
        calc_header (str): Header for calculated moments.
    """

    if not header is None:
        print(Fore.CYAN + f"\n{header:^45}")
        print("-" * 45)

    num_col = "#"
    print(f"{num_col:^15s}|{calc_header:^15s}|{sim_header:^15s}")
    print("-" * 45)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments))):
            calc_mom = calc_moments[j].real if isinstance(calc_moments[j], complex) else calc_moments[j]
            sim_mom = sim_moments[j].real if isinstance(sim_moments[j], complex) else sim_moments[j]
            print(
                Fore.CYAN
                + f"{j + 1:^15d}|"
                + Fore.YELLOW
                + f"{calc_mom:^15.5g}"
                + Fore.CYAN
                + "|"
                + Fore.YELLOW
                + f"{sim_mom:^15.5g}"
            )
    else:
        calc_mom = calc_moments.real if isinstance(calc_moments, complex) else calc_moments
        sim_mom = sim_moments.real if isinstance(sim_moments, complex) else sim_moments
        print(
            Fore.CYAN
            + f"{1:^15d}|"
            + Fore.YELLOW
            + f"{calc_mom:^15.5g}"
            + Fore.CYAN
            + "|"
            + Fore.YELLOW
            + f"{sim_mom:^15.5g}"
        )
    print(Style.RESET_ALL)


def print_central_moments(
    sim_moments: list[float] | float,
    calc_moments: list[float] | float,
    header=None,
    sim_header="Sim",
    calc_header="Num",
):
    """
    Prints the moments of waiting or sojourn time in the system.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments (list): List of calculated moments.
        header (str): Header for the table.
        sim_header (str): Header for simulated moments.
        calc_header (str): Header for calculated moments.
    """

    if not header is None:
        print(Fore.CYAN + f"\n{header:^45}")
        print("-" * 45)

    num_col = "Moment"
    central_moments_names = ["Mean", "Variance", "Skewness", "Kurtosis"]

    print(f"{num_col:^15s}|{calc_header:^15s}|{sim_header:^15s}")
    print("-" * 45)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments))):
            calc_mom = calc_moments[j].real if isinstance(calc_moments[j], complex) else calc_moments[j]
            sim_mom = sim_moments[j].real if isinstance(sim_moments[j], complex) else sim_moments[j]
            name = central_moments_names[j]
            print(
                Fore.CYAN
                + f"{name:^15s}|"
                + Fore.YELLOW
                + f"{calc_mom:^15.5g}"
                + Fore.CYAN
                + "|"
                + Fore.YELLOW
                + f"{sim_mom:^15.5g}"
            )
    else:
        calc_mom = calc_moments.real if isinstance(calc_moments, complex) else calc_moments
        sim_mom = sim_moments.real if isinstance(sim_moments, complex) else sim_moments
        name = central_moments_names[0]
        print(
            Fore.CYAN
            + f"{name:^15s}|"
            + Fore.YELLOW
            + f"{calc_mom:^15.5g}"
            + Fore.CYAN
            + "|"
            + Fore.YELLOW
            + f"{sim_mom:^15.5g}"
        )
    print(Style.RESET_ALL)


def print_waiting_moments(
    sim_moments: list[float] | float,
    calc_moments: list[float] | float,
    sim_header="Sim",
    calc_header="Num",
    convert_to_central=True,
):
    """
    Prints the moments of waiting or sojourn time in the system.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments (list): List of calculated moments.
        sim_header (str): Header for simulated moments.
        calc_header (str): Header for calculated moments
    """

    header = "Moments of waiting time in the system"

    if convert_to_central:
        calc_moments_central = convert_raw_to_central(calc_moments)
        sim_moments_central = convert_raw_to_central(sim_moments)

        print_central_moments(
            sim_moments_central, calc_moments_central, sim_header=sim_header, calc_header=calc_header, header=header
        )
    else:
        print_raw_moments(sim_moments, calc_moments, sim_header=sim_header, calc_header=calc_header, header=header)


def print_sojourn_moments(
    sim_moments: list[float] | float,
    calc_moments: list[float] | float,
    sim_header="Sim",
    calc_header="Num",
    convert_to_central=True,
):
    """
    Prints the moments of waiting or sojourn time in the system.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments (list): List of calculated moments.
        sim_header (str): Header for simulated moments.
        calc_header (str): Header for calculated moments
    """

    header = "Moments of sojourn time in the system"

    if convert_to_central:
        calc_moments_central = convert_raw_to_central(calc_moments)
        sim_moments_central = convert_raw_to_central(sim_moments)

        print_central_moments(
            sim_moments_central, calc_moments_central, sim_header=sim_header, calc_header=calc_header, header=header
        )
    else:
        print_raw_moments(sim_moments, calc_moments, sim_header=sim_header, calc_header=calc_header, header=header)


def print_with_two_numerical(
    sim_moments: list[float] | float,
    calc_moments1: list[float] | float,
    calc_moments2: list[float] | float,
    num1_header="Num1",
    num2_header="Num2",
):
    """
    Prints the moments.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments1 (list): List of calculated moments for the first approximation.
        calc_moments2 (list): List of calculated moments for the second approximation.
        num1_header (str, optional): Header for the first numerical approximation.
        num2_header (str, optional): Header for the second numerical approximation.
    """

    num_col = "#"
    print(f"{num_col:^15s}|{num1_header:^15s}|{num2_header:^15s}|{'Sim':^15s}")
    print("-" * 60)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments1), len(calc_moments2))):
            calc_mom1 = calc_moments1[j].real if isinstance(calc_moments1[j], complex) else calc_moments1[j]
            calc_mom2 = calc_moments2[j].real if isinstance(calc_moments2[j], complex) else calc_moments2[j]
            sim_mom = sim_moments[j].real if isinstance(sim_moments[j], complex) else sim_moments[j]
            print(
                Fore.CYAN
                + f"{j + 1:^15d}|"
                + Fore.YELLOW
                + f"{calc_mom1:^15.5g}"
                + Fore.CYAN
                + "|"
                + Fore.YELLOW
                + f"{calc_mom2:^15.5g}"
                + Fore.CYAN
                + "|"
                + Fore.YELLOW
                + f"{sim_mom:^15.5g}"
            )
    else:
        calc_mom1 = calc_moments1.real if isinstance(calc_moments1, complex) else calc_moments1
        calc_mom2 = calc_moments2.real if isinstance(calc_moments2, complex) else calc_moments2

        sim_mom = sim_moments.real if isinstance(sim_moments, complex) else sim_moments
        print(
            Fore.CYAN
            + f"{1:^15d}|"
            + Fore.YELLOW
            + f"{calc_mom1:^15.5g}"
            + Fore.CYAN
            + "|"
            + Fore.YELLOW
            + f"{calc_mom2:^15.5g}"
            + Fore.CYAN
            + "|"
            + Fore.YELLOW
            + f"{sim_mom:^15.5g}"
        )
    print(Style.RESET_ALL)


def print_waiting_moments_with_two_numerical(
    sim_moments,
    calc_moments1,
    calc_moments2,
    num1_header="Num1",
    num2_header="Num2",
):
    """
    Prints the moments of waiting in the system.
     Args:
        sim_moments (list): List of simulated moments.
        calc_moments1 (list): List of calculated moments for the first approximation.
        calc_moments2 (list): List of calculated moments for the second approximation.
        num1_header (str, optional): Header for the first numerical approximation.
        num2_header (str, optional): Header for the second numerical approximation.

    """

    header = "Moments of waiting time in the system"

    print(Fore.CYAN + f"\n{header:^45}")
    print("-" * 60)
    print_with_two_numerical(
        sim_moments=sim_moments,
        calc_moments1=calc_moments1,
        calc_moments2=calc_moments2,
        num1_header=num1_header,
        num2_header=num2_header,
    )


def print_sojourn_moments_with_two_numerical(
    sim_moments,
    calc_moments1,
    calc_moments2,
    num1_header="Num1",
    num2_header="Num2",
):
    """
    Prints the  nts of sojourn time in the system.
     Args:
         sim_moments (list): List of simulated moments.
         calc_moments1 (list): List of calculated moments for the first approximation.
         calc_moments2 (list): List of calculated moments for the second approximation.
         is_w (bool, optional): If True, prints waiting time moments.
         Otherwise, prints sojourn time moments. Defaults to True.

    """

    header = "Moments of sojourn time in the system"

    print(Fore.CYAN + f"\n{header:^45}")
    print("-" * 60)
    print_with_two_numerical(
        sim_moments=sim_moments,
        calc_moments1=calc_moments1,
        calc_moments2=calc_moments2,
        num1_header=num1_header,
        num2_header=num2_header,
    )


def print_raw_moments_multiclass(sim_moments, calc_moments, convert_to_central=True):
    """
    Print moments with classes
     :param sim_moments: Simulated moments
     :param calc_moments: Calculated moments
    """

    if convert_to_central:
        sim_moments = [convert_raw_to_central(mom) for mom in sim_moments]
        calc_moments = [convert_raw_to_central(mom) for mom in calc_moments]
        blank_col, header_col, cls_col = "", "Moment", "Cls"
    else:
        blank_col, header_col, cls_col = "", "Number of moment", "Cls"

    central_moments_names = ["Mean", "Variance", "Skewness", "Kurtosis"]

    k_num = len(sim_moments)
    size = min(len(sim_moments[0]), len(calc_moments[0]))

    num_col, sim_col = "Num", "Sim"

    print("-" * 60)
    print(f"{blank_col:^11}|{header_col:^47}|")
    print(f"{cls_col:^11}| ", end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(size):
        if convert_to_central:
            s = central_moments_names[j]
        else:
            s = str(j + 1)
        print(f"{s:^15}|", end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + f"{sim_col:^5}|", end="")
        for j in range(size):
            print(Fore.YELLOW + f"{sim_moments[i][j]:^15.3g}" + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + f"{str(i + 1):^5}" + "|" + "-" * 54)

        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + f"{num_col:^5}|", end="")
        for j in range(size):
            print(Fore.YELLOW + f"{calc_moments[i][j]:^15.3g}" + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "-" * 60)

    print("\n")

    print(Style.RESET_ALL)


def print_waiting_multiclass(sim_moments, calc_moments, convert_to_central=True):
    """
    Print waiting moments with classes
     :param sim_moments: Simulated moments
     :param calc_moments: Calculated moments
    """
    header = "Moments of waiting time in the system"

    print(Fore.CYAN + f"{header:^60s}")

    print_raw_moments_multiclass(sim_moments, calc_moments, convert_to_central)


def print_sojourn_multiclass(sim_moments, calc_moments, convert_to_central=True):
    """
    Print sojourn moments with classes
     :param sim_moments: Simulated moments
     :param calc_moments: Calculated moments
    """
    header = "Moments of sojourn time in the system"

    print(Fore.CYAN + f"{header:^60s}")

    print_raw_moments_multiclass(sim_moments, calc_moments, convert_to_central)


def probs_print(p_sim, p_num, size=10):
    """
    Prints the probabilities of states.
    :param p_sim: Simulated probabilities
    :param p_num: Calculated probabilities
    :param size: Number of states to print
    :return: None
    """
    header = "Probabilities of states"
    print(Fore.CYAN + "-" * 36)
    print(f"{header:^36s}")
    print("-" * 36)
    sharp_col, num_col, sim_col = "#", "Num", "Sim"
    print(f"{sharp_col:^4s}|{num_col:^15s}|{sim_col:^15s}")
    print("-" * 36)
    size = min(len(p_sim), len(p_num), size)
    for i in range(size):
        print(
            Fore.CYAN
            + f"{i:^4d}|"
            + Fore.YELLOW
            + f"{p_num[i]:^15.5g}"
            + Fore.CYAN
            + "|"
            + Fore.YELLOW
            + f"{p_sim[i]:^15.5g}"
        )
    print(Fore.CYAN + "-" * 36)

    print(Style.RESET_ALL)


def probs_print_no_compare(probs, size=10, header="Probabilities of states"):
    """
    Print table with probabilities
     :param probs: Probabilities
     :param size: Number of states to print
      :return: None
    """
    print(Fore.CYAN + "-" * 36)
    print(f"{header:^36s}")
    print("-" * 36)
    sharp_col, prob_col = "#", "Probability"
    print(f"{sharp_col:^4s}|{prob_col:^15s}")
    print("-" * 36)

    size = min(len(probs), size)
    for i in range(size):
        print(Fore.CYAN + f"{i:^4d}|" + Fore.YELLOW + f"{probs[i]:^15.5g}" + Fore.CYAN)
    print(Fore.CYAN + "-" * 36)

    print(Style.RESET_ALL)


def print_mrx(mrx, is_short=False):
    """
    Print matrix mrx
    """
    row = mrx.shape[0]
    col = mrx.shape[1]

    print(f"{Fore.GREEN}\n")

    for i in range(row):
        print("| ", end="")

        for j in range(col):
            if math.isclose(mrx[i, j].real, 0.0):
                color = Fore.RED
            else:
                color = Fore.GREEN
            if is_short or mrx[i, j].imag == 0.0:
                print(f"{color}{mrx[i, j].real:^4.2f} | ", end="")
            else:
                print(f"{color}{mrx[i, j]:^5.3f} | ", end="")
        print("\n" + "-------" * col if is_short or mrx[i, j].imag == 0.0 else "\n" + "---------------" * col)

    print(f"{Style.RESET_ALL}")
