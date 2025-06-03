"""
Printing tables for simulation and calculation results.
"""
import math

from colorama import Fore, Style, init

init()


def times_print(sim_moments, calc_moments, is_w=True, header=None,
                sim_header='Sim', calc_header='Num'):
    """
    Prints the initial moments of waiting or sojourn time in the system.
     Args:
         sim_moments (list): List of simulated moments.
         calc_moments (list): List of calculated moments.
         is_w (bool, optional): If True, prints waiting time moments. 
         Otherwise, prints sojourn time moments. Defaults to True.

    """

    if header is None:
        if is_w:
            spec = 'waiting'
        else:
            spec = 'soujorn'
        header = f'Initial moments of {spec} time in the system'

    print(Fore.CYAN + f'\n{header:^45}')
    print("-" * 45)
    num_col = "#"
    print(f"{num_col:^15s}|{calc_header:^15s}|{sim_header:^15s}")
    print("-" * 45)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments))):
            calc_mom = calc_moments[j].real if isinstance(
                calc_moments[j], complex) else calc_moments[j]
            sim_mom = sim_moments[j].real if isinstance(
                sim_moments[j], complex) else sim_moments[j]
            print(
                Fore.CYAN + f"{j + 1:^15d}|" +
                Fore.YELLOW + f"{calc_mom:^15.5g}" +
                Fore.CYAN + "|" +
                Fore.YELLOW + f"{sim_mom:^15.5g}")
    else:
        calc_mom = calc_moments.real if isinstance(
            calc_moments, complex) else calc_moments
        sim_mom = sim_moments.real if isinstance(
            sim_moments, complex) else sim_moments
        print(
            Fore.CYAN + f"{1:^15d}|" +
            Fore.YELLOW + f"{calc_mom:^15.5g}" +
            Fore.CYAN + "|" +
            Fore.YELLOW + f"{sim_mom:^15.5g}")
    print(Style.RESET_ALL)


def times_print_with_two_numerical(sim_moments, calc_moments1, calc_moments2, is_w=True, 
                                   num1_header='Num1', num2_header='Num2'):
    """
    Prints the initial moments of waiting or sojourn time in the system.
     Args:
         sim_moments (list): List of simulated moments.
         calc_moments1 (list): List of calculated moments for the first approximation.
         calc_moments2 (list): List of calculated moments for the second approximation.
         is_w (bool, optional): If True, prints waiting time moments. 
         Otherwise, prints sojourn time moments. Defaults to True.

    """

    if is_w:
        spec = 'waiting'
    else:
        spec = 'soujorn'
    header = f'Initial moments of {spec} time in the system'

    print(Fore.CYAN + f'\n{header:^45}')
    print("-" * 60)
    num_col = "#"
    print(f"{num_col:^15s}|{num1_header:^15s}|{num2_header:^15s}|{'Sim':^15s}")
    print("-" * 60)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments1), len(calc_moments2))):
            calc_mom1 = calc_moments1[j].real if isinstance(
                calc_moments1[j], complex) else calc_moments1[j]
            calc_mom2 = calc_moments2[j].real if isinstance(
                calc_moments2[j], complex) else calc_moments2[j]
            sim_mom = sim_moments[j].real if isinstance(
                sim_moments[j], complex) else sim_moments[j]
            print(
                Fore.CYAN + f"{j + 1:^15d}|" +
                Fore.YELLOW + f"{calc_mom1:^15.5g}" +
                Fore.CYAN + "|" +
                Fore.YELLOW + f"{calc_mom2:^15.5g}" +
                Fore.CYAN + "|" +
                Fore.YELLOW + f"{sim_mom:^15.5g}")
    else:
        calc_mom1 = calc_moments1.real if isinstance(
            calc_moments1, complex) else calc_moments1
        calc_mom2 = calc_moments2.real if isinstance(
            calc_moments2, complex) else calc_moments2

        sim_mom = sim_moments.real if isinstance(
            sim_moments, complex) else sim_moments
        print(
            Fore.CYAN + f"{1:^15d}|" +
            Fore.YELLOW + f"{calc_mom1:^15.5g}" +
            Fore.CYAN + "|" +
            Fore.YELLOW + f"{calc_mom2:^15.5g}" +
            Fore.CYAN + "|" +
            Fore.YELLOW + f"{sim_mom:^15.5g}")
    print(Style.RESET_ALL)


def times_print_no_compare(wait_times=None, sojourn_times=None):
    """
    Prints the wait and sojourn times.
    :param wait_times: Wait times
    :param sojourn_times: Sojourn times
    :return: None
    """

    if sojourn_times is None and wait_times is None:
        raise ValueError(
            "Either wait_times or sojourn_times must be provided.")

    if sojourn_times is not None and wait_times is not None:

        print(Fore.CYAN + 'Initial moments of sojourn and wait times in the system')

        num_col, w_col, v_col = "#", "w", "v"
        print(f"{num_col:^15s}|{w_col:^15s}|{v_col:^15s}")
        print("-" * 45)
        if isinstance(wait_times, list):
            for j in range(min(len(wait_times), len(sojourn_times))):
                w_mom = wait_times[j].real if isinstance(
                    wait_times[j], complex) else wait_times[j]
                v_mom = sojourn_times[j].real if isinstance(
                    sojourn_times[j], complex) else sojourn_times[j]
                print(
                    Fore.CYAN + f"{j + 1:^15d}|" +
                    Fore.YELLOW + f"{w_mom:^15.5g}" +
                    Fore.CYAN + "|" + Fore.YELLOW +
                    f"{v_mom:^15.5g}")
        else:
            w_mom = wait_times.real if isinstance(
                wait_times, complex) else wait_times
            v_mom = sojourn_times.real if isinstance(
                sojourn_times, complex) else sojourn_times
            print(
                Fore.CYAN + f"{1:^15d}|" +
                Fore.YELLOW + f"{w_mom:^15.5g}" +
                Fore.CYAN + "|" + Fore.YELLOW +
                f"{v_mom:^15.5g}")

        print(Style.RESET_ALL)

    else:
        times = wait_times if wait_times is not None else sojourn_times
        spec = 'wait' if wait_times is not None else 'sojourn'
        times_header = 'w' if wait_times is not None else 'v'
        header = f'Initial moments of {spec} time in the system'

        print(header)

        num_col = "#"
        print(f"{num_col:^15s}|{times_header:^15s}")
        print("-" * 30)
        if isinstance(times, list):
            for j, mom in enumerate(times):
                mom = mom.real if isinstance(
                    mom, complex) else mom
                print(
                    Fore.CYAN + f"{j + 1:^15d}|" +
                    Fore.YELLOW + f"{mom:^15.5g}" +
                    Fore.CYAN)
        else:
            mom = times.real if isinstance(
                times, complex) else times

            print(
                Fore.CYAN + f"{1:^15d}|" +
                Fore.YELLOW + f"{mom:^15.5g}" +
                Fore.CYAN)

        print(Style.RESET_ALL)


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
    sharp_col, num_col, sim_col = '#', 'Num', 'Sim'
    print(f"{sharp_col:^4s}|{num_col:^15s}|{sim_col:^15s}")
    print("-" * 36)
    size = min(len(p_sim), len(p_num), size)
    for i in range(size):
        print(
            Fore.CYAN + f"{i:^4d}|" + Fore.YELLOW +
            f"{p_num[i]:^15.5g}" + Fore.CYAN + "|" +
            Fore.YELLOW + f"{p_sim[i]:^15.5g}")
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
    sharp_col, prob_col = '#', "Probability"
    print(f"{sharp_col:^4s}|{prob_col:^15s}")
    print("-" * 36)

    size = min(len(probs), size)
    for i in range(size):
        print(
            Fore.CYAN + f"{i:^4d}|" + Fore.YELLOW +
            f"{probs[i]:^15.5g}" + Fore.CYAN)
    print(Fore.CYAN + "-" * 36)

    print(Style.RESET_ALL)


def times_print_with_classes(sim_moments, calc_moments, is_w=True):
    """
    Print moments with classes
     :param sim_moments: Simulated moments
     :param calc_moments: Calculated moments
     :param is_w: If True, print waiting time, else print sojourn time
      :return: None

    """
    if is_w:
        spec = 'waiting'
    else:
        spec = 'soujorn'

    header = f'Initial moments of {spec} time in the system'

    k_num = len(sim_moments)
    size = len(sim_moments[0])

    print(Fore.CYAN + f'{header:^60s}')

    blank_col, header_col, cls_col = '', 'Number of moment', 'Cls'
    num_col, sim_col = 'Num', 'Sim'

    print("-" * 60)
    print(f"{blank_col:^11}|{header_col:^47}|")
    print(f"{cls_col:^11}| ", end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(size):
        s = str(j + 1)
        print(f"{s:^15}|", end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + f"{sim_col:^5}|", end="")
        for j in range(size):
            print(
                Fore.YELLOW + F"{sim_moments[i][j]:^15.3g}" +
                Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + F"{str(i + 1):^5}" + "|" + "-" * 54)

        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + F"{num_col:^5}|", end="")
        for j in range(size):
            print(
                Fore.YELLOW + f"{calc_moments[i][j]:^15.3g}" +
                Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "-" * 60)

    print("\n")

    print(Style.RESET_ALL)


def print_mrx(mrx, is_short=False):
    """
    Print matrix mrx
    """
    row = mrx.shape[0]
    col = mrx.shape[1]

    print(f'{Fore.GREEN}\n')

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
        print("\n" + "-------" *
              col if is_short or mrx[i, j].imag == 0.0 else "\n" + "---------------" * col)

    print(f'{Style.RESET_ALL}')