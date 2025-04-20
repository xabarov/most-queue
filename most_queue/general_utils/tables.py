from colorama import Fore, Style, init

init()


def times_print(sim_moments, calc_moments, is_w=True):
    """
    Prints the initial moments of waiting or sojourn time in the system.
     Args:
         sim_moments (list): List of simulated moments.
         calc_moments (list): List of calculated moments.
         is_w (bool, optional): If True, prints waiting time moments. Otherwise, prints sojourn time moments. Defaults to True.

    """
    if is_w:
        spec = 'waiting'
    else:
        spec = 'soujorn'

    header = f'Initial moments of {spec} time in the system'

    print(Fore.CYAN + f'\r{header}')

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("#", "Num", "Sim"))
    print("-" * 45)
    if isinstance(sim_moments, list):
        for j in range(min(len(sim_moments), len(calc_moments))):
            calc_mom = calc_moments[j].real if isinstance(
                calc_moments[j], complex) else calc_moments[j]
            sim_mom = sim_moments[j].real if isinstance(
                sim_moments[j], complex) else sim_moments[j]
            print(
                Fore.CYAN + f"{j + 1:^15d}|" + Fore.YELLOW + f"{calc_mom:^15.5g}" + Fore.CYAN + "|" + Fore.YELLOW + f"{sim_mom:^15.5g}")
    else:
        calc_mom = calc_moments.real if isinstance(
            calc_moments, complex) else calc_moments
        sim_mom = sim_moments.real if isinstance(
            sim_moments, complex) else sim_moments
        print(
            Fore.CYAN + f"{1:^15d}|" + Fore.YELLOW + f"{calc_mom:^15.5g}" + Fore.CYAN + "|" + Fore.YELLOW + f"{sim_mom:^15.5g}")

    print(Style.RESET_ALL)


def probs_print(p_sim, p_ch, size=10):
    """
    Prints the probabilities of states.
    :param p_sim: Simulated probabilities
    :param p_ch: Calculated probabilities
    :param size: Number of states to print
    :return: None
    """
    print(Fore.CYAN + "-" * 36)
    print("{0:^36s}".format("Probabilities of states"))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("#", "Num", "Sim"))
    print("-" * 36)
    size = min(len(p_sim), len(p_ch), size)
    for i in range(size):
        print(
            Fore.CYAN + f"{i:^4d}|" + Fore.YELLOW + f"{p_ch[i]:^15.5g}" + Fore.CYAN + "|" + Fore.YELLOW + f"{p_sim[i]:^15.5g}")
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

    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Number of moment'))
    print("{0:^11s}| ".format('Cls'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(size):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + "{:^5s}|".format("Sim"), end="")
        for j in range(size):
            print(
                Fore.YELLOW + "{:^15.3g}".format(sim_moments[i][j]) + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + "{:^5s}|".format("Num"), end="")
        for j in range(size):
            print(
                Fore.YELLOW + "{:^15.3g}".format(calc_moments[i][j]) + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "-" * 60)

    print("\n")

    print(Style.RESET_ALL)
