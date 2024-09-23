from colorama import init
from colorama import Fore, Back, Style

init()


def times_print(sim_moments, calc_moments, is_w=True):
    if is_w:
        spec = 'ожидания'
    else:
        spec = 'пребывания'

    header = f'\rНачальные моменты времени {spec} заявок в системе:'

    print(Fore.CYAN + f'\r{header}')

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(min(len(sim_moments), len(calc_moments))):
        print(
            Fore.CYAN + f"{j + 1:^15d}|" + Fore.YELLOW  + f"{calc_moments[j]:^15.5g}" + Fore.CYAN + "|" + Fore.YELLOW + f"{sim_moments[j]:^15.5g}")

    print(Style.RESET_ALL)


def probs_print(p_sim, p_ch, size=10):
    print(Fore.CYAN + "-" * 36)
    print("{0:^36s}".format("Вероятности состояний"))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 36)
    size = min(len(p_sim), len(p_ch), size)
    for i in range(size):
        print(
            Fore.CYAN + f"{i:^4d}|" + Fore.YELLOW + f"{p_ch[i]:^15.5g}" + Fore.CYAN + "|" + Fore.YELLOW + f"{p_sim[i]:^15.5g}")
    print(Fore.CYAN + "-" * 36)

    print(Style.RESET_ALL)


def times_print_with_classes(sim_moments, calc_moments, is_w=True):
    if is_w:
        spec = 'ожидания'
    else:
        spec = 'пребывания'

    header = f'Начальные моменты времени {spec} заявок в системе'
    k_num = len(sim_moments)
    size = len(sim_moments[0])

    print(Fore.CYAN + f'{header:^60s}')

    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    print("{0:^11s}| ".format('№ кл'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(size):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + "{:^5s}|".format("ИМ"), end="")
        for j in range(size):
            print(Fore.YELLOW + "{:^15.3g}".format(sim_moments[i][j]) + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(Fore.CYAN + " " * 5 + "|", end="")
        print(Fore.CYAN + "{:^5s}|".format("Р"), end="")
        for j in range(size):
            print(Fore.YELLOW + "{:^15.3g}".format(calc_moments[i][j]) + Fore.CYAN + "|", end="")
        print("")
        print(Fore.CYAN + "-" * 60)

    print("\n")

    print(Style.RESET_ALL)
