from colorama import init
from colorama import Fore, Back, Style

init()


def times_print(sim_moments, calc_moments, is_w=True):
    if is_w:
        spec = 'ожидания'
    else:
        spec = 'пребывания'

    header = f'\rЗначения начальных моментов времени {spec} заявок в системе:'

    print(Fore.CYAN + f'\r{header}')

    print(Fore.BLUE + "{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^15d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, calc_moments[j], sim_moments[j]))

    print(Style.RESET_ALL)


def probs_print(p_sim, p_ch, size=10):
    print(Fore.CYAN + "-" * 36)
    print("{0:^36s}".format("Вероятности состояний"))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 36)
    for i in range(size):
        print(Fore.BLUE + "{0:^4d}|{1:^15.5g}|{2:^15.5g}".format(i, p_ch[i], p_sim[i]))
    print(Fore.CYAN + "-" * 36)

    print(Style.RESET_ALL)