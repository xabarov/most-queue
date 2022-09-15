from most_queue.sim import smo_im
from most_queue.sim import rand_destribution as rd
from most_queue.theory.ek_d_n_calc import Ek_D_n


def test():
    ro = 0.8  # коэффициент загрузки
    a1 = 1  # среднее время между заявками вх потока
    n = 4  # число каналов
    coev_a = 0.56  # коэффициент вариации вх потока
    num_of_jobs = 800000  # количество заявок для ИМ

    k, l = rd.Erlang_dist.get_params_by_mean_and_coev(a1, coev_a)
    b = a1 * n * ro
    ekdn = Ek_D_n(l, k, b, n)
    p_ch = ekdn.calc_p()

    smo = smo_im.SmoIm(n)
    smo.set_sources([k, l], "E")
    smo.set_servers(b, "D")
    smo.run(num_of_jobs)
    v_im = smo.v
    p_im = smo.get_p()

    print("-" * 36)
    print("{0:^36s}".format("Вероятности состояний СМО E{0:d}/D/{1:d}".format(k, n)))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 36)
    for i in range(11):
        print("{0:^4d}|{1:^15.5g}|{2:^15.5g}".format(i, p_ch[i], p_im[i]))
    print("-" * 36)

if __name__=="__main__":
    test()
