import sim.rand_destribution as rd
import numpy as np


def get_min_max(value, min_value, max_value=None):
    if max_value:
        return min(max(min_value, value), max_value)

    return max(min_value, value)


def get_samples(moments, num=1000, is_cummulative=True, memory_factor=0.5, min_value=0, max_value=None):
    a_k = rd.Pareto_dist.get_a_k(moments)
    pa = rd.Pareto_dist(a_k)
    samples = []
    for i in range(num):
        if is_cummulative:
            if i != 0:
                p = np.random.random()
                if p < 0.5:
                    value = samples[i - 1] - memory_factor * pa.generate()
                    value = get_min_max(value, min_value, max_value)
                else:
                    value = samples[i - 1] + memory_factor * pa.generate()
                    value = get_min_max(value, min_value, max_value)
                samples.append(value)
            else:
                value = get_min_max(pa.generate(), min_value, max_value)
                samples.append(value)
        else:
            value = get_min_max(pa.generate(), min_value, max_value)
            samples.append(value)

    return samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    import matplotlib
    matplotlib.use('TkAgg')

    num = 1000
    moments = [1, 100]

    x = [x + 1 for x in range(num)]
    y = get_samples(moments, num, is_cummulative=False, min_value=0, max_value=30, memory_factor=0.95)
    fig, ax = plt.subplots()

    ax.plot(x, y, label="Samples", c='green')
    ax.set_xlabel('t')
    ax.set_ylabel('load')
    plt.legend()
    plt.show()
