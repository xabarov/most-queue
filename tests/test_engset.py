"""
Test for Engset model (M/M/1 with a finite number of sources)
"""
from most_queue.general.tables import probs_print_no_compare, times_print_no_compare
from most_queue.theory.closed.engset_model import Engset


def test_engset():
    """
    Test for Engset model (M/M/1 with a finite number of sources)
    """
    
    lam = 0.3
    mu = 1.0
    m = 7

    engset = Engset(lam, mu, m)

    ps = engset.get_p()

    probs_print_no_compare(ps)

    N = engset.get_N()
    Q = engset.get_Q()
    kg = engset.get_kg()

    print(f'N = {N:3.3f}, Q = {Q:3.3f}, kg = {kg:3.3f}')

    w1 = engset.get_w1()
    v1 = engset.get_v1()
    w = engset.get_w()
    v = engset.get_v()

    print(f'v1 = {v1:3.3f}, w1 = {w1:3.3f}')

    times_print_no_compare(wait_times=w, soujourn_times=v)
    
if __name__ == "__main__":
    
    test_engset()