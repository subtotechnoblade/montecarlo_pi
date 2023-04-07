import numba as nb
import numpy as np
import multiprocessing as mp
@nb.njit(nogil=True, cache=True, parallel=True)
def mcts_pi(visits):
    wins = 0
    for _ in nb.prange(visits):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1:
            wins += 1
    return 4.0 * wins / visits

if __name__ == "__main__":
    # print(mcts_pi.parallel_diagnostics(level=4))
    calc_pi = mcts_pi(1000000000000)

    pi = 3.14159265359
    print(calc_pi)
    print(f"Deviation={calc_pi / pi - 1}")
