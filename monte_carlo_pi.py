import numba as nb
import numpy as np
import multiprocessing as mp
@nb.njit(nogil=True, cache=True, parallel=True)
def mcts_pi(iteration_limit):
    hits = 0
    for _ in nb.prange(iteration_limit):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1:
            hits += 1
    return 4.0 * hits / iteration_limit

if __name__ == "__main__":
    # print(mcts_pi.parallel_diagnostics(level=4))
    calc_pi = mcts_pi(1000000000000)

    pi = 3.14159265359
    print(calc_pi)
    print(f"Deviation={calc_pi / pi - 1}")
