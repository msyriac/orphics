import numpy as np
import pytest
from mpi4py import MPI

from orphics.stats import Statistics


def ranksize(comm: MPI.Comm) -> tuple[int, int]:
    return comm.Get_rank(), comm.Get_size()


def test_scalar_mean_with_closed_form(comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Each rank r contributes the integers 1..(r+1) as 1-D samples.
    Global N = sum_{r=0}^{P-1} (r+1) = P(P+1)/2
    Global SUM = sum_{r=0}^{P-1} (r+1)(r+2)/2
    Global MEAN = SUM / N
    """
    rank, P = ranksize(comm)

    m_r = rank + 1
    # Build a vector of length-1 samples [1], [2], ..., [m_r]
    X = np.arange(1, m_r + 1, dtype=np.float64).reshape(m_r, 1)

    acc = Statistics(comm=comm, dtype=np.float64)
    acc.extend("A", X)
    acc.allreduce()

    # Closed-form totals
    N = P * (P + 1) // 2  # ∑ (r+1)
    SUM = 0
    for r in range(P):
        m = r + 1
        SUM += m * (m + 1) // 2  # ∑_{k=1..m} k = m(m+1)/2
    MEAN = SUM / N

    # Check mean (note: our accumulator returns shape (1,))
    np.testing.assert_allclose(acc.mean("A")[0], MEAN, rtol=0, atol=0)



def test_covariance_from_outer_sums_closed_form(comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Stats label with 2-D vectors. Each rank r contributes m_r = r+1 copies of x_r = [r, 2r].
    Let:
      N   = ∑ m_r = P(P+1)/2
      S1  = ∑ m_r * r = ∑ r(r+1) = ∑(r^2 + r)
      T   = ∑ m_r * r^2 = ∑ r^2(r+1) = ∑(r^3 + r^2)

    Sum vector S = [S1, 2*S1]
    Cross sum C  = ∑ m_r * (x_r x_r^T)
                 = [[T, 2T],
                    [2T, 4T]]

    Centered second moment: M = C - (1/N) S S^T
                           = (T - S1^2/N) * [[1, 2],
                                            [2, 4]]

    Sample covariance (ddof=1): Cov = M / (N - 1)
    """
    rank, P = ranksize(comm)
    m_r = rank + 1

    x_r = np.array([float(rank), 2.0 * float(rank)], dtype=np.float64)
    X = np.tile(x_r, (m_r, 1))

    acc = Statistics(comm=comm, dtype=np.float64)
    acc.extend("C", X)
    acc.allreduce()

    # Closed-form helpers over r = 0..P-1
    def sum_r(P):
        return (P - 1) * P // 2

    def sum_r2(P):
        return (P - 1) * P * (2 * P - 1) // 6

    def sum_r3(P):
        s = sum_r(P)
        return s * s  # (∑ r)^2 for r starting at 0

    N = P * (P + 1) // 2                    # ∑ m_r
    S1 = sum_r2(P) + sum_r(P)               # ∑ r(r+1)
    T = sum_r3(P) + sum_r2(P)               # ∑ r^2(r+1) = ∑(r^3 + r^2)

    # M scale factor:
    scale = float(T - (S1 * S1) / N)
    M = scale * np.array([[1.0, 2.0],
                          [2.0, 4.0]], dtype=np.float64)

    if N > 1:
        Cov_expected = M / (N - 1)
    else:
        Cov_expected = np.full((2, 2), np.nan, dtype=np.float64)

    np.testing.assert_allclose(acc.cov("C", ddof=1), Cov_expected, rtol=0, atol=0)


def test_label_present_on_subset_of_ranks_closed_form(comm: MPI.Comm = MPI.COMM_WORLD):
    """
    'train' present on all ranks: each rank r adds m_r=r+1 copies of [1].
      N = ∑ m_r = P(P+1)/2
      SUM = ∑ m_r * 1 = N
      MEAN = 1

    'valid' present only on even ranks: r even adds (r+1) copies of [2].
      N_valid = ∑_{r even} (r+1)
              = sum over k=0..⌊(P-1)/2⌋ of (2k+1)
              = number_of_even_ranks^2
      SUM_valid = 2 * N_valid
      MEAN_valid = 2
    """
    rank, P = ranksize(comm)

    # train: all ranks add 1 repeated (r+1) times
    m_r = rank + 1
    acc = Statistics(comm=comm, dtype=np.float64)
    acc.extend("train", np.ones((m_r, 1), dtype=np.float64))

    # valid: only even ranks add 2 repeated (r+1) times
    if rank % 2 == 0:
        acc.extend("valid", 2.0 * np.ones((m_r, 1), dtype=np.float64))

    acc.allreduce()

    # train expectations
    N_train = P * (P + 1) // 2
    mean_train = 1.0
    np.testing.assert_allclose(acc.mean("train")[0], mean_train, rtol=0, atol=0)

    # valid expectations
    # number of even ranks in 0..P-1 is E = ceil(P/2)
    E = (P + 1) // 2
    N_valid = E * E  # ∑_{k=0}^{E-1} (2k+1) = E^2
    if N_valid > 0:
        mean_valid = 2.0
        np.testing.assert_allclose(acc.mean("valid")[0], mean_valid, rtol=0, atol=0)


def test_stack_2d_array():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base = np.arange(6, dtype=np.float64).reshape(2, 3)  # [[0,1,2],[3,4,5]]
    arr = (rank + 1) * base

    stats = Statistics(comm=comm, dtype=np.float64)
    stats.add_stack("A", arr)
    stats.allreduce()

    # Analytic expectations
    factor = size * (size + 1) / 2.0
    expected_sum = factor * base
    expected_count = size

    got_sum = stats.stack_sum("A")
    got_count = stats.stack_count("A")

    # Use allclose for safety (float)
    assert np.allclose(got_sum, expected_sum)
    assert got_count == expected_count
        
