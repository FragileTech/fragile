"""Outward certificate for native realized-array fitness and signed pressure."""

import mpmath as mp


def main():
    mp.mp.dps = 100
    mp.iv.dps = 85
    iv = mp.iv
    interval = iv.mpf

    def lower(value):
        return mp.mpf(value._mpi_[0])

    def upper(value):
        return mp.mpf(value._mpi_[1])

    # For 0<=r<=1.5 the alternating-free derivative proof of
    # tanh(x)>=x-x**3/3 yields remainder/r**2<=r/24<=.0625.
    # For r>=10 use 0<=tanh(r/2), yielding remainder/r**2<=1/(2r)<=.05.
    # On the intervening compact interval both endpoints are directed.
    maximum = mp.mpf(0)
    for j in range(170):
        left = interval("1.5") + interval(".05") * j
        right = left + interval(".05")
        exponential = iv.exp(left)
        tanh_left = (exponential - 1) / (exponential + 1)
        bound = (right / 2 - tanh_left) / left**2
        maximum = max(maximum, upper(bound))
    print("compact_negative_tanh_remainder_upper", mp.nstr(maximum, 30))
    assert maximum < mp.mpf(".07")
    mean = interval("1.1") ** 2 + interval("2.2") * interval(".07") + interval(".25")
    assert upper(mean) <= mp.mpf("1.614") + mp.mpf("1e-80")
    excess = (
        interval("2.2") * interval(".89") - mean * interval(1000) / interval(999) / interval(".89")
    ) / (interval("2.2") + interval(".000001"))
    print("native_mean_fitness_bound", mp.nstr(upper(mean), 30))
    print("finite_source_excess_lower", mp.nstr(lower(excess), 30))
    assert lower(excess) > mp.mpf(".0648")
    print("All native logistic-fitness and signed-pressure margins certified.")


if __name__ == "__main__":
    main()
