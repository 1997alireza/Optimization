def norm_p(x, p=2):
    assert p >= 1
    return sum(x**p) ** (1/p)
