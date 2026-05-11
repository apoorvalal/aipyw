import numpy as np

from aipyw import AIPyW


def test_balancing_riesz_is_zero_off_arm_and_normalized_on_arm():
    rng = np.random.default_rng(123)
    n = 300
    x = rng.normal(size=(n, 4))
    logits = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    p = 1 / (1 + np.exp(-logits))
    w = rng.binomial(1, p)
    y = 1 + w + x[:, 0] + rng.normal(size=n)

    model = AIPyW(riesz_method="balancing", bal_obj="quadratic")
    model.fit(x, w, y)

    for arm in range(model.K):
        alpha = model.a_x[:, arm]
        mask = w == arm
        assert np.allclose(alpha[~mask], 0.0)
        assert np.isclose(alpha[mask].sum(), 1.0, atol=1e-7)
