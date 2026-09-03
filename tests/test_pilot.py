from fxmoment.pilot import sample_size_per_arm, weeks_to_power


def test_sample_size_reasonable():
    n = sample_size_per_arm(0.10, 0.02)
    assert 3000 < n < 4000
    assert weeks_to_power(10_000, 1.5, 0.9, 0.10, 0.02) < 2
