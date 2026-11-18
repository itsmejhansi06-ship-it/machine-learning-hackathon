
import math


def gaussian_emission_probability(distance_m, sigma=30.0):
    """
    Calculate the Gaussian emission probability.

    distance_m: Distance from GPS observation to candidate road (metres).
    sigma: Standard deviation of GPS measurement error (metres).
    """

    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    if not math.isfinite(distance_m) or distance_m < 0:
        return 1e-12

    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))

    exponent = -(distance_m ** 2) / (2 * sigma ** 2)

    probability = coefficient * math.exp(exponent)

    return max(probability, 1e-12)


def log_emission_probability(distance_m, sigma=30.0):
    """Return log Gaussian emission probability."""

    probability = gaussian_emission_probability(distance_m, sigma)

    return math.log(probability)
