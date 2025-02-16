import numpy as np


def lognormal_pdf(x, mu, sigma):
    term1 = 1 / (x * sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
    return term1 * term2


def calc_nu_dist(nu, mu, sigma):
    return lognormal_pdf(nu, mu, sigma)
