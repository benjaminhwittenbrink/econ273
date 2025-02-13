"""
Title: Problem Set 1 -- simulate.py
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

import numpy as np


def simulate(params):
    """
    Simulate data for the model.

    This function generates simulated data and returns a dictionary containing
    various model inputs.

    Parameters
    ----------
    params : dict
        Model parameters.

    Returns
    -------
    dict
        Dictionary with the following keys:

        X : array_like, shape (3, J, M)
            Product characteristics.
        p : array_like, shape (J, M)
            Prices.
        s : array_like, shape (J , M)
            Market shares.
        W : array_like, shape (J,)
            Cost shifter.
        Z : array_like, shape (J, M)
            Cost shifter.
    """
    jm_shape = (params["J"], params["M"])
    Xdist = params["X"]
    cdist = params["cost"]

    # construct product characteristics
    X1 = np.ones(jm_shape)
    X2 = np.random.uniform(Xdist["X2"]["a"], Xdist["X2"]["b"], jm_shape)
    X3 = np.random.normal(Xdist["X3"]["mu"], Xdist["X3"]["sigma"], jm_shape)
    X = np.column_stack((X1, X2, X3))

    # draw shocks and cost shifters
    xi = np.random.normal(params["xi"]["mu"], params["xi"]["sigma"], jm_shape)
    Z = np.random.normal(cdist["Z"]["mu"], cdist["Z"]["sigma"], jm_shape)
    W = np.random.normal(cdist["W"]["mu"], cdist["W"]["sigma"], params["J"])

    # @TODO: just placeholder right now
    # determine prices and shares
    p_init = np.zeros(jm_shape)
    s = derive_shares(params, X, p_init, W, Z)
    p = solve_prices(params, s, X, W, Z)

    return {
        "X": X,
        "p": p,
        "s": s,
        "W": W,
        "Z": Z,
    }


def derive_shares(params, X, p, W, Z):
    """
    Derive market shares from the model.

    Parameters
    ----------
    params : dict
        Model parameters.
    X : array_like, shape (3, J, M)
        Product characteristics.
    p : array_like, shape (J, M)
        Prices.
    W : array_like, shape (J,)
        Cost shifter.
    Z : array_like, shape (J, M)
        Cost shifter.

    Returns
    -------
    array_like, shape (J, M)
        Market shares.
    """
    return np.zeros((params["J"], params["M"]))


def solve_prices(params, s, X, W, Z):
    """
    Solve for prices from the model.

    Parameters
    ----------
    params : dict
        Model parameters.
    s : array_like, shape (J, M)
        Market shares.
    X : array_like, shape (3, J, M)
        Product characteristics.
    W : array_like, shape (J,)
        Cost shifter.
    Z : array_like, shape (J, M)
        Cost shifter.

    Returns
    -------
    array_like, shape (J, M)
        Prices.
    """
    return np.zeros(params["J"] * params["M"])


def marginal_cost(W, Z, eta, gammas):
    """
    Helper function to implement marginal cost function.

    Parameters
    ----------
    W : array_like, shape (J,)
        Cost shifter.
    Z : array_like, shape (J, M)
        Cost shifter.
    eta : array_like, shape (J, M)
        Cost error term.
    params : dict
        Model parameters.

    Returns
    -------
    array_like, shape (J,)
        Marginal cost.
    """
    return gammas[0] + gammas[1] * W[:, np.newaxis] + gammas[2] * Z + eta
