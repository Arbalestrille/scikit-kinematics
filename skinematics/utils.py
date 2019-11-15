"""Utility functions for IMU data analysis

"""

import numpy as np


# Mapping of error type with corresponding tau and slope
_ERROR_DEFS = {"Q": [np.sqrt(3), -1], "ARW": [1.0, -0.5],
               "BI": [np.nan, 0], "RRW": [3.0, 0.5],
               "RR": [np.sqrt(2), 1]}


def _line_fun(t, alpha, tau_crit, adev_crit):
    """Find Allan sigma coefficient from line and point

    Log-log parameterization of the point-slope line equation.

    Parameters
    ----------
    t : {float, array_like}
        Averaging time
    alpha : float
        Slope of Allan deviation line
    tau_crit : float
        Observed averaging time
    adev_crit : float
        Observed Allan deviation at `tau_crit`
    """
    return(10 ** (alpha * (np.log10(t) - np.log10(tau_crit))
                  + np.log10(adev_crit)))


def allan_slope_coefs(taus, adevs):
    """Compute Allan deviation coefficient for each error type

    Given averaging intervals `taus` and corresponding Allan deviation
    `adevs`, compute the Allan deviation coefficient for each error type:

      - Quantization
      - (Angle, Velocity) Random Walk
      - Bias Instability
      - Rate Random Walk
      - Rate Ramp

    Parameters
    ----------
    taus : array_like
        Averaging times
    adevs : array_like
        Allan deviation

    Returns
    -------
    sigmas_hat: dict
        Dictionary with `tau` value and associated Allan deviation
        coefficient for each error type.

    Notes
    -----
    See e.g. Jurado et al. (2019)::

      Jurado, J, Schubert Kabban, CM, Raquet, J (2019).  A regression-based
      methodology to improve estimation of inertial sensor errors using
      Allan variance data. Navigation 66:251-263.

    """
    # Find the gradient (1st derivative)
    tau_log = np.log10(taus)
    adev_log = np.log10(adevs)
    dadev = np.gradient(adev_log, tau_log)

    sigmas_hat = dict.fromkeys(_ERROR_DEFS.keys())
    for alpha_type, details in _ERROR_DEFS.items():
        alpha_x = details[0]
        alpha = details[1]
        # Find index where slope is closest to alpha
        tau_alpha_idx = np.argmin(np.abs(dadev - alpha))
        tau_crit = taus[tau_alpha_idx]
        adev_crit = adevs[tau_alpha_idx]
        if alpha_type == "BI":
            alpha_x = tau_crit
            scale_B = np.sqrt(2 * np.log(2) / np.pi)
            sigma_hat = adev_crit / scale_B
        else:
            sigma_hat = _line_fun(alpha_x, alpha, tau_crit, adev_crit)
        sigmas_hat[alpha_type] = [alpha_x, sigma_hat]

    return(sigmas_hat)
