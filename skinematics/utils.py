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


def fit_ellipsoid(vectors, f="rxyz"):
    """Fit an (non) rotated ellipsoid or sphere to 3D vector data

    Parameters
    ----------
    vectors: (N,3) array
        Array of measured x, y, z vector components.
    f: string
        String indicating the model to fit (one of 'rxyz', 'xyz', 'xy',
        'xz', 'yz', or 'sxyz'):
        rxyz : rotated ellipsoid (any axes)
        xyz  : non-rotated ellipsoid
        xy   : radius x=y
        xz   : radius x=z
        yz   : radius y=z
        sxyz ; radius x=y=z sphere

    Returns
    -------
    otuple: tuple
        Tuple with offset, gain, and rotation matrix, in that order.

    """

    FTYPES = ["rxyz", "xyz", "xy", "xz", "yz", "sxyz"]
    if f not in FTYPES:
        raise ValueError("f must be one of: {}".format(FTYPES))

    x = vectors[:, 0, np.newaxis]
    y = vectors[:, 1, np.newaxis]
    z = vectors[:, 2, np.newaxis]

    if f == "rxyz":
        D = np.hstack((x ** 2, y ** 2, z ** 2,
                       2 * x * y, 2 * x * z, 2 * y * z,
                       2 * x, 2 * y, 2 * z))
    elif f == "xyz":
        D = np.hstack((x ** 2, y ** 2, z ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "xy":
        D = np.hstack((x ** 2 + y ** 2, z ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "xz":
        D = np.hstack((x ** 2 + z ** 2, y ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "yz":
        D = np.hstack((y ** 2 + z ** 2, x ** 2,
                       2 * x, 2 * y, 2 * z))
    else:                       # sxyz
        D = np.hstack((x ** 2 + y ** 2 + z ** 2,
                       2 * x, 2 * y, 2 * z))

    v = np.linalg.lstsq(D, np.ones(D.shape[0]), rcond=None)[0]

    if f == "rxyz":
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8], -1]])
        ofs = np.linalg.lstsq(-A[:3, :3], v[[6, 7, 8]], rcond=None)[0]
        Tmtx = np.eye(4)
        Tmtx[3, :3] = ofs
        AT = Tmtx @ A @ Tmtx.T    # ellipsoid translated to 0, 0, 0
        ev, rotM = np.linalg.eig(AT[:3, :3] / -AT[3, 3])
        rotM = np.fliplr(rotM)
        ev = np.flip(ev)
        gain = np.sqrt(1.0 / ev)
    else:
        if f == "xyz":
            v = np.array([v[0], v[1], v[2], 0, 0, 0, v[3], v[4], v[5]])
        elif f == "xy":
            v = np.array([v[0], v[0], v[1], 0, 0, 0, v[2], v[3], v[4]])
        elif f == "xz":
            v = np.array([v[0], v[1], v[0], 0, 0, 0, v[2], v[3], v[4]])
        elif f == "yz":
            v = np.array([v[1], v[0], v[0], 0, 0, 0, v[2], v[3], v[4]])
        else:
            v = np.array([v[0], v[0], v[0], 0, 0, 0, v[1], v[2], v[3]])

        ofs = -(v[6:] / v[:3])
        rotM = np.eye(3)
        g = 1 + (v[6] ** 2 / v[0] + v[7] ** 2 / v[1] + v[8] ** 2 / v[2])
        gain = (np.sqrt(g / v[:3]))

    return(ofs, gain, rotM)


def _refine_ellipsoid_fit(gain, rotM):
    """Refine ellipsoid fit"""
    # m = 0
    # rm = 0
    # cm = 0
    pass


def apply_ellipsoid(vectors, offset, gain, rotM, ref_r):
    """Apply ellipsoid fit to vector array"""
    vectors_new = vectors.copy() - offset
    vectors_new = vectors_new @ rotM
    # Scale to sphere
    vectors_new = vectors_new / gain * ref_r
    return(vectors_new)
