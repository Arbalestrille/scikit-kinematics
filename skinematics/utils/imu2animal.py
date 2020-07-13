"""Estimation of IMU orientation on body frame

One of the first tasks in analysis of IMU data from devices mounted on wild
animals is estimating the orientation of the instrument on the animal.  The
:class:`IMU2Animal` class facilitates this process for devices mounted on
air-breathing marine/aquatic animals that regularly come up to the surface
to breathe.  See Johnson (2011) for details on the approach.

Notes
-----

A right-handed coordinate system is assumed in the input `IMU` data.

.. image:: .static/images/rhs_frame.png
   :scale: 40%

The :class:`IMU2Animal` class provides an integrated approach to estimating
the relative orientation of two reference frames: a) body (b) and b) sensor
(s).

.. image:: .static/images/imu2animal_frames.png
   :scale: 40%

.. autosummary::

   IMU2Animal
   scatterIMU3D
   scatterIMU_svd
   tsplotIMU_depth

"""

import logging
import numpy as np
import pandas as pd
import shelve
from scipy.spatial.transform import Rotation as R  # for zyx covention
import scipy.signal as sig
import matplotlib as mpl        # noqa:F401
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa:F401
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from skinematics.vector import normalize as normalize_vectors
from skinematics.vector import angle as vectors_angle

__all__ = ["scatterIMU_svd", "scatterIMU3D", "tsplotIMU_depth",
           "IMU2Animal"]

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())

# Some shortcuts for names
_ACCEL_NAME = "acceleration"
_OMEGA_NAME = "angular_velocity"
_MAGNT_NAME = "magnetic_density"
# Plot constants
_FIG3X1 = (11, 11)
_LEG2X1_ANCHOR = (0.5, -0.23)
_LEG3X1_ANCHOR = (0.5, -0.23)


class Arrow3D(FancyArrowPatch):
    """Subclass extending FancyArrowPatch to 3D"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def scatterIMU_svd(vectors, svd, R_ctr2i, normalize=False, center=False,
                   title=None, animate=False, animate_file=None, **kwargs):
    """3D plotting of vectors and singular vector matrices

    Plots a surface of the 2 top-most singular vectors computed from SVD
    overlaid on the scatter of vectors.

    Parameters
    ----------
    vectors : array_like, shape (N, 3) or (3,)
        Array with vectors to plot.
    svd : tuple
        The 3 arrays from Singular Value Decomposition (left singular
        vectors, sigma, right singular vectors).
    R_ctr2i : Rotation
        `Rotation` object describing the rotation from the *centered* body
        frame to the `IMU` frame.
    normalize : bool, optional
        Whether to normalize vectors.  Default assumes input is normalized.
    center : bool, optional
        Whether to center vectors (i.e. subtract the mean).  Default
        assumes input is centered.
    title : str, optional
        Title for the plot.
    animate : bool, optional
        Whether to animate the plot.
    animate_file : str, optional
        Output file for the animation.
    **kwargs : bool, optional keyword arguments
        Arguments passed to ``pyplot.figure`` (e.g. `figsize`).

    Returns
    -------
    Axes
        Matplotlib `Axes` object.

    """
    if normalize:
        vectors_n = normalize_vectors(vectors)
    else:
        vectors_n = vectors

    if center:
        vectors_c = vectors_n - vectors_n.mean(axis=0)
    else:
        vectors_c = vectors_n

    view_elev = 15              # elevation for 3D view
    # Unpack SVD
    uu, ss, vv = svd
    # Rows across `uu` define the eigenvectors or axes of the
    # orthonormal basis
    sss = ss * 15   # scale eigenvectors by singular values
    xx_ax, yy_ax, zz_ax = uu * sss  # unpack the rows
    xx = np.array([xx_ax[:2], -xx_ax[1::-1]])
    yy = np.array([yy_ax[:2], -yy_ax[1::-1]])
    zz = np.array([zz_ax[:2], -zz_ax[1::-1]])

    Reuler_lab = (r"$\phi={0:.1f}$, $\theta={1:.1f}$, $\psi={2:.1f}$"
                  .format(*_euler_ctr2body(R_ctr2i)))
    # Visualization of vectors
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.text2D(0.5, -0.05, Reuler_lab, ha="center", transform=ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view_elev)
    # ax.set_aspect(0.7)
    ax.scatter3D(vectors_c[:, 0], vectors_c[:, 1], vectors_c[:, 2],
                 s=3, alpha=0.5)
    arrow_opts = dict(arrowstyle="<|-|>,head_width=0.1", mutation_scale=15,
                      shrinkA=0, shrinkB=0)
    for i, col in zip(np.arange(2), ["red", "green"]):  # only first 2 PCs
        arrow_opts.update(color=col)
        scale_i = sss[i]
        a = Arrow3D((-(uu[0, i] * scale_i), uu[0, i] * scale_i),
                    (-(uu[1, i] * scale_i), uu[1, i] * scale_i),
                    (-(uu[2, i] * scale_i), uu[2, i] * scale_i),
                    **arrow_opts)
        ax.add_artist(a)
    surf_opts = dict(rstride=1, cstride=1, color="plum", alpha=0.5)
    ax.plot_surface(xx, yy, zz, **surf_opts)

    if animate:
        def anim_update(azim):
            ax.view_init(azim=azim, elev=view_elev)
            return(ax,)
        anim = animation.FuncAnimation(fig, anim_update, blit=False,
                                       frames=360, interval=20)
        anim.save(animate_file, fps=20)

    return(ax)


def _scatterIMU_svd(vectors, svd, R_b2i, normalize=False, title=None,
                    animate=False, animate_file=None, **kwargs):
    """3D plotting of vectors and singular vector matrices

    Plots a surface of the 2 top-most singular vectors computed from SVD
    overlaid on the scatter of vectors.

    Parameters
    ----------
    vectors : array_like, shape (N, 3) or (3,)
        Array with vectors to plot.
    svd : tuple
        The 3 arrays from Singular Value Decomposition (left singular
        vectors, sigma, right singular vectors).
    R_b2i : Rotation
        `Rotation` object describing the rotation from the body frame to
        the `IMU` frame.
    normalize : bool, optional
        Whether to normalize vectors.  Default assumes input is normalized.
    title : str, optional
        Title for the plot.
    animate : bool, optional
        Whether to animate the plot.
    animate_file : str, optional
        Output file for the animation.
    **kwargs : bool, optional keyword arguments
        Arguments passed to ``pyplot.figure`` (e.g. `figsize`).

    Returns
    -------
    Axes
        Matplotlib `Axes` object.

    """
    if normalize:
        vectors_n = normalize_vectors(vectors)
    else:
        vectors_n = vectors.copy()

    vectors_n -= vectors_n.mean(axis=0)

    view_elev = 15              # elevation for 3D view
    # Unpack SVD
    uu, ss, vv = svd
    # Rows across `uu` define the eigenvectors or axes of the
    # orthonormal basis
    sss = np.ones(3) / 10   # scale singular vectors
    xx_ax, yy_ax, zz_ax = uu * sss  # unpack the rows
    xx = np.array([xx_ax[1:], -xx_ax[:0:-1]])
    yy = np.array([yy_ax[1:], -yy_ax[:0:-1]])
    zz = np.array([zz_ax[1:], -zz_ax[:0:-1]])

    Reuler_lab = (r"$\phi={0:.1f}$, $\theta={1:.1f}$, $\psi={2:.1f}$"
                  .format(*_euler_ctr2body(R_b2i)))
    # Visualization of vectors
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.text2D(0.5, -0.05, Reuler_lab, ha="center", transform=ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view_elev)
    # ax.set_aspect(0.7)
    ax.scatter3D(vectors_n[:, 0], vectors_n[:, 1], vectors_n[:, 2],
                 s=3, alpha=0.5)
    arrow_opts = dict(arrowstyle="<|-|>,head_width=0.1", mutation_scale=15,
                      shrinkA=0, shrinkB=0)
    for i, col in zip(np.arange(1, 3), ["green", "blue"]):  # only last 2 PAs
        arrow_opts.update(color=col)
        scale_i = sss[i]
        a = Arrow3D((-(uu[0, i] * scale_i), uu[0, i] * scale_i),
                    (-(uu[1, i] * scale_i), uu[1, i] * scale_i),
                    (-(uu[2, i] * scale_i), uu[2, i] * scale_i),
                    **arrow_opts)
        ax.add_artist(a)
    surf_opts = dict(rstride=1, cstride=1, color="plum", alpha=0.5)
    ax.plot_surface(xx, yy, zz, **surf_opts)

    if animate:
        def anim_update(azim):
            ax.view_init(azim=azim, elev=view_elev)
            return(ax,)
        anim = animation.FuncAnimation(fig, anim_update, blit=False,
                                       frames=360, interval=20)
        anim.save(animate_file, fps=20)

    return(ax)


def scatterIMU3D(vectors, col_vector, normalize=True, title=None,
                 animate=True, animate_file=None,
                 cbar_label=None, **kwargs):
    """3D plotting of vectors along with depth

    Scatter plot of vectors, colored by depth.

    Parameters
    ----------
    vectors : array_like, shape (N, 3) or (3,)
        Array with vectors to plot.
    col_vector : array_like, shape (N,)
        The array to be used for coloring symbols.
    normalize : bool, optional
        Whether to normalize vectors.
    title : str, optional
        Title for the plot.
    animate : bool, optional
        Whether to animate the plot.
    animate_file : str
        Output file for the animation.
    cbar_label : str, optional
        Title for the color bar.
    **kwargs : bool, optional keyword arguments
        Arguments passed to ``pyplot.figure`` (e.g. `figsize`).

    Returns
    -------
    Axes
        Matplotlib `Axes` object.

    """
    view_elev = 10
    if normalize:
        vecs = normalize_vectors(vectors)
    else:
        vecs = vectors

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view_elev)
    # ax.set_aspect(0.7)
    pts3d = ax.scatter3D(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                         s=3, c=col_vector, cmap="jet")
    cbar = plt.colorbar(pts3d, orientation="horizontal", fraction=0.05,
                        pad=0.05, shrink=0.7, aspect=50)
    if cbar_label:
        cbar.set_label(cbar_label)

    if animate:
        def anim_update(azim):
            ax.view_init(azim=azim, elev=view_elev)
            return(ax,)
        anim = animation.FuncAnimation(fig, anim_update, blit=False,
                                       frames=360, interval=20)
        anim.save(animate_file, fps=20)

    return(ax)


def tsplotIMU_depth(vectors, depth, **kwargs):
    """Plot depth and each column of (N,3) array

    Parameters
    ----------
    vectors : pandas.Dataframe, shape (N,3)
        Array with columns to plot in subplot.
    depth : pandas.Series, shape (N,)
        Array with depth measurements.
    **kwargs : bool, optional keyword arguments
        Arguments passed to ``pyplot.subplots`` (e.g. `figsize`).

    Returns
    -------
    Axes
        Matplotlib `Axes` object.

    """
    fig, axs = plt.subplots(4, 1, sharex=True, **kwargs)
    axs[0].set_ylabel("Depth [m]")
    axs[0].invert_yaxis()
    depth.plot(ax=axs[0], color="k")
    axs[0].axhline(0, linestyle="--", linewidth=0.75, color="k")
    vectors.plot(ax=axs[1:], subplots=True, legend=False)
    for i, col in enumerate(vectors.columns):
        axs[i + 1].set_ylabel(col)
        axs[i + 1].axhline(0, linestyle="--", linewidth=0.75, color="k")
    fig.tight_layout()
    return(axs)


def _euler_ctr2body(R_ctr2i):
    """Helper function to compute Euler angles from the body frame

    The rotation obtained from SVD on the covariance matrix is for the
    basis defined by the *centered* data, so pitch and roll refer to the
    backward facing animal.  Flipping the sign of these two angles makes
    them relative to the forward facing animal.

    Parameters
    ----------
    R_ctr2i : Rotation
        `Rotation` object describing the rotation from the *centered* body
        frame to the `IMU` frame.

    Returns
    -------
    ndarray, shape (3,)
        Array with `x`, `y`, and `z' Euler angles

    """
    euler_ctr = R_ctr2i.as_euler("XYZ", degrees=True)
    euler_ctr[:2] = -euler_ctr[:2]
    return(euler_ctr)


class IMU2Animal:
    """IMU orientation estimation framework

    The framework is based on the approach taken in TagTools (Matlab).
    However, the estimation of the orientation uses the covariance matrix,
    rather than the unscaled correlation (outer product) of acceleration as
    the input for Singular Value Decomposition.

    Attributes
    ----------
    surface_details : pandas.DataFrame
    imu : pandas.DataFrame
    accel_sg : pandas.DataFrame
        Smoothed acceleration signals, if so requested.  Otherwise, a
        pointer to the input acceleration signals.
    orientations : pandas.DataFrame
        A summary table describing the orientation of the IMU relative to
        the body frame for each surface period.

    Examples
    --------
    Construct IMU2Animal from two accessible DataFrame objects

    >>> import os.path as osp
    >>> import shelve
    >>> here = osp.dirname(__file__)
    >>> ifile = osp.join(osp.dirname(here), "tests", "data",
    ...                  "gertrude", "gert_imu_frame")
    >>> with shelve.open(ifile) as db:
    ...     imu = db["gert_imu"]
    ...     long_srfc = db["dive"]["long_srfc"]

    Apply Savitzky-Golay filter to acceleration to use as source data for
    Singular Value Decomposition.  The required parameters for the filter
    are: 1) window length and 2) polynomial order, expected as a tuple, in
    that order.

    >>> imu2whale = IMU2Animal(long_srfc, imu, savgol_parms=(99, 2))

    .. autosummary::

       describe_surfacing_durations
       get_surface_vectors
       get_orientation
       get_orientations
       orient_surfacing
       orient_surfacings
       orient_IMU
       filter_surfacings
       scatterIMU3D
       tsplotIMU_depth

    """

    def __init__(self, surface_details, imu, savgol_parms=None):
        """Set up attributes required for estimation

        Parameters
        ----------
        surface_details : pandas.DataFrame
            Table with details of surfacing intervals defining the periods
            to be used for estimating the orientation of the IMU.  It must
            have columns named `beg.surface` and `end.surface` as
            `datetime64` date type.  These indicate the beginning and end
            times for each surfacing period to analyze in the `imu` object.
            The index of this DataFrame should be a datetime indicating the
            end of ascent of the last dive prior to the beginning of the
            surface segment.
        imu : pandas.DataFrame
            Table with high-frequency IMU data providing the source data
            for the estimation.  The following column names are expected as
            the first level of a `pandas.MultiIndex` index structure::
                - "depth"
                - "acceleration"
                - "angular_velocity"
                - "magnetic_density"
        savgol_parms : tuple, optional
            A 2-element tuple with the window length and polynomial order
            for the Savitzky-Golay filter to smooth acceleration.

        """
        self.surface_details = surface_details
        self.imu = imu
        self.orientations = None

        acc = imu[_ACCEL_NAME]  # NOT A COPY
        if savgol_parms:
            win_width, polyord = savgol_parms
            acc_sg = sig.savgol_filter(acc.values, win_width, polyord,
                                       axis=0)
            self.accel_sg = pd.DataFrame(acc_sg, index=acc.index,
                                         columns=acc.columns)
        else:
            self.accel_sg = acc

    def __str__(self):
        imu_summary = self.imu.describe()
        srfc_dur_summary = self.describe_surfacing_durations()
        msg = ("IMU summary:\n{0}\n"
               "Surface segment duration summary:\n{1}")
        return(msg.format(imu_summary, srfc_dur_summary))

    def describe_surfacing_durations(self):
        """Return a summary of surfacing durations

        Returns
        -------
        pandas.Series
            Summary description of durations

        """
        srfc_dur = (self.surface_details["end.surface"] -
                    self.surface_details["beg.surface"])
        return(srfc_dur.describe())

    def _get_surface_mask(self, surface_idx):
        """Return mask for given index in surface details table

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.

        Returns
        -------
        mask : ndarray
            Boolean array.

        """
        # Isolate surface segment
        tlims = (self.surface_details.loc[surface_idx,
                                          ["beg.surface", "end.surface"]])
        mask = ((self.imu.index >= tlims[0]) &
                (self.imu.index < tlims[1]))
        return(mask)

    def get_surface_vectors(self, surface_idx, name, smoothed_accel=False):
        """Subset vectors of given name from IMU object for given index

        Return vectors of given name from IMU object for given index in
        surface details table.

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.
        name : str, {"acceleration", "magnetic_density", "depth"}
            Name of the vector array to subset.
        smoothed_accel : bool, optional
            Whether to return the smoothed acceleration (if
            `name="acceleration"`)

        Returns
        -------
        vectors : array_like, shape (N, 3)

        """
        names = [_ACCEL_NAME, _MAGNT_NAME, "depth"]
        if name not in names:
            msg = "name must be one of "
            raise ValueError(msg + ", ".join("\"{}\"".format(m) for m in
                                             names))

        # Isolate surface segment
        sfci_mask = self._get_surface_mask(surface_idx)
        # Apply mask
        if name == names[0]:
            if smoothed_accel:
                sfci_vector = self.accel_sg.loc[sfci_mask]
            else:
                sfci_vector = self.imu.loc[sfci_mask, _ACCEL_NAME]
        elif name == names[1]:
            sfci_vector = self.imu.loc[sfci_mask, _MAGNT_NAME]
        else:
            sfci_vector = self.imu.loc[sfci_mask, "depth"]
        return(sfci_vector)

    def get_orientation(self, surface_idx, plot=True, **kwargs):
        """Compute orientation for a given index in surface details table

        The orientation is computed via Singular Value Decomposition (SVD),
        assuming that the chosen `IMU` data correspond to the animal close
        to the surface and moving with negligible pitching and rolling
        motions.

        The function returns the rotation from the animal body frame to the
        `IMU` frame, along with the SVD matrices.  Note that a right-handed
        coordinate system is assumed.

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.
        plot : bool, optional
            Whether to generate a plot of the estimate
        **kwargs : bool, optional keyword arguments
            Arguments passed to ``scatterIMU_svd``.

        Returns
        -------
        Rotation, (U, S, V) : tuple
            `Rotation` object with potentially modified left singular
            vectors from Singular Value Decomposition (SVD), and full
            matrices from SVD (left singular vectors, sigma, and right
            singular vectors).

        Notes
        -----

        The reference frame for the output `Rotation` object is defined by
        the *centered* acceleration vectors.  All three components of this
        reference frame point in the positive direction of the centered
        data.  Furthermore, this rotation is not identical to the left
        singular vectors because it has been flipped 180 degrees so that
        the rotated acceleration is negatively related to the depth
        gradient, as it should.  Therefore, a copy of that basis was
        transformed by flipping the signs of these components so as to
        match the right-handed coordinate system assumed in the class.

        """
        # Isolate surface segment
        sfci_acc = self.get_surface_vectors(surface_idx, "acceleration",
                                            smoothed_accel=True)
        sfci_depth = self.get_surface_vectors(surface_idx, "depth")
        # Normalize vectors
        sfci_acc_n = normalize_vectors(sfci_acc.values)
        # Center vectors
        sfci_acc_mu = sfci_acc_n.mean(axis=0)
        sfci_acc_ctr = sfci_acc_n - sfci_acc_mu
        # For computing the covariance matrix, it doesn't really matter
        # whether we use centered data or not, as the operation
        # intrinsically centers anyway
        sfci_acc_cov = np.cov(sfci_acc_ctr, rowvar=False)
        # Singular value decomposition of the covariance matrix (i.e. PCA)
        uu, ss, vv = np.linalg.svd(sfci_acc_cov)
        Rfull = R.from_dcm(uu)
        Reuler = Rfull.as_euler("XYZ")
        eulers_lab = r"$\phi={0:.1f}$, $\theta={1:.1f}$, $\psi={2:.1f}$"
        eulers_maybe_lab = "Candidate Euler angles: " + eulers_lab
        eulers_final_lab = "Adjusted Euler angles: " + eulers_lab
        logger.info(eulers_maybe_lab.format(*np.degrees(Reuler)))
        sfci_acc_ctr_body = Rfull.apply(sfci_acc_ctr, inverse=True)
        # Check that relationship between acceleration along the
        # longitudinal axis and depth derivative is negative
        sfci_depth_1d = sfci_depth.values.flatten()
        accx_depth_coef = np.polyfit(sfci_acc_ctr_body[:, 0],
                                     np.gradient(sfci_depth_1d), deg=1)
        logger.info(r"Acceleration ~ $\nabla$Depth $b_1$={:.2f}"
                    .format(accx_depth_coef[0]))
        if accx_depth_coef[0] > 0:
            Rfull = Rfull * R.from_euler("Z", np.pi)
            Reuler = Rfull.as_euler("XYZ")
        logger.info(eulers_final_lab.format(*np.degrees(Reuler)))

        if plot:
            title = kwargs.pop("title",
                               "IMU-Frame Centered Acceleration [g]")
            animate = kwargs.pop("animate", True)
            surface_idx_str = surface_idx.strftime("%Y%m%d%H%M%S")
            animate_file = kwargs.pop("animate_file",
                                      ("imu2whale_{}.mp4"
                                       .format(surface_idx_str)))
            scatterIMU_svd(sfci_acc_ctr, (uu, ss, vv), Rfull, title=title,
                           animate=animate, animate_file=animate_file)

        return(Rfull, (uu, ss, vv))

    def get_orientations(self):
        """Obtain orientation for all periods in surface details table

        A quality index (`q`) for each estimate is calculated as::

            q = s_{3} / s_{2}

        where the dividend and divisor are the singular values for the
        third and second singular vectors, respectively.  A second
        indicator of the quality of the estimates is given as the standard
        deviation of the projection of the acceleration vector onto the
        second singular vector (i.e. pitching axis).

        Returns
        -------
        pandas.DataFrame

            DataFrame indexed the by the rows of the surface details table,
            and having the following columns:

            - `R`: Rotation
            - `SVD`: SVD matrices
            - `quality`: Tuple (quality index, std) for the estimate
            - `phi`: Roll angle (degrees) from body to IMU frame
            - `theta`: Pitch angle (degrees) from body to IMU frame
            - `psi`: Yaw angle (degrees) from body to IMU frame

        See Also
        --------
        get_orientation

        """
        srfcs = self.surface_details
        subd_l = ["R", "SVD", "quality", "phi", "theta", "psi"]
        # Careful with associating a new sub-dictionary here!
        orientations = {idx: dict.fromkeys(subd_l) for idx in srfcs.index}
        for idx in orientations.keys():
            msg = "Surface Period: {}"
            idx_str = idx.strftime("%Y%m%dT%H%M%S")
            logger.info(msg.format(idx_str))
            Ridx, svd = self.get_orientation(idx, plot=False)
            uu, ss, vv = svd
            acc_idx = self.get_surface_vectors(idx, "acceleration",
                                               smoothed_accel=True)
            acc_idx = normalize_vectors(acc_idx)
            acc_theta_sd = np.std(acc_idx @ uu[1, :][np.newaxis].T)
            orientations[idx]["quality"] = (ss[2] / ss[1], acc_theta_sd)
            orientations[idx]["R"] = Ridx
            orientations[idx]["SVD"] = svd
            phi, theta, psi = _euler_ctr2body(Ridx)
            orientations[idx]["phi"] = phi
            orientations[idx]["theta"] = theta
            orientations[idx]["psi"] = psi

        orientations = pd.DataFrame.from_dict(orientations,
                                              orient="index")
        orientations.index.rename(self.surface_details.index.name,
                                  inplace=True)
        self.orientations = orientations
        return(orientations)

    def orient_surfacing(self, surface_idx, R_b2i):
        """Apply orientation for a given index in surface details table

        Re-orient acceleration and magnetic density of `IMU` object
        relative to body frame, and return re-oriented `IMU` object.  Note
        that the provided `R_b2i` must describe the rotation from the body
        frame to the IMU frame.  The opposite rotation is applied to the
        acceleration and magnetic field vectors of the selected surface
        segment to orient the data in the body frame.

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.
        R_b2i : Rotation
            `Rotation` object representing the rotation from the body frame
            to the IMU frame.

        Returns
        -------
        pandas.DataFrame:
            The re-oriented IMU object.

        """
        sfci_mask = self._get_surface_mask(surface_idx)
        imu_sfci = self.imu.loc[sfci_mask, :].copy()
        acc = imu_sfci.xs(_ACCEL_NAME, axis=1)
        magnt = imu_sfci.xs(_MAGNT_NAME, axis=1)
        acc_body = R_b2i.apply(acc.values, inverse=True)
        magnt_body = R_b2i.apply(magnt.values, inverse=True)
        imu_sfci.loc[:, _ACCEL_NAME] = acc_body
        imu_sfci.loc[:, _MAGNT_NAME] = magnt_body
        return(imu_sfci)

    def orient_surfacings(self, R_all=None):
        """Apply orientation to all periods in surface details table

        If the `orientations` attribute was created, use it to build
        multi-index dataframe, othewise, create it.

        Parameters
        ----------
        R_all : Rotation, optional
            A `Rotation` object representing the rotation from the
            *centered* body frame to the IMU frame to be applied to all
            surface periods.  Default is to use the period-specific
            rotation.

        Returns
        -------
        pandas.DataFrame:
            MultiIndex table with the re-oriented surface IMU objects.

        See Also
        --------
        orient_surfacing

        """

        if self.orientations is None:
            logger.info("Calculating orientations")
            self.get_orientations()

        sfces = []
        for idx, row in self.orientations.iterrows():

            if R_all:
                sfci = self.orient_surfacing(idx, R_all)
            else:
                sfci = self.orient_surfacing(idx, row["R"])

            sfces.append(sfci)

        return(pd.concat(sfces, keys=self.orientations.index))

    def orient_IMU(self, R_all=None):
        """Apply orientations to the `IMU` object

        Use the rotations for each surface period segment to re-orient the
        `IMU` object to the animal frame.  Alternatively, apply the
        supplied rotation to the entire `IMU` object.

        An overview of the re-orientation process is illustrated below.

        .. image:: .static/images/time_series_rotation.png
           :width: 100%

        Each surface segment :math:`s_i`, delineated by beginning and
        ending times :math:`t_{0}^{s_i}` and :math:`t_{1}^{s_i}`
        (:math:`i=1` to :math:`i=n`), respectively, allows for an estimate
        of the `IMU` device orientation on the animal.  The corresponding
        rotation :math:`R_{s_{i}}` for transforming data from the `IMU`
        frame to the animal frame is applied to the data segments in the
        second line above, from the beginning of the deployment at
        :math:`t_0` to the end at :math:`t_k`.

        Parameters
        ----------
        R_all : Rotation, optional
            A `Rotation` object representing the rotation from the
            *centered* body frame to the IMU frame to be applied to the
            entire `IMU` object.

        Returns
        -------
        pandas.DataFrame:
            The re-oriented `IMU` object.  By default it is a MultiIndex
            DataFrame, with the topmost index representing the surface
            period index.

        See Also
        --------
        orient_surfacing
        orient_surfacings

        """

        if self.orientations is None:
            logger.info("Calculating orientations")
            self.get_orientations()

        def orient_segment(seg, R_seg):
            logger.info("Re-orienting from {0} to {1}"
                        .format(seg.index[0], seg.index[-1]))
            acc = seg.xs(_ACCEL_NAME, axis=1)
            magnt = seg.xs(_MAGNT_NAME, axis=1)
            acc_b = R_seg.apply(acc, inverse=True)
            magnt_b = R_seg.apply(magnt, inverse=True)
            seg.loc[:, _ACCEL_NAME] = acc_b
            seg.loc[:, _MAGNT_NAME] = magnt_b
            return(seg)

        imu = self.imu.copy()
        if R_all is None:
            imu_l = []
            # Subset first segment (start through beginning of descent into
            # dive at end of first surface period)
            next_desc = self.surface_details.iloc[0]["beg.next.desc"]
            # Dealing with annoying copy/view.  Drop the last index to
            # avoid duplicates
            seg0 = orient_segment((imu.loc[:next_desc]
                                   .copy().drop(next_desc)),
                                  self.orientations.iloc[0]["R"])
            imu_l.append(seg0)
            # Re-orient the middle section from the beginning of first dive
            # in prior diving section through beginning of next dive in
            # next diving section
            for idx, row in self.orientations.iloc[1:-1].iterrows():
                next_desc_i = (self.surface_details
                               .loc[idx]["beg.next.desc"])
                seg_i = (imu.loc[next_desc:next_desc_i]
                         .copy().drop(next_desc_i))
                imu_l.append(orient_segment(seg_i, row["R"]))
                # Set next descent to current one
                next_desc = next_desc_i
            # Last segment (use remnant `row` object)
            seg_end = orient_segment(imu.loc[next_desc:].copy(), row["R"])
            imu_l.append(seg_end)
            imus = pd.concat(imu_l, keys=self.orientations.index)
        else:
            imus = orient_segment(imu, R_all)

        return(imus)

    def filter_surfacings(self, qual_thresh):
        """Filter records from `surface_details` and `orientations`

        Remove records from the `surface_details` and `orientations`
        attributes, based on `quality` and, optionally, duration of the
        surface period.  Records with quality higher than the specified
        value, and (optionally) duration lower than the specified number of
        seconds, are removed.

        Parameters
        ----------
        qual_thresh : tuple
            Tuple with quality index and standard deviation of rolling
            motion during surface periods, in that order.  Records with
            quality value higher, and standard deviation of rolling motion
            higher than these thresholds are removed.

        """
        qual = self.orientations["quality"].apply(pd.Series)
        qual.columns = ["q_index", "phi_std"]
        bad = ((qual["q_index"] > qual_thresh[0]) |
               (qual["phi_std"] > qual_thresh[1]))
        bad_ids = self.orientations.index[bad]
        self.surface_details.drop(bad_ids, inplace=True)
        # Recalculate orientations as the edges have changed
        self.get_orientations()

    def scatterIMU3D(self, surface_idx, vec_name, smoothed_accel=False,
                     **kwargs):
        """Class plotting wrapper for module-level function

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.
        vec_name : str, {"acceleration", "magnetic_density"}
            Name of the vector array to subset.
        smoothed_accel : bool, optional
            Whether to plot the smoothed acceleration (if
            `vec_name="acceleration"`)
        **kwargs : bool, optional keyword arguments
            Arguments passed to ``scatterIMU3D``.

        Returns
        -------
        Axes
            Matplotlib `Axes` object.

        """
        if not smoothed_accel:
            vectors = self.get_surface_vectors(surface_idx, vec_name)
        else:
            vectors = self.get_surface_vectors(surface_idx, vec_name,
                                               smoothed_accel=True)
        depth = self.get_surface_vectors(surface_idx, "depth")
        ax = scatterIMU3D(vectors.values, depth.values.flatten(),
                          cbar_label="Depth [m]", **kwargs)
        return(ax)

    def tsplotIMU_depth(self, vec_name, surface_idx=None,
                        smoothed_accel=False, **kwargs):
        """Class plotting wrapper for module-level function

        Plot selected vector along with depth for the entire time series,
        and the edges of all surface segments overlaid on the depth panel.
        Alternatively, a specific surface segment can be plotted
        individually.

        Parameters
        ----------
        vec_name : str, {"acceleration", "magnetic_density"}
            Name of the vector array to subset.
        surface_idx : datetime64, optional
            Index in surface details table to be analyzed.
        smoothed_accel : bool, optional
            Whether to plot the smoothed acceleration (if
            `vec_name="acceleration"`)
        **kwargs : bool, optional keyword arguments
            Arguments passed to ``pyplot.subplots`` (e.g. `figsize`).

        Returns
        -------
        Axes
            Matplotlib `Axes` object.

        """
        if surface_idx:
            if not smoothed_accel:
                vectors = self.get_surface_vectors(surface_idx, vec_name)
            else:
                vectors = self.get_surface_vectors(surface_idx, vec_name,
                                                   smoothed_accel=True)
            depth = self.get_surface_vectors(surface_idx, "depth")
        else:
            if not smoothed_accel:
                vectors = self.imu[vec_name]
            else:
                vectors = self.accel_sg
            depth = self.imu["depth"]

        axs = tsplotIMU_depth(vectors, depth, **kwargs)

        if not surface_idx:
            sfc_begend = self.surface_details[["beg.surface",
                                               "end.surface"]]
            for sfc_id, lims in sfc_begend.iterrows():
                axs[0].axvspan(lims[0], lims[1], facecolor="g", alpha=0.5)

        return(axs)


class _TagTools(IMU2Animal):
    """A subclass implementing the approach taken by TagTools"""

    def get_orientation(self, surface_idx, plot=True, **kwargs):
        """Compute orientation for a given index in surface details table

        The orientation is computed via Singular Value Decomposition (SVD),
        assuming that the chosen `IMU` data correspond to the animal close
        to the surface and moving with negligible pitching and rolling
        motions.

        The function returns the rotation from the animal body frame to the
        `IMU` frame, along with the SVD matrices.  Note that a right-handed
        coordinate system is assumed.

        Parameters
        ----------
        surface_idx : datetime64
            Index in surface details table to be analyzed.
        plot : bool, optional
            Whether to generate a plot of the estimate
        **kwargs : bool, optional keyword arguments
            Arguments passed to ``_scatterIMU_svd``.

        Returns
        -------
        Rotation, (U, S, V) : tuple
            `Rotation` object with potentially modified left singular
            vectors from Singular Value Decomposition (SVD), and full
            matrices from SVD (left singular vectors, sigma, and right
            singular vectors).

        Notes
        -----

        The reference frame for the output `Rotation` object is defined by
        the origin of acceleration vectors.  The first component of this
        reference frame points in the positive direction from the origin of
        the data.  Furthermore, this rotation is not identical to the left
        singular vectors because the Rodrigues formula was used to
        transform that basis.

        """
        # Isolate surface segment (we transform input acceleration)
        sfci_acc = self.get_surface_vectors(surface_idx, "acceleration",
                                            smoothed_accel=False)
        sfci_depth = self.get_surface_vectors(surface_idx, "depth")
        # Normalize vectors
        sfci_acc_n = normalize_vectors(sfci_acc.values)
        sfci_acc_opr = sfci_acc_n.T @ sfci_acc_n
        # Singular value decomposition of the outer product
        uu, ss, vv = np.linalg.svd(sfci_acc_opr)
        logger.info("Lowest PA, x={0:.2E}, y={1:.2E}, z={2:.2E}"
                    .format(*uu[:, 2]))
        # Target y-vector in the animal frame
        ytarget = np.array([0, 1, 0])
        # Find unit vector normal to the plane formed by the animal frame
        # y-coordinate and lowest singular vector (the one along the
        # transversal axis of the animal, i.e. pitching axis)
        phi = normalize_vectors(np.cross(ytarget, uu[:, 2]))
        logger.info("Phi axis, x={0:.2E}, y={1:.2E}, z={2:.2E}"
                    .format(*phi))
        # Angle between target y-vector and lowest singular vector
        alpha = vectors_angle(ytarget, uu[:, 2])
        logger.info("Angle between y-axis and lowest PA: {:.2f} [deg]"
                    .format(np.degrees(alpha)))
        # Skew or cross-product matrix of phi
        skew = np.array([[0, -phi[2], phi[1]],
                         [phi[2], 0, -phi[0]],
                         [-phi[1], phi[0], 0]])
        # Assemble rotation matrix using Rodrigues' formula
        Rrod = (np.eye(3) + (np.sin(alpha) * skew) +
                (1 - np.cos(alpha)) * (skew @ skew))
        # [MJ]: So far we have anchored the rotation axis to the y-axis but
        # that only takes care of 2 of the 3 degrees of freedom in the tag
        # orientation. The tag could still be pitched up or down as this is
        # a rotation around the y-axis. To do something about this we have
        # to assume that the animal has a mean pitch of zero when at the
        # surface. We force this condition by rotating around the y-axis
        # (i.e., a pitch rotation).  Mean acceleration vector after
        # rotation by `Rrod`
        acc_muR = sfci_acc_n.mean(axis=0) @ Rrod
        # Remaining pitch angle that should be reduced to zero (need to
        # re-express this in RHS)
        p0 = np.arctan2(-acc_muR[0], acc_muR[2])
        logger.info("Pitch angle: {:.2f} [deg]".format(np.degrees(p0)))
        # Combine the pitching rotation above with `Rrod`
        Rfull_rod = R.from_euler("y", p0).as_dcm() @ Rrod.T
        Rfull = R.from_dcm(Rfull_rod.T)
        Reuler = Rfull.as_euler("XYZ")
        eulers_lab = r"$\phi={0:.1f}$, $\theta={1:.1f}$, $\psi={2:.1f}$"
        eulers_maybe_lab = "Candidate Euler angles: " + eulers_lab
        eulers_final_lab = "Adjusted Euler angles: " + eulers_lab
        logger.info(eulers_maybe_lab.format(*np.degrees(Reuler)))
        sfci_acc_n_body = Rfull.apply(sfci_acc_n, inverse=True)
        # Check that relationship between acceleration along the
        # longitudinal axis and depth derivative is negative
        sfci_depth_1d = sfci_depth.values.flatten()
        accx_depth_coef = np.polyfit(sfci_acc_n_body[:, 0],
                                     np.gradient(sfci_depth_1d), deg=1)
        logger.info(r"Acceleration ~ $\nabla$Depth $b_1$={:.2f}"
                    .format(accx_depth_coef[0]))
        if accx_depth_coef[0] > 0:
            Rfull = Rfull * R.from_euler("Z", np.pi)
            Reuler = Rfull.as_euler("XYZ")
        logger.info(eulers_final_lab.format(*np.degrees(Reuler)))

        if plot:
            title = kwargs.pop("title",
                               "IMU-Frame Centered Acceleration [g]")
            animate = kwargs.pop("animate", True)
            surface_idx_str = surface_idx.strftime("%Y%m%d%H%M%S")
            animate_file = kwargs.pop("animate_file",
                                      ("imu2whale_{}.mp4"
                                       .format(surface_idx_str)))
            _scatterIMU_svd(sfci_acc_n, (uu, ss, vv), Rfull, title=title,
                            animate=animate, animate_file=animate_file)

        return(Rfull, (uu, ss, vv))


if __name__ == '__main__':
    # import tempfile as tmp
    import os.path as osp
    logging.basicConfig(level=logging.INFO)
    pd.set_option("display.precision", 3)
    np.set_printoptions(precision=3, sign="+")

    # Bring in saved pre-processed data
    here = osp.dirname(__file__)
    in_file = osp.join(osp.dirname(here), "tests", "data", "gertrude",
                       "gert_imu_frame")  # no suffix needed
    with shelve.open(in_file) as gert_db:
        gert_imu = gert_db["gert_imu"]
        gert_long_srfc = gert_db["dive"]["long_srfc"]

    # Set up an instance to use throughout
    imu2whale = IMU2Animal(gert_long_srfc, gert_imu, (99, 2))
    print(imu2whale)
    # See documentation for richer demonstration
