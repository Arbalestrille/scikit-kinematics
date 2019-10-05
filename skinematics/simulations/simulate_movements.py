"""Simulations of IMU-signals during 3D movements"""

import numpy as np
import scipy as sp
import skinematics as skin
from scipy.constants import g


def make_gauss(rate=1000, duration=1, t_0=0.5, sigma=0.1):
    """Generate a Gaussian curve with an area of 1 under the curve.

    Parameters
    ----------
    rate : Sample rate [Hz]; float
    duration : Length of Gaussian curve [sec]; float
    t_0 : Center of Gaussian curve [sec]; float
    sigma : Standard deviation of Gaussian curve [sec]; float

    Returns
    -------
    gauss : ndarray, shape (n,)
        Gaussian curve
    t : ndarray, shape (n,)
        Correponding time vector
    dt : float
        Time-step
    """

    dt = 1.0 / rate
    t = np.arange(0, duration, dt)
    gaussian = np.exp(-(t - t_0) ** 2 / (2 * sigma ** 2))
    gauss_integral = sp.integrate.simps(gaussian, dx=dt)
    gauss = gaussian / gauss_integral

    return (gauss, t, dt)


def change_orientation(rate, duration, q_start=[0, 0, 0],
                       rotation_axis=[0, 0, 1], deg=90):
    """Simulate a simple rotation with Gaussian velocity profile about a fixed
    axis.

    Parameters
    ----------
    rate : Sample rate [Hz]; float
    duration : Duration of rotation [sec]; float
    q_start: Quaternion or quaternion vector, describing the initial
        orientation; 3D vector
    rotation_axis : Axis of rotation; 3D vector, or list of 3 numbers
    deg : Angle of rotation [deg]; float

    Returns
    -------
    omega : ndarray, shape (n,3)
        Angular velocity [deg/s]
    quat : ndarray, shape (n,4)
        Orientation, expressed as quaternion
    t : ndarray, shape (n,)
        Time-vector [s]

    """

    gauss, t, dt = make_gauss(rate, duration)
    rad = np.deg2rad(deg)

    # Rotation about a single axis, with a gaussian velocity profile
    rotation_axis = np.atleast_2d(skin.vector.normalize(rotation_axis))
    omega = (rotation_axis.T * gauss).T * rad

    # Corresponding orientation
    quat = skin.quat.calc_quat(omega, q0=q_start, rate=rate, CStype="sf")

    return (omega, quat, t)


def change_position(rate, duration, start_pos=[0, 0, 0],
                    direction=[1, 0, 0], distance=0.5):
    """Simulate a simple translation with Gaussian velocity profile along a
    space-fixed axis.

    Parameters
    ----------
    rate : Sample rate [Hz]; float
    duration : Duration of movement [sec]; float
    start_pos: Starting position [m]; 3D vector, or list of 3 numbers
    direction : Axis of translation; 3D vector, or list if 3 numbers
    distance : Magnitude of displacement [m]; float

    Returns
    -------
    pos : ndarray, shape (n,3)
        Position [m]
    vel : ndarray, shape (n,3)
        Velocity [m/s]
    acc : ndarray, shape (n,3)
        Acceleration (m/s**2)

    """

    # Catch a zero-translation
    if direction == [0, 0, 0]:
        direction = [1, 0, 0]
        distance = 0

    gauss, t, dt = make_gauss(rate, duration)
    direction = np.atleast_2d(skin.vector.normalize(direction))

    # Translation along a single axis, with a gaussian velocity profile
    linear = {}     # linear movement
    linear["vel"] = (direction.T * gauss).T * distance

    linear["acc"] = sp.signal.savgol_filter(linear["vel"], window_length=5,
                                            polyorder=3, deriv=1,
                                            delta=dt, axis=0)

    # note: 1 point less than vel!
    linear["pos"] = sp.integrate.cumtrapz(linear["vel"], dx=dt, axis=0)
    linear["pos"] = np.vstack((np.zeros(3), linear["pos"]))
    linear["pos"] += start_pos

    return linear["pos"], linear["vel"], linear["acc"]


def simulate_imu(rate, t_move, t_total, q_init=[0, 0, 0],
                 rotation_axis=[0, 0, 1], deg=0, pos_init=[0, 0, 0],
                 direction=[1, 0, 0], distance=0, B0=[-1, 0, -1]):
    """Simulate the signals in an IMU, based on acc, etc. in space-fixed
    coordinates After the movement part, the IMU remains stationary for the
    rest of the "duration".

    Parameters
    ----------
    rate : Sample rate [Hz]; float
    t_move : Duration of movement [sec]; float
    t_total : Duration of total segment (movement + stationary) [sec]; float
    q_init : Quaternion or quaternion vector, describing the initial
        orientation; 3D vector
    rotation_axis : Axis of rotation; 3D vector, or list of 3 numbers
    deg : Angle of rotation [deg]; float
    pos_init: Starting position [m]; 3D vector, or list of 3 numbers
    direction : Axis of translation; 3D vector, or list if 3 numbers
    distance : Magnitude of displacement [m]; float
    B0 : ndarray, shape (1,3)
        Orientation of magnetic field, with respect to space-fixed CS

    Returns
    -------
    imu : dictionary, with the following fields
        * rate : sample rate [Hz]; float
        * time : time stamps [sec]; ndarray, shape (n,)
        * gia : gravito-inertial acceleration with respect to the IMU
            [m/s**2]; ndarray, shape (n,3)
        * omega : angular velocity re IMU [rad/s]; ndarray, shape (n,3)
        * magnetic : orientation of local magnetic field re IMU; ndarray,
            shape (n,3)
    body : dictionary, with the following fields
        * pos : position of object [m]; ndarray, shape (n,3)
        * quat : orientation of object; ndarray, shape (n,4)

    """

    omega, quat, t = change_orientation(rate=rate, duration=t_move,
                                        q_start=q_init,
                                        rotation_axis=rotation_axis, deg=deg)
    pos, vel, acc = change_position(rate, duration=t_move,
                                    start_pos=pos_init,
                                    direction=direction, distance=distance)

    num_pts = t_total * rate
    num_quat = quat.shape[0]
    remaining = num_pts - num_quat

    omega = np.vstack((omega, np.tile(np.zeros(3), (remaining, 1))))
    quat = np.vstack((quat, np.tile(quat[-1], (remaining, 1))))
    acc = np.vstack((acc, np.tile(np.zeros(3), (remaining, 1))))
    vel = np.vstack((vel, np.tile(np.zeros(3), (remaining, 1))))
    pos = np.vstack((pos, np.tile(pos[-1], (remaining, 1))))

    # gravito-inertial acceleration (GIA) = gravity + acceleration
    acc_g = np.r_[0, 0, g]   # [m/s**2]
    gia_sf = acc + acc_g        # space-fixed CS

    # Calculate the corresponding IMU-signals
    imu = {}
    q_inv = skin.quat.q_inv(quat)
    # GIF, in a body-fixed coordinate system(CS)
    imu["gia"] = skin.vector.rotate_vector(gia_sf, q_inv)

    # Omega, in a body-fixed CS
    imu["omega"] = skin.vector.rotate_vector(omega, q_inv)

    # Magnetic field
    imu["magnetic"] = skin.vector.rotate_vector(B0, q_inv)

    # Rate and time-stamps
    imu["rate"] = rate
    imu["time"] = np.arange(num_pts) / rate

    # Position and orientation of body
    body = {}
    body["pos"] = pos
    body["quat"] = quat

    return imu, body


def save_as(imu_data, data_format, file_name):
    """Save the input in a specifie data-format

    Parameters
    ----------
    imu_data : dictionary, with the following fields

        * gia : gravito-inertial acceleration with respect to the IMU
            [m/s**2]; ndarray, shape (n,3)
        * omega : angular velocity re IMU [rad/s]; ndarray, shape (n,3)
        * magnetic : orientation of local magnetic field re IMU; ndarray,
            shape (n,3)
    data_format : string
        Pre-defined data-type. Has to be one of the following
        * "ngimu" [tbd]
        * "xsens" [tbd]

    """
    pass


def _imu_cat(N, **kwargs):
    """Generate a sequence of movements with simulated IMU signals"""
    rate = kwargs.pop("rate")
    t_moves = kwargs.pop("t_moves")
    t_totals = kwargs.pop("t_totals")
    q_init = kwargs.pop("q_init")
    rotation_axes = kwargs.pop("rotation_axes")
    degs = kwargs.pop("degs")
    pos_init = kwargs.pop("pos_init")
    directions = kwargs.pop("directions")
    distances = kwargs.pop("distances")
    B0 = kwargs.pop("B0")

    imu_l = []
    pq_l = []
    for i in range(N):
        if i > 0:
            q_init_i = q_init[-1]
            pos_init_i = pos_init[-1]
        else:
            q_init_i = q_init
            pos_init_i = pos_init

        new_move, new_pq = simulate_imu(rate=rate, t_move=t_moves[i],
                                        t_total=t_totals[i],
                                        q_init=q_init_i,
                                        rotation_axis=rotation_axes[i],
                                        deg=degs[i], pos_init=pos_init_i,
                                        direction=directions[i],
                                        distance=distances[i], B0=B0)
        # Need only vector part of quat here to build q_init
        q_init = np.vstack((q_init, new_pq["quat"][-1][1:]))
        pos_init = np.vstack((pos_init, new_pq["pos"][-1]))
        imu_l.append(new_move)
        pq_l.append(new_pq)

    imu_all = imu_l[0]
    pq_all = pq_l[0]
    for imu in imu_l[1:]:
        for key in imu_all.keys():
            if (key == "rate") or (key == "time"):
                continue
            else:
                imu_all[key] = np.vstack((imu_all[key], imu[key]))
        imu_all["rate"] = rate
        imu_all["time"] = np.arange(imu_all["gia"].shape[0]) / rate

    for pq in pq_l[1:]:
        for key in pq_all.keys():
            pq_all[key] = np.vstack((pq_all[key], pq[key]))

    return(imu_all)


if __name__ == "__main__":
    # Set up a series of 6 movements
    rate = 100                  # [Hz]
    t_moves = [1] * 6           # [s]
    t_totals = [1] * 6          # [s]
    q_init = np.array([0, 0, 0])
    degs = [50, -50, 50, -50, 50, -50]
    rotation_axes = [[0, 0, 1]] * 2
    rotation_axes.extend([[0, 1, 0]] * 2)
    rotation_axes.extend([[1, 0, 0]] * 2)
    pos_init = np.array([-10, -5, 0])
    translations = [[1, 0, 0]] * 6
    distances = [0] * 6
    B0 = skin.vector.normalize([0, -1, -1])

    imu = _imu_cat(N=6, rate=rate, t_moves=t_moves, t_totals=t_totals,
                   q_init=q_init, rotation_axes=rotation_axes, degs=degs,
                   pos_init=pos_init, directions=translations,
                   distances=distances, B0=B0)
    q, pos, vel = skin.imus.analytical(R_initialOrientation=np.eye(3),
                                       omega=imu["omega"],
                                       initialPosition=np.zeros(3),
                                       accMeasured=imu["gia"],
                                       rate=rate)
