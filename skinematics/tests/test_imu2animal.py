"""Unit test of imu2animal module

"""

import os.path as osp
import unittest as ut
import numpy as np
import numpy.testing as npt
import shelve
from skinematics.utils import imu2animal as i2a
import skinematics.vector as skvector


class TestIMU2Animal(ut.TestCase):
    """Test `IMU2Animal` methods

    """
    def setUp(self):
        here = osp.dirname(osp.abspath(__file__))
        in_file = osp.join(here, "data", "gertrude", "gert_imu_frame")
        with shelve.open(in_file) as gert_db:
            self.imu = gert_db["gert_imu"]
            self.long_srfc = gert_db["dive"]["long_srfc"]
        # An instance with good accelerometer filtering
        self.imu2animal = i2a.IMU2Animal(self.long_srfc, self.imu, (99, 2))

    def test_init(self):
        imu2animal = i2a.IMU2Animal(self.long_srfc, self.imu)
        self.assertIsInstance(imu2animal, i2a.IMU2Animal)
        # Check providing filtering argument
        imu2animal = i2a.IMU2Animal(self.long_srfc, self.imu, (25, 2))
        self.assertIsInstance(imu2animal, i2a.IMU2Animal)

    def test_get_surface_vectors(self):
        imu2animal = self.imu2animal
        idx = imu2animal.surface_details.index[10]
        acc_idx = imu2animal.get_surface_vectors(idx, "acceleration")
        acc_idx_E = [-0.1770691, -0.2477986,  1.0155698]  # expected
        acc_idx_mu = acc_idx.mean(axis=0).values
        npt.assert_almost_equal(acc_idx_mu, acc_idx_E)
        # Check smoothed acceleration
        acc_idx = imu2animal.get_surface_vectors(idx, "acceleration",
                                                 smoothed_accel=True)
        acc_idx_E = [-0.1772635, -0.24425, 1.0174348]  # expected
        acc_idx_mu = acc_idx.mean(axis=0).values
        npt.assert_almost_equal(acc_idx_mu, acc_idx_E)
        # Check getting magnetic density
        magnt_idx = imu2animal.get_surface_vectors(idx, "magnetic_density")
        magnt_idx_E = [14.6451379, 5.4446945, -41.141773]  # expected
        magnt_idx_mu = magnt_idx.mean(axis=0).values
        npt.assert_almost_equal(magnt_idx_mu, magnt_idx_E)
        # Check getting depth
        depth_idx = imu2animal.get_surface_vectors(idx, "depth")
        depth_idx_E = 1.070827300498207  # expected
        depth_idx_mu = depth_idx.mean()
        npt.assert_almost_equal(depth_idx_mu, depth_idx_E)

    def test_get_orientation(self):
        imu2animal = self.imu2animal
        idx = imu2animal.surface_details.index[20]
        Rctr2i, svd = imu2animal.get_orientation(idx, plot=False,
                                                 animate=False)
        Rctr2i_E = np.array([[0.9843929, 0.104275, -0.1417652],
                             [-0.134066, 0.9661851, -0.2202559],
                             [0.1140042, 0.2358242, 0.9650855]])
        svd_E = (np.array([[-0.9843929, -0.104275, -0.1417652],
                           [0.134066, -0.9661851, -0.2202559],
                           [-0.1140042, -0.2358242, 0.9650855]]),
                 np.array([1.6949134e-02, 2.0848817e-03, 5.8511064e-05]),
                 np.array([[-0.9843929, 0.134066, -0.1140042],
                           [-0.104275, -0.9661851, -0.2358242],
                           [-0.1417652, -0.2202559, 0.9650855]]))
        npt.assert_almost_equal(Rctr2i.as_dcm(), Rctr2i_E)
        npt.assert_almost_equal(svd[0], svd_E[0])
        npt.assert_almost_equal(svd[1], svd_E[1])
        npt.assert_almost_equal(svd[2], svd_E[2])
        # Covariance of normalized (smoothed) acceleration in the
        # transformed frame for the selected surfacing segment should be
        # close to zero, as this is the one used to find the plane
        acci_sg = imu2animal.get_surface_vectors(idx, i2a._ACCEL_NAME,
                                                 smoothed_accel=True)
        acci_sg_body = Rctr2i.apply(acci_sg, inverse=True)
        acci_sg_cov = np.cov(skvector.normalize(acci_sg_body),
                             rowvar=False)
        npt.assert_array_almost_equal(np.tril(acci_sg_cov, k=-1),
                                      np.zeros((3, 3)))

    def test_get_orientations(self):
        imu2animal = self.imu2animal
        orientations = imu2animal.get_orientations()
        euler_mu = orientations[["phi", "theta", "psi"]].mean(axis=0)
        euler_mu_E = [-7.0749517, 8.0709853, -7.7198771]
        npt.assert_almost_equal(euler_mu, euler_mu_E)

    def test_orient_surfacing(self):
        imu2animal = self.imu2animal
        idx = imu2animal.surface_details.index[33]
        Rctr2i, svd = imu2animal.get_orientation(idx, plot=False,
                                                 animate=False)
        imu_bodyi = imu2animal.orient_surfacing(idx, Rctr2i)
        acci = imu_bodyi[i2a._ACCEL_NAME]
        acci_mu = acci.mean(axis=0).values
        acci_mu_E = [-0.0243503, 0.0657713, 1.0543464]
        npt.assert_almost_equal(acci_mu, acci_mu_E)

    def test_orient_surfacings(self):
        imu2animal = self.imu2animal
        orients = imu2animal.orient_surfacings()
        shape_E = [68, 82800]
        npt.assert_almost_equal(orients.index.levshape, shape_E)

    def test_filter_surfacings(self):
        imu2animal = self.imu2animal
        imu2animal.get_orientations()
        imu2animal.filter_surfacings((0.04, 0.06))
        srfc_shape_E = [23, 7]
        npt.assert_equal(imu2animal.surface_details.shape, srfc_shape_E)
        euler_mu = (imu2animal.orientations[["phi", "theta", "psi"]]
                    .mean(axis=0))
        euler_mu_E = [-7.2519366,  7.6130653, -6.9231422]
        npt.assert_almost_equal(euler_mu, euler_mu_E)

    def test_orient_IMU(self):
        imu2animal = self.imu2animal
        imu2animal.get_orientations()
        imu2animal.filter_surfacings((0.04, 0.06))
        imus_body = imu2animal.orient_IMU()
        shape_E = [23, 171135]
        npt.assert_almost_equal(imus_body.index.levshape, shape_E)


if __name__ == '__main__':
    ut.main()
