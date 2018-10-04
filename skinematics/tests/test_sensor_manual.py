"""
Test manual data entry, through subclassing "IMU_Base"

Author: Thomas Haslwanter
"""

import unittest
from time import sleep
from skinematics.sensors.xsens import XSens
from skinematics.sensors.manual import MyOwnSensor
import os


class TestSequenceFunctions(unittest.TestCase):

    def test_import_manual(self):
        # Get data, with a specified input from an XSens system
        myPath = os.path.dirname(os.path.abspath(__file__))
        in_file = os.path.join(myPath, "data", "data_xsens.txt")
        sensor = XSens(in_file=in_file, q_type=None)
        transfer_data = {'rate': sensor.rate,
                         'acc': sensor.acc,
                         'omega': sensor.omega,
                         'mag': sensor.mag}
        my_sensor = MyOwnSensor(in_file="My own 123 sensor.",
                                in_data=transfer_data)

        self.assertEqual(my_sensor.rate, 50.)
        self.assertAlmostEqual((my_sensor.omega[0, 2] -
                                0.050860000000000002), 0)


if __name__ == "__main__":
    unittest.main()
    print("Thanks for using programs from Thomas!")
    sleep(0.1)
