import unittest


class TestSequenceFunctions(unittest.TestCase):

    def test_skinematics(self):
        mdls = ['imus', 'markers', 'quat', 'rotmat', 'vector', 'viewer']
        for module in mdls:
            print(dir(module))
