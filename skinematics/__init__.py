""""Functions for working with 3D kinematics

Functions and classes useful for working with quaternions and rotation
matrices.

Compatible Python 3.

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2020 Thomas Haslwanter <thomas.haslwanter@fh-ooe.at>

"""

import importlib

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at>"
__license__ = "BSD 2-Clause License"
__version__ = "0.8.3"


required_imports = ['markers', 'quat', 'rotmat', 'vector', 'sensors']
optional_imports = ['imus', 'misc', 'view']

__all__ = []
for _m in required_imports:
    importlib.import_module('.'+_m, package='skinematics')
    __all__.append(_m)

for _m in optional_imports:
    try:
        importlib.import_module('.'+_m, package='skinematics')
        __all__.append(_m)
    except ModuleNotFoundError:
        print(("Failed to import optional module {}. "
               "Install optional dependencies").format(_m))
