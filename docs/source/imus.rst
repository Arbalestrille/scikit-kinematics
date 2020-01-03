.. _inertial-label:

======
 IMUs
======

.. automodule:: imus
   :members:

Sub-classing IMU-Base for custom sensor type
--------------------------------------------

If you have your own data format, you have to implement the corresponding
``get_data`` method. You can base it on:

* ``xsens.py`` if all your data are in one file.
* ``polulu.py`` if you have to manually enter data not stored by your
  program.
* ``xio.py`` if your data are stored in multiple files.

Existing Sensor Implementations
-------------------------------

XIO
^^^
.. automodule:: sensors.xio
   :members:

x-io NGIMU
----------
.. automodule:: sensors.xio_ngimu
   :members:

XSens
^^^^^
.. automodule:: sensors.xsens
   :members:

YEI
^^^
.. automodule:: sensors.yei
   :members:

Polulu
^^^^^^
.. automodule:: sensors.polulu
   :members:
