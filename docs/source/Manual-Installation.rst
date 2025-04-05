Installing OpenVSF
================================================

TL;DR
-----

Install locally in editable mode::

   pip install -e .

Troubleshooting
---------------

1. Cannot install ``mesh2sdf`` due to build errors
--------------------------------------------------

If you encounter the following error during installation:

.. code-block:: bash

   RuntimeError: Unsupported compiler -- at least C++11 support is needed!
     [end of output]

   note: This error originates from a subprocess, and is likely not a problem with pip.
   ERROR: Failed building wheel for mesh2sdf

This indicates that your system lacks a compatible C++ compiler with at least C++11 support.

**Solution:**

Ensure you have an appropriate C++ compiler installed. For example, on Ubuntu, you can install ``g++`` by running:

.. code-block:: bash

   sudo apt-get install g++

   