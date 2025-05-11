Installing OpenVSF
==================

You can install OpenVSF via `pip`:

.. code-block:: bash

   pip install openvsf

To install the latest development version directly from GitHub in editable mode:

.. code-block:: bash

   git clone https://github.com/uiuc-iml/openvsf.git
   cd openvsf
   pip install -e .

**Troubleshooting**

This section provides solutions to some common issues during standard pip installation process.

1. Potential NumPy 2.0 Conflict

   OpenVSF 0.1.0 requires NumPy >= 2.0 and uses its updated `.npz` load/save functionality. All point VSF models are saved in this format.

   This is known to have conflict with `open3d <= 0.18.0` and `klampt <= 0.8.0`.  
   Please install dependencies via `pip` or `pyproject.toml` to ensure compatibility.

2. Cannot install ``mesh2sdf`` due to build errors

   If you encounter the following error during installation:

   If you see an error like:

   .. code-block:: bash

      RuntimeError: Unsupported compiler -- at least C++11 support is needed!

   it means a C++11-compatible compiler is missing. On Ubuntu, install it with:

   .. code-block:: bash

      sudo apt-get install g++
   