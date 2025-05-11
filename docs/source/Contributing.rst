Contributing to and Extending OpenVSF
=====================================

We welcome contributions from the community and are excited to grow OpenVSF with your help.
This section outlines general guidelines and best practices to ensure consistency and maintainability across the codebase.


General Contribution Guidelines
--------------------------------


1. **Bug Fixes**

    - If you encounter broken or misbehaving functionality, feel free to submit a fix.
    - Clearly describe the issue being addressed and include test cases if possible.

2. **New Tools**
    
    - You may contribute utilities such as new visualization tools (e.g., for visualizing VSF deformation during simulation).
    - New tools should follow the structure and style of existing ones and integrate cleanly with the system.

3. **Extending Sensors, Simulation, and Estimation Modules**

    - The OpenVSF codebase is designed to be extensible, supporting the integration of new tactile sensors, simulation processes, and estimation algorithms.
    - The core development team is actively working on expanding these components, and community contributions are encouraged.
    - New components should inherit from the appropriate base abstraction and implement all required methods. For example, a new sensor should inherit from `BaseSensor`` and implement `predict()` to simulate tactile signals.
    - Adhering to these interfaces ensures seamless integration with the existing simulation and estimation pipelines.

4. **New Modules**

    - The core module structure is expected to remain stable.
    - Any proposed architectural changes should be discussed with the whole development team.
