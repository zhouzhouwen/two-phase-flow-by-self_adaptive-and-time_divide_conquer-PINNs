Self-Adaptive Loss and Time Divide-and-Conquer Physics-Informed Neural Networks for Two-Phase Flow Simulations: An Integrated Approach with Advanced Interface Tracking Methods

Wen Zhou1, Shuichiro Miwa1, Koji Okamoto1

1Department of Nuclear Engineering and Management, School of Engineering, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8654, Japan

Abstract
Physics-informed neural networks (PINNs) are emerging as a promising artificial intelligence approach for solving complex two-phase flow simulations. A critical challenge in these simulations is an accurate representation of the gas-liquid interface using interface tracking methods. While numerous studies in conventional computational fluid dynamics (CFD) have addressed this issue, there remains a notable absence of research within the context of PINNs-based two-phase flow simulations. Therefore, this study aims to develop a robust and generic PINNs for two-phase flow by incorporating the governing equations with three advanced interface tracking methods—specifically, the Volume of Fluid, Level Set, and Phase-Field method—into an improved PINNs framework that has been previously proposed and validated. To further enhance the performance of the PINNs in simulating two-phase flow, the phase field constraints, residual connection and the time divide-and-conquer strategies are employed for restricting neural network training within the scope of physical laws. This self-adaptive and time divide-and-conquer (AT) PINNs then is optimized by minimizing both the residual and loss terms of partial differential equation. By incorporating the three different interface tracking methods, it efficiently handles high-order derivative terms and captures the phase interface. The case of single rising bubble in two-phase flow is simulated to validate the robustness and accuracy of the AT PINNs. The simulation's accuracy is evaluated by comparing its performance in terms of velocity, pressure, phase field, center of mass, and rising velocity with that of conventional PINNs and CFD benchmarks. The results indicate that the AT PINNs coupled with these interface tracking methods offers a satisfactory performance in simulating rising bubble phenomenon.


Keywords: physics-informed neural networks, interface tracking methods, two-phase flow, bubble dynamics

