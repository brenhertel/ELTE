# ELTE
 Implementation of Elastic-Laplacian Trajectory Editing (ELTE)

Corresponding paper can be found for free (COMING SOON), please read for method details. Accompanying video available [here](https://youtu.be/H342Y0Hxl_0).

Robot skill learning and execution in uncertain and dynamic environments is a challenging task. This paper proposes an adaptive framework that combines Learning from Demonstration (LfD), environment state prediction, and high-level decision making. Proactive adaptation prevents the need for reactive adaptation, which lags behind changes in the environment rather than anticipating them. We propose a novel LfD representation, Elastic-Laplacian Trajectory Editing (ELTE), which continuously adapts the trajectory shape to predictions of future states. Then, a high-level reactive system using an Unscented Kalman Filter (UKF) and Hidden Markov Model (HMM) prevents unsafe execution in the current state of the dynamic environment based on a discrete set of decisions. We first validate our LfD representation in simulation, then experimentally assess the entire framework using a legged mobile manipulator in 36 real-world scenarios. We show the effectiveness of the proposed framework under different dynamic changes in the environment. Our results show that the proposed framework produces robust and stable adaptive behaviors.

This repository is the implementation of Elastic-Laplacian Trajectory Editing, an LfD method for adaptation under heavy perturbations.

<img src="https://github.com/brenhertel/ELTE/blob/main/pictures/elte_box.png" alt="" width="800"/>

This repository implements the method described in the paper above using Python. Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{donald2024adaptive,
  title={An Adaptive Framework for Manipulator Skill Reproduction in Dynamic Environments},
  author={Donald, Ryan and Hertel, Brendan and Misenti, Stephen and Gu, Yan and Azadeh, Reza},
  booktitle={21st International Conference on Ubiquitous Robots (UR)},
  year={2024}
}
```
