# Daoming Chen's report

> Student Name: Daoming Chen     Student number: 2132643

## Week 3

### Seminars: Mon 11th Oct 2021. Robotics challenges and opportunities in the nuclear sector -- Prof. Tom B Scott

Robots that work in the nuclear field is called a hot robot. The reason for letting robots work in the nuclear field is less cost, more output, reduce the risk for humans and save time. Many types of robots are employed here to do tasks like nuclear waste store scanning, hot spot detect and 3D map the hazard area. Lidar SLAM technology is significantly used in atomic robots. Many of the projects are to use robots to provide 3D point cloud maps about a hazard area. Then find where the hot spot region is based on the radiation map. 

### Seminars: Mon 11th Oct 2021. Robots with simulation-based internal models -- Alan Winfield

An internal model is a centralized or distributed mechanism for a robot to simulate itself and the environment inside itself. The internal model is used for a robot to control itself accurately and precisely. The picture below shows the flowchart of the consequence engine. Experiments like avoiding collision in the dynamic environment can be done by using the consequence engine. It can achieve ethical tasks like preventing people from encountering danger. Another application of the consequence engine is a chatbot. It let robots tell stories to each other and explain themselves.

![ ](week3/1.png)
<center> Flowchart of consequence engine</center>

### Paper: LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping -- Tixiao Shan

This paper proposed a Lidar SLAM framework called LIO-SAM based on Lidar Odometry and Mapping (LOAM) framework. Using factor map as its core system structure, LIO-SAM makes the fusion of absolute measurement available to seamlessly adjust the robot's trajectory. The keyframe selection method helped reduce computation, which makes the framework faster than others. LIO-SAM can provide more accurate maps during the tests with the sliding window method and the scan match method used during Lidar scanning. When comparing LIO-SAM with other SLAM frameworks by testing them in 5 datasets, LIO-SAM can always prove itself to be the best, and that makes it stand out.

### Group Reading: Tues 12th Oct 2021. Robust Autonomous Navigation of a Small-Scale Quadruped Robot in Real-World Environments -- Thomas Dudzik

> I'm the only one presented in my group this week, so here is another paper I read.

This paper proposed a small-scale quadruped robot -- MIT Mini Cheetah Vision. A motion and path plan system that is built for small robots with limited computing power. The robot can autonomously plan a path in the real world and run with a speed of 1m/s. The figure below shows that the framework for state estimation is a two-tiered hierarchical structure. The two highers are high-level planning and low-level locomotion control. As for motion control, methods like Regularized Predictive Control (RPC) to find reaction forces, Whole-Body Impulse Control (WBIC) to control the joints, A* to plan path. 

![ ](week3/dudzi3-p8-dudzi-large.png)
<center>High-Level System Architecture</center>
