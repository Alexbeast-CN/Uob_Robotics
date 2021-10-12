# Daoming Chen's report

## Week 3

### Seminars: Robotics challenges and opportunities in the nuclear sector -- Prof. Tom B Scott

Robots that work in the nuclear field is called a hot robot. The reason for letting robots work in the nuclear field is less cost, more output, reduce the risk for humans and save time. Many types of robots are employed here to do tasks like nuclear waste store scanning, hot spot detect and 3D map the hazard area. Lidar SLAM technology is significantly used in atomic robots. Many of the projects are to use robots to provide 3D point cloud maps about a hazard area. Then find where the hot spot region is based on the radiation map. 

### Seminars: Robots with simulation-based internal models -- Alan Winfield

An internal model is a centralized or distributed mechanism to simulate itself and the environment inside itself. The internal model is used for a robot to control itself accurately and precisely. The picture below shows the flowchart of the consequence engine. Experiments like avoiding collision using the consequence engine have successfully proved some degree of intelligence of this framework. Moreover, it can achieve some ethical tasks like avoiding people to encounter dangerous. Another application of the consequence engine is a chatbot, like let the robot tell stories to each other or explain themselves.

![ ](week3/1.png)

### Paper: LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping -- Tixiao Shan

This paper proposed a Lidar SLAM framework called LIO-SAM based on Lidar Odometry and Mapping (LOAM) framework. Using factor map as its core system structure, LIO-SAM makes the fusion of absolute measurement available to seamlessly adjust the robot's trajectory. The keyframe selection method helped reduce computation, which makes the framework faster than others. LIO-SAM can provide more accurate maps during the tests with the sliding window method and the scan match method used during Lidar scanning. When comparing LIO-SAM with other SLAM frameworks by testing them in 5 datasets, LIO-SAM can always prove itself to be the best, and that makes it stand out.