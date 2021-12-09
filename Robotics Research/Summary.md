# RRTM Assessment 1

> Student Name: Daoming Chen
> Student number: 2132643

## Week 3

### Seminars:

#### Mon 11th October 2021. Robotics challenges and opportunities in the nuclear sector -- Prof. Tom B Scott

Robots that work in the nuclear field is called hot robot. The reason for letting robots work in the nuclear field is less cost, more output, reduced risk for humans, and saving time. Many types of robots are employed here to do tasks like nuclear waste store scanning, hot spot detect and 3D mapping the hazard area.

One challenge in this area is the automation and robustness of the robots. Since the robots are working in hazardous areas, engineers can't fix them when they lose control. Another challenge is the mapping accuracy. If the robot can't provide an accurate environment map, it's hard to complete its tasks without human assistance.

#### Mon 11th October 2021. Robots with simulation-based internal models -- Alan Winfield

An internal model is a centralized or distributed mechanism for a robot to simulate itself and its surroundings using a simulator inside a robot. The internal model is used for a robot to control itself accurately and precisely. It can train robots to do tasks like avoiding collision in a dynamic environment. 

One challenge of this field is to simulate the dynamic environment and make decisions in real-time. In the real world, the robot may have planed a path, but it needs to change its move very quickly due to the changing environment. But if the robot can change its mind, then when it faces many tasks simultaneously, the robot may keep changing its mind and fail all the missions.

### Group Reading:

#### Tues 12th Oct 2021. Paper: LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping -- Tixiao Shan

LIO-SAM is a Lidar SLAM framework using factor map as its system structure. The paper discovered that when tightly-coupled Lidar with IMU, the accuracy of SLAM is increased. The factory map and keyframe selection method can help reduce the redundant data processing volume, which allows the framework to run in real-time.

I think LIO-SAM is an excellent framework because it allows GPS to be an optional trajectory-adjust device. It's closer to real-world situations where vehicles can rely on GPS and Lidar to obtain their location and mapping the environment. However, it's still hard to handle a highly dynamic environment like a city road.

#### Tues 12th Oct 2021. Robust Autonomous Navigation of a Small-Scale Quadruped Robot in Real-World Environments -- Thomas Dudzik

> I'm the only one presented in my group this week, so here is another paper I read.

This paper proposed a small-scale quadruped robot -- MIT Mini Cheetah Vision. The challenge for this project is to build a motion and path plan system based on visual perception for low computing power small robots. But the team devised a 2-hierarchy control structure to achieve path plan and locomotion control tasks.

It's essential but challenging to equip the previous blind Mini Cheetah robot with vision. Now, the robot can make higher-level path plan decisions to increase the success of Mini Cheetah running in complex real-world terrain. However, I think the robot can perform better if a pre-train neural network is applied to the robot.

## Week 4

### Seminars:

#### Mon 18th Oct 2021. Machine Vision in the composites industry: Polarised light or deep learning -- Gary Atkinson

Machine vision and deep learning are widely used in industrial defect detection. To detect composites like carbon fiber is difficult because regular cameras can only get black and shiny pictures. A new technology using a rotating polarizer or special sensors can effectively collect detection data in the composites industry. 

One challenge is that researchers need to train the convolutional neural network (CNN) based on a large amount of good quality data. However, it's not easy to collect data that could cover all possible defects in real life. Another challenge is finding the algorithm to interpret the data. That's difficult because there are many patterns for composites' pictures.
####  Mon 18th Oct 2021. Morphological Computation: the hidden superpower of soft bodies -- Helmut Hauser

Morphological Computation studies the shape of a robot that can react intelligently to the changing environment without controllers, sensors, or brain. This subject is inspired by nature. It aims to build a robot body more like a living creature that can perform intelligently even grow and heal itself. 

One challenge is how to make those soft bodies function logically. One possible approach is to use chemical and biological elements, which can be activated in certain situations, to build the body. Another challenge is how to make the robot function in a universal environment, which is still waiting to be solved.

### Group Reading: 

#### Tues 19th Oct 2021. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM -- Carlos Campos

ORB-SLAM3 is an open-sourced visual-inertial SLAM framework. The first discovery in the paper is that using Maximum-a-Posteriori(MAP) in a tightly-integrated SLAM system can improve the accuracy. The second one is a multiple map system that can merge local maps with a global map. That allows to generate and merge a new map with the previous map after losing track of features. 

It's impressive how ORB-SLAM3 can provide such accurate trajectories results. However, they didn't show the 3D map produced during SLAM.  Because visual data is more complex than the Lidar point cloud, visual SLAM only capture feature point from pictures. Therefore, I suppose their map is not so good as Lidar SLAM can offer.

#### Tues 19th Oct 2021. Design of a Low-Cost Miniature Robot to Assist the COVID-19 Nasopharyngeal Swab Sampling -- written by Shuangyi Wang, presented by Shiqi Gao

This paper proposed a low-cost COVID nasopharyngeal swab sampling(np) robot. The robot has a 2-DOF gripper with an integrated force sensor and a 6-DOF arm with an inertial measurement unit(IMU). By limiting the motor's current and using variable-monitor software, the robot can perform gently during sampling.

I think this low-cost robot could be a solution for reducing COVID sampling risk. However, the robot has no visual sensor to help it find a nostril. Therefore, it requires human-assisted pre-aligned before it can perform sampling. I think this is a drawback that needs to be overcome in the following generation product. 

## Week 5

### Seminars:

#### Mon 25th Oct 2021. Robot learning for dexterous manipulation -- Dandan Zhang

This Seminar introduced some applications of reinforcement learning (RL) in robotics. RL is an explainable AI with fast learning speed. Explainable means the researchers can understand what causes the failure cases, therefore, reduce the training time. Another technology called deep imitation learning can make robots learn from humans' operations in real life and simulation. 

The challenge is the difficulty of setting a proper rewarding mechanism for an RL model. Because distinct from machine learning, there is no label on the data in RL. Therefore, the researchers need to learn from trial and error before their model can get a rewarding system that can lead the robot in the right way.

#### Thur 28th Oct 2021. From AI to Robot Dexterity: A Revolution Happening Now -- Prof. Nathan Lepora

The key for tactile robots is a sensor using a camera to measure the displacement of the little lamps inside the sensor to simulate tactile force. The strategy to increase robot dexterity is equipping this sensor on intelligent robots. Then the tactile robot can easily pick up fragile objects and recognize objects' shapes.

The challenge for the tactile sensor is how to model the non-linear displacement's relationship with the sense of touch. AI and lots of experimental data are used to improve the model. The challenge for robot dexterity is how to handle the tactile data. Again AI is a good tool to decide grasping force and posture.

### Group Reading: 

#### Tues 26th Oct 2021. Simultaneous Localization and Mapping: A Survey of Current Trends in Autonomous Driving -- Guillaume Bresson

Simultaneous Localization and Mapping (SLAM) has a long history of development. Recent solutions can achieve high accuracy and low drift, but there are still challenges unsolved, like real-time 3D map reconstruction, high-speed SLAM. Multi-vehicle SLAM is an excellent method for the urban driving scene. Centralized SLAM and decentralized SLAM are both possible approaches for this idea.

This SLAM review does not mention artificial intelligence (AI). However, AI can do better than humans in computer vision (CV). Therefore it's probably also better than humans in visual-SLAM. I think it could be a solution to building a semantic labeled map and help with autonomous driving decisions.

#### Tues 26th Oct 2021. Continuum Robots for Medical Applications: A Survey -- written by Jessica Burgner, presented by Zhengri Xu

The precision and continuum make the robotic system an excellent solution to perform minimally invasive surgery. Compared to traditional rigid robots, Continuum robots have more flexibility and agility, which makes continuum robots essential in minimally invasive surgery. Many medical areas like neurosurgery, otolaryngology, etc., are demanding continuum robots. 

From my point, the challenge for continuum robots is the complexity, accuracy, and computational expense of its kinemics model. Therefore, many continuum robots today are not truly continuum. To simulate the continuum kinetics model, "continuum robots" are composed of many small discrete joints. How to make continuum robots more precise and small requires future research.

## Week 7

### Seminars: 
#### Mon 8th Nov 2021. Evolutionary Swarm Robotics -- Matt Studley

Swarm robots are a group of robots working on a big project without a global controller. Robots can work distributively and parallel to each other. To achieve that, each robot has a simulator that can simulate its behavior and the dynamic environment. This strategy can provide lower cost and fault tolerance in mass production. 

The challenge is how to create a rule for robots to interact with each other and complete the project. One approach for this challenge is to simulate swarm evolution. Researchers use machine learning with a positive award mechanism to let the swarm inherit the optimal controller to achieve the evolution.

#### Mon 8th Nov 2021. Robotics Ethical, Legal, Societal -- Matt Studley

Ethical, legal, and societal are three aspects roboticists need to think of before making a robot. From an ethical view, robots should promote human wellbeing and do no harm to humans. From a legal perspective, robots should respect humans' rights. From a societal aspect, will robots cause some negative effects on the society?

Technology is neutral with no ethics. But when humans apply technology to the world, there are feedbacks on ourselves. Because the world we are living in is a dynamic system. Therefore, as a roboticist, while pursuing advanced technology, we must constantly reflect on ourselves. Can our technology cause any ethical, legal, or societal problem?

### Group Reading: 

####  Tues 9th Nov 2021. Algorithm and hardware implementation for visual perception system in autonomous vehicles: A survey -- WeijingShi

The unique hardware of self-driving cars can usually be divided into: perception system and computing system. Perception system uses camera, Lidar, radar, and sonar to capture real-world information. A computing system, including CPU, GPU, FPGA, provides a stable controlling and operating system and data processing capabilities for the vehicles. 

Although modern autonomous vehicles hardware has achieved significant technological progress, there are still many urgent problems in this field. For example, we still don't have a senor to provide 3D images. Some algorithms cannot run in real time due to the limitation of on-board chip calculation. 

#### Tues 9th Nov 2021. The SPIR: An Autonomous Underwater Robot for Bridge Pile Cleaning and Condition Assessment -- written by Khoa Le, presented by Yuehang YU

SPIR is an autonomous underwater robot that can navigate and do divers' tasks like inspecting and removing marine from wharf piles. The challenge for the robot is to keep stable while working in a turbulent underwater environment. Therefore the robot is designed to have four grasping arms to keep itself stick on the pile.

The design of the robot in the paper makes it stable when removing marine on piles. However, how can the robot move steadily to the target in turbulent water? This, I suppose, could be a big challenge to this robot, as all heavy equipment of SPIR, like the battery, is placed onshore.

## Week 8

### Seminars:

#### Mon 15th Nov 2021. Robot Skill Learning and Human Robot Share Control -- Prof. Chenguang Yang

This Seminar introduced how robots can co-work with humans and take advantage of human's intelligence and robot strength. By applying reinforcement learning, robots can imitate human motion and force from teaching. The teaching method can be either by hand or through a joystick, and the force loaded on the manipulator can be feedback as an impedance to the operator.

The challenge in this field is to give better sensory feedback to human operators, be more adaptable to different human operators, and be more general to different performing environments. That requires the system to have better sensor fusion ability and a more advanced reinforcement learning method. 

#### Thur 18th Nov 2021. Two Case Study of Aerial Robotics -- Tom Richardson

The first case study is giraffe conservation in Cameroon. It requires drones to detect giraffes in a vast nature reserve. The second case is sampling volcanic ash in Guatemala and providing a 3D mountain map. The research of Prof Richardson focused on novel sensing and novel control system for aerial robotics.

The overall challenge is how to achieve a high degree of automation, high-precision perception, and robust recognition of drones. For the first case, identifying animals with camouflage colors requires powerful object recognition capability. For the second case, drones need to perceive high-speed dynamic environments and react quickly to fly autonomously around a volcano.

## Group Reading:

#### Tues 16th Nov 2021. Software Architecture for Humanoid Robots with Versatile Task Performing Capabilities -- written by Hyeong-Seok Jeon, presented by Lin Yuan

This paper proposed a software architecture with a clearly defined structure and function set for a humanoid robot. It has been tested on a THORMANG robot both in simulation and real-world to rotate valves. This Architecture uses nodes to break down complex processes and use packages to encapsulate functions.

I don't think it's a good paper because the proposed software architecture shares the same function with Robot Operation System (ROS). However, They didn't compare it with ROS in the experiments.  They only tested if the robot could achieve the rotating valve task in their architecture. Therefore readers don't know how good it is.

#### Tues 16th Nov 2021. ORB-SLAM: A Versatile and Accurate Monocular SLAM System -- Raúl Mur-Artal

ORB-SLAM is a visual SLAM system developed from Parallel Tracking and Mapping (PTAM). It uses ORB feature points and keyframes selection and deletes method to keep the mapping procedure stable. It's the first visual-SLAM framework with loop closure, processing occlusion function. It also has a recovery mechanism that allows the SLAM procedure back on track after failure. 

The ORB-SLAM series has always maintained a high level of development. Focus on solving the pain points of SLAM. ORB-SLAM solved the problem of closed-loop detection and occlusion. ORB-SLAM2 solves the problems of real-time and large scenes. ORB-SLAM3 uses the IMU to solve the problem of trajectory drift in large scenes.

## Week 9

### Seminars:

#### Mon 22th Nov 2021. Swarm Engineering Across Scale -- Sabine Hauert

In nature, birds or ants can achieve remarkable things as a group. Inspired by that, swarm robots aim to discover organizing individual robots without a central computer. So far, machine learning and behavior tree are tools that researchers are using. The applications for swarm robots are warehouses, agriculture, etc.

The challenge is how to achieve group behavior for robots without central control. It isn't easy because we wish swarm robots intelligent enough to find a way themselves to follow commands. I think the key points are the reward mechanism and communication system. If we can teach how human is organized for projects, we may solve this problem.

#### Mon 22th Nov 2021. Computer Vision, Machine Learning -- Chollette Olisah

This Seminar introduced two computer vision (CV) and machine learning application scenarios. In the medical area, CV can be used to detect and diagnose diseases from CT images. In the agriculture area, CV is often used to help pick up the fruit in the orchard and identify species in the wild.

The challenge of CV with machine learning is the reliance on large amounts of data. For CV, low-quality pictures will pollute the data set and cause training failure. Another challenge is to design a convolutional neural network (CNN). it's like an art of math that needs trial and error.
### Group Reading:

#### Tues 23th Nov 2021. Context Dependant Iterative Parameter Optimisation for Robust RobotNavigation -- written by Adam Binch, presented by Fengrui Zhang

This paper proposed a framework for robotic navigation in agriculture with the genetic algorithm. This framework can find the suitable parameters for different navigation algorithms on different robots. The tuning process uses an iterative method with a reward mechanism depending on spatial context. The results showed robots using this framework can perform better than those with manually tuned parameters.

I think it's an improvement in agricultural robots to have this parameter optimization framework. It can save time when setting up new robots in a new environment. However, the downside is this framework keeps running simulations during tasks, which will reduce the efficiency of robots. 

#### Tues 23th Nov 2021. Autonomous Driving System based on Deep Q Learning -- Takafumi Okuyama

This paper uses deep Q learning to train autonomous vehicles driving in the lane and bypassing obstacles in a simulation environment. The car only uses one monocular camera to obtain scenes and respond to the scene with the highest Q value action. The results show Deep Q learning has high accuracy in autonomous driving.

I'm looking forward to seeing more machine learning experiments in the autonomous driving area. The complexity and dynamics of autonomous driving scenarios make it difficult for traditional methods to make decisions close to human level. Therefore, I think using CNN with a state machine could be a solution. 

## Week 10

### Seminars:

#### Mon 29th Nov 2021. Embedded Cognition for Human-Robot Interaction -- Manuel Giuliani

### Group Reading:

#### Tues 30th Nov 2021. A New Approach to Linear Filtering  and Prediction Problems -- written by R.E. Kalman, presented by Bofang Zheng

The original aim of the Kalman filter is to solve Wiener's problem of separating the interest signal from random noise in a dynamic model. To achieve that Kalman filter uses a linear dynamic model to get an estimated result. Then subtract the estimated result from the measured result to get an estimated error. Then use the error to modify the estimated result.

Now, the Kalman filter has become an algorithm that can obtain optimal estimated results from measurements that contain noise. The applications of the Kalman filter exist in aerospace, signal processing, and navigation field. It's good at estimating variables when only indirection measurement is available and fuse measurements from different sensors.

#### Tues 30th Nov 2021. Uncertain Geometry in  Robotics -- HUGH F. DURRANT-WHYTE



#### 

## Week 11

### Seminars:

#### Mon 6th Dec 2021.

### Group Reading:

#### Tues 7th Dec 2021.
