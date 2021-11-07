#! https://zhuanlan.zhihu.com/p/416903088
# Lec2. Line follower

> 目前这一阶段的代码写的与老师要求的还有一定的区别，这个慢慢改掉。之前说的 C++ 的控制器代码将会在 `labsheet3` 之前做出来。那么，目前为止 `labsheet2`要求的功能已经全部达成，代码位于 [labsheet_2_2.c](https://github.com/Alexbeast-CN/Uob_Robotics/blob/main/Robotics%20Systems/labsheet_2/labsheet_2_2.c),请各位客官放心食用。另外使用 weight-measurement 的代码位于 [labsheet_2_3.c](https://github.com/Alexbeast-CN/Uob_Robotics/blob/main/Robotics%20Systems/labsheet_2/labsheet_2_3.c)

## Lec

目前来看所有的 Lecture 都不会以授课的形式进行，而全部是 Q&A，没有特殊情况的话，以后的文章里不会再有 Lec 章节了。
## Lab

> 本阶段的笔记基于 [labsheet2](https://colab.research.google.com/github/paulodowd/EMATM0053_21_22/blob/main/WLabsheets/L2_LineFollowing.ipynb)

### Sec1. Ground sensor

为了达成巡线的目的，本阶段 `e-puck` 机器人提供了，使用基于红外反射原理的地面传感器。传感器照射到不同颜色时，会返回不同的信号。 这种传感器只在特定的情况下有用（高对比度，低干扰 的实验地面）。而3个传感器（`GS0`, `GS1`, `GS2`）两两之间的间距为 12mm，因此决定了，机器人对于黑线宽度的要求为小于等于12mm。传感器原理图如下：

![ ](pics/Webots_GSReflectance.png)

想要利用传感器达成实验目的所提供的解决方案也很简单，如下图所示：

![ ](pics/Webots_LineFeedback.png)

当传感器`GS0`返回信号为 “黑线” 时，则机器人需要直行。而如果是`GS0`/`GS2`时，机器人则需要左转或右转。

### Exercise 1: Ground Sensor (30 minutes)

Feel free to discuss these questions with your peers:

1. Is the ground sensor an **`active`** or **`passive`** sensor?  What informs your answer?

**Ans:**

It's an **active** sensor, because it can transmit a beam of infra-red light, and then measuring the intensity of the reflection.

ref: [Passive vs. Active Sensing](https://www.nrcan.gc.ca/maps-tools-publications/satellite-imagery-air-photos/remote-sensing-tutorials/introduction/passive-vs-active-sensing/14639)

2. Using the logic provided above, what would you expect to observe your robot do if it started off the line (not on the line)?

**Ans:**

If the robot is starting off the line, I expect it can turn itself to trying to get back to line. To be more specific, if the side sensor's (GS0/GS2) is black, then it will turn itself to get back to GS1 is black.
  
3. What motion would you expect to observe in the robot motion if the feedback-signal was inverted?
  - is there a case where line following can be achieved with an inverted feedback signal?

> see ans2.

4. An exceptional case not caught with the above logic would be if all three ground-sensors were activated simultaneously.  This can happen even though the black line is less than 12mm wide.  Under what condition might this occur?

**Ans:**

![ ](./pics/4.png)

5. What would be appropriate feedback responses for exceptional (non-defined) cases of the sensor activation?

> Skiped

6. If your robot was to calculate a performance score within `loop()` for line-following (a **`metric`**) as it operated:
  - what **`proprioceptive`** information could be used?
  - what **`exteroceptive`** information could be used?

**Ans:**

- The proprioceptive information should come from the encoder and imu.
- The exteroceptive information should come from the ground sensor.

### Sec2. Bang bang controller

这是一个不平滑的转动控制器，使用`if`语句编写程序。在正式编写程序之前，先写几个运动`function()`，机器人转弯的时候要注意控制速度，否则比较容易偏航。

```c

// movement functions
void moving_forwards()
{
    wb_motor_set_velocity(left_motor, 0.75);
    wb_motor_set_velocity(right_motor, 0.75);
}

void moving_backwards()
{
    wb_motor_set_velocity(left_motor, -1.0);
    wb_motor_set_velocity(right_motor, -1.0);
}

void stop_moving()
{
    wb_motor_set_velocity(left_motor, 0);
    wb_motor_set_velocity(right_motor, 0);
}

void turn_left()
{
    wb_motor_set_velocity(left_motor, 0);
    wb_motor_set_velocity(right_motor, 1);
}

void turn_right()
{
    wb_motor_set_velocity(left_motor, 1);
    wb_motor_set_velocity(right_motor, 0);
}

void rotate_right()
{
wb_motor_set_velocity(left_motor, 2.5);
wb_motor_set_velocity(right_motor, -2.5);
}

void rotate_left()
{
wb_motor_set_velocity(left_motor, -2.5);
wb_motor_set_velocity(right_motor, 2.5);
}
```

### Exercise 2: Bang-Bang Controller (4 hours)

<p align="center">
<img src="https://github.com/paulodowd/EMATM0053_21_22/blob/main/images/Webots_StartStraight.png?raw=true">
</p>

1. Is a bang-bang controller **`open-loop`** or **`closed-loop`** control?  What is the difference?

**Ans:**

It is a closed loop control. Difference see the ref below.

ref: [Difference between Open Loop & Closed Loop Control System](https://www.elprocus.com/difference-between-open-loop-closed-loop-control-system/)

2. Before you write code to command your robot to move, use the graphical user interface to position your robot ground sensor over the black line and record the ground sensor values.
  - Of the three sensors indexed `0,1,2`, which is left, centre and right?
  - What are typical values over a white surface?
  - What are typical values over a black surface?
  - How much **`variance`** is there in these readings?
  - What would be a good value to use as a **`threshold`** for the condition of your if() statements?

**Ans:**

   - gs[0,1,2] namely represents left centre and right.
   - typical white value is 1000, but values above 800 can be regarded as white.
   - typical black value is 0, but values below 300 can be regarded as black.
   - **I'm not sure how to find. Do I need to collect data?**
   - As we are in a ideal simulation environment, use above 800 and below 300 to reprsent white and black is acceptable.

3. Implement the discussed bang-bang controller logic to achieve line-following:
  - it is easiest to start the simulation with your robot already placed on a `straight` section of the line.  Check that it can follow a straight line first.
  - it is recommended you start with `slow` or `low` motor velocities.
  - check that your `feedback signal` turns your robot in the appropriate direction.
  - **help**: remember your conditional statement can use: 
    - `<`  less than
    - `<=` less than or equal to
    - `==` equal to
    - `>=` greater than or equal to
    - `>`  greater than
    - `!=` not equal to
  - **help**: what is the functional difference between the two code examples immediately below?
  
**Ans:**

Drag the robot to the line.

![ ](./pics/5.png)

Then, save the world.

Adding the code block below to the `loop()`:

  ```c
  // before the loop define the color
    #define BLACK 550
    #define WHITE 800

  // Example 1
    if (gs_value[0] <= BLACK && gs_value[1] >= WHITE && gs_value[2] >= WHITE)
    {
        turn_left();
    }
    else if (gs_value[0] >= WHITE && gs_value[1] <= BLACK && gs_value[2] >= WHITE)
    {
        moving_forwards();
    }
    else if (gs_value[0] >= WHITE && gs_value[1] <= WHITE && gs_value[2] <= BLACK)
    {
        turn_right();
    }
    // 这里我希望让机器人旋转 90°，但目前应该还缺少一个位姿传感器，或 pid 控制器，这里先用开环的方法代替
    else if (gs_value[0] <= BLACK && gs_value[1] <= BLACK && gs_value[2] <= BLACK)
    {
        rotate_halfpi();
    }
    else if (gs_value[0] >= WHITE && gs_value[1] >= WHITE && gs_value[2] >= WHITE)
    {
        moving_forwards();
        rotate_pi();
    }
  ```

![ ](pics/2.gif)

4. Does your robot conduct `turn` and `move fowards` operations seperately?  
  - Can these be integrated so that the robot does not stop moving forwards?
  - How is performance effected?
  - Write down a set of **`task requirements`** where fast forward speed might be desirable for the robotic system.
  - What is the quickest forward speed you can utilise and still achieve reliable line-following?
  - For a selected foward speed, which elements of the line following map can be completed, and which cannot?  Decide upon a set of consistent labels for the elements of the line following map.
  - Decide a discrete list of forward speed intervals to capture the **`failure modes`** you observe.  Are the different forward speeds clearly seperable in their line following performance?

**Ans:**

- To let the robot moving forward while its turn, just give the 2 wheels a diffent forward speed, like:
```c
void turn_left()
{
    wb_motor_set_velocity(left_motor, 0.5);
    wb_motor_set_velocity(right_motor, 1);
}
```

Not sure how to do other questions.

1. Start your robot off the line, and allow it to travel forward to join and follow the line.  Currently, what is the most extreme <a href="https://en.wikipedia.org/wiki/Angle_of_incidence_(optics)">angle of incidence</a> where your controller can still successfully begin line following?
  - if you were to create a results table of different angles when joining the line, how could you quantify the reliability of the controller?

**Ans:**

To answer this question, I need to figure out how to control the turning motion of the robot.

My first thought is to use a open-loop controller.

- Try1：

The theory of this open-loop controller is to one rotating angle finished in a single loop time, which about is 0.532s.

Use the pattern on the map to determine the control method. Put the robot on the rotating scale rule. Use a test program to find out how the robot rotates.

![ ](pics/6.png)

```c
// rotate function
void rotate_right()
{
wb_motor_set_velocity(left_motor, x);
wb_motor_set_velocity(right_motor, -x);
}
```

```c
// In the loop write a test function
  static int i = 0;

  if (i%2)
    rotate_right();
  else
    stop_moving();

  i++;
```

![ ](pics/3.gif)

|Angel| pi/4| pi/2| pi| 2pi|
|-    |-    |-    |-  |-   |
|x    |1.93 |F    | F |   F|

> This method works, when the rotation angel is small. When the angel goes up to pi/2, it doesn't work any more not matter how large the x value is. The reason might be the limited loop time. Therefore, I'm going fixed the rotation speed, but let the time varries.

- Try 2:

```c
#define T 0.7
#define Rotate_Speed 1.5

// in the loop
rotate_left();
delay_ms(T*1000);
stop_moving();
```

|Angel| pi/4| pi/2|pi    | 2pi|
|-    |-    |-    |-     |-   |
|T    |0.7  |1.42 |2.95  |5.9 |

>知乎不能渲染markdown表格，让我很是无语

![ ](pics/7.png)

**Now let's do the problem** 

```c
if (gs_value[0] <= BLACK && gs_value[1] >= WHITE && gs_value[2] >= WHITE)
    {
        turn_left();
    }
    else if (gs_value[0] >= WHITE && gs_value[1] <= BLACK && gs_value[2] >= WHITE)
    {
        moving_forwards();
    }
    else if (gs_value[0] >= WHITE && gs_value[1] <= WHITE && gs_value[2] <= BLACK)
    {
        turn_right();
    }
    // 这里我希望让机器人旋转90°，但目前应该还缺少一个位姿传感器，所以先用开环的方法代替
    else if (gs_value[0] <= BLACK && gs_value[1] <= BLACK && gs_value[2] <= BLACK)
    {
        rotate_right();
        // 这里的时间我一直调不好，与之前测试的不太一样，很令我困扰。
        delay_ms(T_Pi_2*1000);
        // 这里旋转之后，机器人不在线上了，我也感到很奇怪。
    }
    else if (gs_value[0] >= WHITE && gs_value[1] >= WHITE && gs_value[2] >= WHITE)
    {
        moving_forwards();
    }

```

> 做到这里我发现，其实自己被老师给出的机器人转向的条件误导了。机器人的3个灯，两两之间就可以形成条件，这样相比与中间的灯只用于控制前进多了一个微调的动作。而这个微调的条件不再取决与黑色或白色，而是直接比较3个灯返回来数值的大小。依照这个逻辑，便有了下面的代码：

```c
if (gs_value[0] >= WHITE && gs_value[1] <= BLACK && gs_value[2] >= WHITE)
    {
        moving_forwards();
    }
    // 这里我希望让机器人旋转90°，但目前应该还缺少一个位姿传感器，所以先用开环的方法代替
    else if (gs_value[0] <= BLACK && gs_value[1] <= BLACK && gs_value[2] <= BLACK)
    {
        moving_forwards();
        delay_ms(1500);
        rotate_right();
        delay_ms(T_Pi_2*1000);
    }
    else if (gs_value[0] >= WHITE && gs_value[1] >= WHITE && gs_value[2] >= WHITE)
    {
        moving_forwards();
    }
    else if (gs_value[0] + 100 < gs_value[1])
    {
        turn_left();
    }
    else if (gs_value[2] + 100 < gs_value[1])
    {
        turn_right();
    }
```

> 做到这里发现，已经达成了巡线的功能，剩下的题目就不会那么认真的做了，主要是从老师的思路中找到优化代码的方法。

6. What information about the line does the robot have when no sensors are activated?
  - When might this circumstance occur?
  - What would be an appropriate response in this condition?
  - What other information is available to the robot that might be useful?
  
> Skiped

7. Write a function to simply confirm if the robot is on a black line.  The function should report a `true` or `false` value when called.
  - is there a reason to discriminate between which of the sensors is active?  Explain your reasoning, adjust the function if necessary.
  
> Skiped

8. Write a specific function to operate the robot when it is **`initialised`** ("powered on" for the first time) off the line, so that it can successfully join a line at 90 degrees (orthogonal) to the forward direction of travel. 
  - Aligning your robot to 45 degrees (approx. `0.785` radians) in the `start box` on the provided arena floor will achieve this starting condition.
  - Decide what your robot should do when it meets an orthogonal line, or how to bias your robot behaviour away from this circumstance.
  - Create a **`global variable`** called **`STATE`** to indicate if the robot is operating to `join line` or `follow line`.  Which state should the robot initialise into when powered on? 
  - Use the `setup()` function in the template to set initial variable values, such as `STATE`.
  - Use an `if()` statement to switch the behaviour between `join line` or `follow line` depending on the global variable `STATE`.
  - Use the prior line detection function (`true` / `false`) to transition your robot between the **`join line`** behaviour, and the **`follow line`** behaviour.

> 此处利用设置模式：“加入线”与“巡线”，来优化代码。但是我选择了在loop中做这样的模式选择。

**Ans:**

When all 3 ground sensor values are above 800, it should be at `join line` state. When meeting the condition of any of those 3 sensor values are below 400, it should change to `follow line` behaviour. 

```c
// in the loop() function

// When all 3 gs are white, the robot is in join line mode.
if (gs_value[0]>800 && gs_value[1] > 800 && gs_value[2] > 800)
  bool STATE=0;
// Otherwise, it is in follow line mode.
else
  bool STATE=1;
```

### Sec3. Weighted-Measurement

<p align="center">
<img width="50%" src="https://github.com/paulodowd/EMATM0053_21_22/blob/main/images/Webots_GS_Weighted.png?raw=true">
</p>

Weight Measurement 看起来是用来衡量小车左转还是右转的最佳方案。这里我们无需确定颜色所代表的数值，只需根据这3个传感器数值之间的差，来控制小车的转动方向和转动速度的大小。具体的方法如下：

### Exercise 3: Weighted-Line Measurement

1.  Implement the weighted-line sensing as discussed above. There are still further considerations to make:
  - is your ground sensor `active low` or `active high`?   How could you invert the value of a sensor reading?  Is this necessary?
  - does the `sign` (-/+) of the error signal intuitively indicate the direction your robot should turn?  Which direction of rotation should -/+ be?
  - how can you invert the sign of the error signal?
  - **Decompose the problem**: first of all, test your robot without any forward velocity.  Is your robot able to re-centre itself over the line?  Check by manually placing your robot in different circumstances.
  - **help**: What would you expect your robot to do when it is not on a line?  Why is this?
  - **help**: The following code extract may help you to get started:

- I think it's not necessary to know since I only using simulation.
- The sign (-/+) indicate the direction, when the sign is -, the robot should turn left, otherwise, it should turn right.
- Change the position of gs[0] and gs[2] in the algorithm.

```c

void loop() {

  // Get the line error
  float e_line;
  e_line = getLineError();

  // Determine a proportional rotation speed
  float turn_velocity;
  turn_velocity = 1;  // What is a sensible maximum speed?
  turn_velocty = turn_velocity * e_line;

  // Set motor values.
  // What does "0 -" and "0 +" achieve here?
  wb_motor_set_velocity(right_motor, 0 - turn_velocity);
  wb_motor_set_velocity(left_motor, 0 + turn_velocity);
  
}

// A function to return an error signal representative
// of the line placement under the ground sensor.
float getLineError() {
  float e_line;

  // Read ground sensor, store result
  gs_value[0] = wb_distance_sensor_get_value(gs[0]);
  gs_value[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_value[2] = wb_distance_sensor_get_value(gs[2]);

  // Sum ground sensor activation
  float sum_gs;
  sum_gs = gs_value[0] + gs_value[1] + gs_value[2];

  // Normalise individual sensor readings
  gs_value[0] /= sum_gs;
  gs_value[1] /= sum_gs;
  gs_value[2] /= sum_gs;

  // Calculated error signal
  double w_left;
  w_left = gs_value[0] + (gs_value[1] * 0.5);
  double w_right;
  w_right = (gs_value[1]*0.5) + gs_value[2];
  e_line = (w_left - w_right)/sum_gs;

  // Return result
  return e_line;
}

```

2. Explore the gain value for the central sensing element (described with value `0.5` above:
  - What is the effect of alternative gain values?
  - What other formulation could be used to utilise the weighted 3 sensing elements in combination?

> I can't see any points on this prob

3. Modify your solution to (1) to implement a forward velocity with your robot.
  - How fast can your robot move forwards and still reliably follow the line?
  - Which parts of the line following map present the greatest challenge to your solution?
  
>skiped

4. Implement a mechanism to vary the foward velocity of your robot independently of the turning velocity:
  - Aim for high forward velocity on straight-line segments, and low forward velocity on sharp corners.
  - How can your robot measure it's own line following performance?
  - How can you convert this line performance measure into another `proportional control` mechanism for forward velocity?

> This is a optimization process. I will try to done this part, but it's hard to get 100% correct answer. My answer will be included in my codes.

[用于实现Q4的代码](https://github.com/Alexbeast-CN/Uob_Robotics/blob/main/Robotics%20Systems/labsheet_2/labsheet_2_3.c)