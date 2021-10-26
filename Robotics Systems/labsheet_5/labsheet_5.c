// Library Includes

#include <stdio.h>                   
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/distance_sensor.h> 
#include <webots/position_sensor.h>
#include <math.h>

// 全局定义
#define TIME_STEP       32     //时间常数的定义
#define TIME_90_degree  750
#define TIME_180_degree 1500

#define BLACK 550              //黑白两色的阈值定义
#define WHITE 800
#define Max_speed 1.5

#define NB_PS          8       //避障传感器的定义
#define NB_LEDS        8       //LED灯的定义
#define NB_GROUND_SENS 3       //巡线传感器的定义
#define GS_LEFT        0
#define GS_CENTER      1
#define GS_RIGHT       2

#define WHEEL_RADIUM        36.5
#define WhEEL_DISTANCE_HALF 52

#define OBS   120
#define OBS_E 30

// STATE的定义
#define STATE_INITIAL     0
#define STATE_FIND_LINE   1
#define STATE_FOUND_LINE  2
#define STATE_GO_HOME     3
#define STATE_HOME_ROTATE 4
#define STATE_OBS_R       5
#define STATE_OBS_L       6


// 定义里程计结构体
struct Kinematics_s {

  // Left wheel
  float l_last_angle;  // angular position of wheel in last loop()
  float l_delta_angle; // calucated difference in wheel angle.

  // Right wheel
  float r_last_angle;  // angular position of wheel in last loop()
  float r_delta_angle; // calucated difference in wheel angle.

  float x;             // Global x position of robot
  float y;             // Global y position of robot
  float th;            // Global theta rotation of robot.

}pose;

//各个传感器读取值的函数声明以及赋初值
// LEDs
WbDeviceTag led[NB_LEDS];
// Motors
WbDeviceTag left_motor, right_motor;
// Ground sensors
WbDeviceTag gs[NB_GROUND_SENS]; 
float gs_value[NB_GROUND_SENS] = {0, 0, 0};
// Proximity Sensors
WbDeviceTag ps[NB_PS];
float ps_value[NB_PS] = {0, 0, 0, 0, 0, 0, 0, 0};
// Position Sensors (encoders)
WbDeviceTag left_position_sensor, right_position_sensor;

int state;

//函数调用声明
double ElineE();
float dis_error();
void delay_ms( float ms );

// movement function
void moving_forwards();
void stop_moving();
void rotate_right();
void rotate_left();

// 里程计函数
void updateKinematics();
struct Kinematics_s pose;

// STATE函数声明
int init_state();
int join_line();
int follow_line(double e_line);
int go_home();
int home_rotate();
int obs_onr(float e_obs);
int obs_onl(float e_obs);


// One time setup, at beginning of simulation
void setup( );
void loop( );

// 主函数
int main(void) {
  wb_robot_init();
  setup();
  // Run the simulator forever.
  while( wb_robot_step(TIME_STEP) != -1 ) {
    loop();
  }
  return 0;
}

void setup() {

  char name[20];
  int i;

  state = STATE_INITIAL;

  // Setup LEDs
  for (i = 0; i < NB_LEDS; i++) {
    sprintf(name, "led%d", i);
    led[i] = wb_robot_get_device(name); 
  }
  
  // Setup Ground Sensors
  for (i = 0; i < NB_GROUND_SENS; i++) {
    sprintf(name, "gs%d", i);
    gs[i] = wb_robot_get_device(name); 
    wb_distance_sensor_enable(gs[i], TIME_STEP);
  }
  
  // Setup motors
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);
  
  // get a handler to the position sensors and enable them.
  left_position_sensor = wb_robot_get_device("left wheel sensor");
  right_position_sensor = wb_robot_get_device("right wheel sensor");
  wb_position_sensor_enable(left_position_sensor, TIME_STEP);
  wb_position_sensor_enable(right_position_sensor, TIME_STEP);

  //对结构体数据赋初值
  wb_robot_step(TIME_STEP);
  wb_position_sensor_get_value(left_position_sensor);
  wb_position_sensor_get_value(right_position_sensor);
  pose.l_last_angle = 6.59;
  pose.l_delta_angle = 0; 
  pose.r_last_angle = 6.60;  
  pose.r_delta_angle = 0; 
  pose.x = 0;             
  pose.y = 0;            
  pose.th = M_PI/2;          

  // Setup proximity sensors
  for (i = 0; i < 8; i++) {
    sprintf(name, "ps%d", i);
    ps[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(ps[i], TIME_STEP);
  }
  
 
}

void loop() {
  // Report current time.
  printf("Loop at %.2f (secs)\n", wb_robot_get_time() );

  updateKinematics();

  // Get latest ground sensor readings
  gs_value[0] = wb_distance_sensor_get_value(gs[0]);
  gs_value[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_value[2] = wb_distance_sensor_get_value(gs[2]);
  // Report ground sensor values
  // printf("Ground_Sensor: GS0: %.2f, GS1: %.2f, GS2: %.2f\n", gs_value[0], gs_value[1], gs_value[2]);

  for(int i = 0; i < NB_PS; i++)
  {
    ps_value[i] = wb_distance_sensor_get_value(ps[i]);
  }
  // printf("PS[0]: %.2f, PS[5]: %.2f, PS[6]: %.2f, PS[7]: %.2f\n", ps_value[0], ps_value[5], ps_value[6],  ps_value[7]);
  printf("state: %d\n", state);

  if(state == 0)
  {
    state = init_state();
  }
  else if(state == 1)
  {
    state = join_line();
  }
  else if(state ==2)
  {
    state = follow_line(ElineE());
  }
  else if(state ==3)
  {
    state = go_home();
  }
  else if(state ==4)
  {
    state = home_rotate();
  }
  else if(state ==5)
  {
    state = obs_onr(dis_error());
  }
  else if(state ==6)
  {
    state = obs_onl(dis_error());
  }
  
  delay_ms( 200 );
 
}

// 函数区
double ElineE() {
  double e_line;
  // Read ground sensor, store result
  gs_value[0] = wb_distance_sensor_get_value(gs[0]);
  gs_value[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_value[2] = wb_distance_sensor_get_value(gs[2]);
  // Sum ground sensor activation
  double sum_gs;
 sum_gs = gs_value[0] + gs_value[1] + gs_value[2];
  // Normalise individual sensor readings
  double w_left;
  w_left = gs_value[0] + (gs_value[1] * 0.5);
  double w_right;
  w_right = (gs_value[1]*0.5) + gs_value[2];
  // Calculated error signal
  e_line = (w_left - w_right)/sum_gs;
  // Return result
  // printf("e_line =  %.2f \n", e_line );
  return e_line;
}

float dis_error()
{
  // Same number of weigths as sensors
  float weights[NB_PS] = {0.3, 0.2, 0.1, -0.5, 0.5, -0.1, -0.2, -0.3};
  float e_obs;

  // Set initial value.
  e_obs = 0.0;
  for( int i = 0; i < NB_PS; i++ )
   {
      // Simple summation of weighted sensor values.
      e_obs = e_obs + ( ps_value[i] * weights[i]);
  }
  printf("e_obs = %.2f\n",e_obs);
  return e_obs;
}

void delay_ms( float ms ) {
  float millis_now;
  float millis_future;
  
  millis_now = wb_robot_get_time() * 1000.0;
  millis_future = millis_now + ms;
  
  // Wait for the elapsed time to occur
  // Note, steps simulation, so blocks
  // any further updates the rest of the code.
  while( millis_now < millis_future ) {
    millis_now = wb_robot_get_time() * 1000.0;
    wb_robot_step( TIME_STEP );
  } 
  
  return;
}

// movement functions
void moving_forwards()
{
  wb_motor_set_velocity(left_motor, 1);
  wb_motor_set_velocity(right_motor, 1);
}

void stop_moving()
{
  wb_motor_set_velocity(left_motor, 0);
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

// 巡线state函数组
int init_state()
{
  stop_moving();
  delay_ms(1000);
  return STATE_FIND_LINE;
}

int join_line()
{
    if (gs_value[0]>=WHITE && gs_value[2]>=WHITE )
    {
      moving_forwards();
      return STATE_FIND_LINE;
    }
    else
    {
      moving_forwards();
      delay_ms(500);
      rotate_left();
      delay_ms(300);
      return STATE_FOUND_LINE;

    }
}

int follow_line(double e_line)
{
    static int j = 0;
    static int k = 0;

  if (ps_value[0]>OBS)
    {
        rotate_left();
        delay_ms(375);
        return STATE_OBS_R;
    }
    else if (ps_value[7]>OBS || ps_value[6]>OBS)
    {
        rotate_right();
        delay_ms(50);
        return STATE_OBS_L;
    }
 
  if(e_line>-0.01 && e_line<0.01)
  {
    if (gs_value[0] + gs_value[1] + gs_value[2] < 1000 )
    {
        j = 0;
        rotate_right();
        delay_ms(100);
    }
    else if(gs_value[0] + gs_value[1] + gs_value[2] > 2530)
    {
        if(j/15)
        {
          rotate_left();
          delay_ms(TIME_180_degree);
          j = 0;
          k++;
        }
        else
        {
          moving_forwards();
          j++;
        }
    }
    else
    {
      moving_forwards();
    }
  }
 else if(e_line!=0)
 {
      j = 0;
      double turn_velocity;
      turn_velocity = 5;  // What is a sensible maximum speed?
      turn_velocity = turn_velocity * e_line;
       // Set motor values.
      wb_motor_set_velocity(right_motor, 1 - turn_velocity);
      wb_motor_set_velocity(left_motor, 1 + turn_velocity);
  }
  if(k == 2)
  { 
    return STATE_GO_HOME;
  }

  return STATE_FOUND_LINE;
}

int go_home()
{

  stop_moving();
  delay_ms(200);
  float tan_angle = atan(pose.y/pose.x);
  float theta = tan_angle + M_PI;
  printf("x的坐标: %.2f，y的坐标:%.2f\n", pose.x, pose.y);
  printf("返程角度是：%.2f\n", theta);

  float e_theta = theta - pose.th;
  printf("pose.th: %.2f\n", pose.th);
  printf("e_theta: %.2f\n", e_theta);

  if(e_theta>-0.01 && e_theta<0.01)
  {
    moving_forwards();
    delay_ms(200);
  }
  else if(e_theta!=0)
  {
    double turn_velocity;
    turn_velocity = 1;  // What is a sensible maximum speed?
    turn_velocity = turn_velocity * e_theta;
    // Set motor values.
    wb_motor_set_velocity(right_motor, 0 + turn_velocity);
    wb_motor_set_velocity(left_motor, 0 - turn_velocity);
  }

  if(pose.x<155.96 && pose.x>155.9 && pose.y<262.01 && pose.y>259.9 )
  {
    return STATE_HOME_ROTATE;
  } 
  
  return STATE_GO_HOME;
}

int home_rotate()
{
  float target_theta = (M_PI/2) - pose.th;
  printf("pose.th: %.2f, 误差角度：%.2f\n", pose.th, target_theta);
    
  if(target_theta>-0.01 && target_theta<0.01)
  { 
    stop_moving();
    delay_ms(2000);
    return STATE_INITIAL;
  }
  else if(target_theta !=0)
  {
    double turn_velocity;
    turn_velocity = 1;  // What is a sensible maximum speed?
    turn_velocity = turn_velocity * target_theta;
    // Set motor values.
    wb_motor_set_velocity(right_motor, 0 + turn_velocity);
    wb_motor_set_velocity(left_motor, 0 - turn_velocity);
  }
  return STATE_HOME_ROTATE;
}

int obs_onr(float e_obs)
{
  static int count_r = 0;
  if (count_r > 20)
  {
    if (gs_value[0]<500||gs_value[2]<500)
      {
        count_r = 0;
        return STATE_FOUND_LINE;
      }
  }

  // Determine a proportional rotation speed
  float turn_velocity;
  turn_velocity = 0.1;  // What is a sensible maximum speed?
  turn_velocity = turn_velocity * (e_obs - OBS_E);

  if (turn_velocity > 6.2 - Max_speed)
    turn_velocity = 6.2 - Max_speed;
  else if (turn_velocity < -6.2 - Max_speed)
    turn_velocity = -6.2 - Max_speed;

  printf("vr = %.2f\n",turn_velocity);
  // Set motor values.
  wb_motor_set_velocity(right_motor, Max_speed + turn_velocity);
  wb_motor_set_velocity(left_motor, Max_speed - turn_velocity);
  count_r++;
  
  return STATE_OBS_R;
}

int obs_onl(float e_obs)
{
  static int count_l = 0;
  if (gs_value[0]<500 || gs_value[1]<500 || gs_value[2]<500)
  { 
    return STATE_FOUND_LINE;
  }

  // Determine a proportional rotation speed
  float turn_velocity;
  turn_velocity = 0.1;  // What is a sensible maximum speed?
  turn_velocity = turn_velocity * (e_obs + OBS_E);

  if (turn_velocity > 6.2 -Max_speed)
    turn_velocity = 6.2 - Max_speed;
  else if (turn_velocity < -6.2 + Max_speed)
    turn_velocity = -6.2 + Max_speed;

  printf("vl = %2f\n",turn_velocity);
  // Set motor values.
  wb_motor_set_velocity(right_motor, Max_speed + turn_velocity);
  wb_motor_set_velocity(left_motor, Max_speed - turn_velocity);
  count_l++;
  printf("计数次数：%d\n",count_l);
  return STATE_OBS_L;
}

void updateKinematics() {

  // Get current wheel angular positions
  float Current_left = wb_position_sensor_get_value(left_position_sensor);
  float Current_right = wb_position_sensor_get_value(right_position_sensor);
  //printf("Current_Left: %.2f, Current_Right: %.2f\n", Current_left, Current_right);
  // Use current and last wheel angular positions
  // to determine l_delta_angle and r_delta_angle
  pose.l_delta_angle = Current_left - pose.l_last_angle;
  pose.r_delta_angle = Current_right - pose.r_last_angle;
  //printf("Delta_left: %.2f, Delta_right: %.2f\n", pose.l_delta_angle, pose.r_delta_angle);
  // Since current value has been used, save the current
  // left and right angles into  l_last_angle and 
  // r_last_angle for the next iteration.
  pose.l_last_angle = Current_left;
  pose.r_last_angle = Current_right;
  // Apply delta_angle within the kinematic equations provided.
  // 1) Work out forward contribution of motion
  // 2) Work out theta contribution of rotation
  float forward_contribution = ((WHEEL_RADIUM * pose.l_delta_angle)/2) + ((WHEEL_RADIUM * pose.r_delta_angle)/2);
  float theta_contribution = (pose.l_delta_angle-pose.r_delta_angle)* (0.5*(WHEEL_RADIUM)/WhEEL_DISTANCE_HALF);
  //printf("for_contri: %.2f, theta_contri: %.2f\n", forward_contribution,theta_contribution);
  // Update the X, Y and th values.
  // You can reference the struct in global scope
  pose.x  = pose.x + (forward_contribution * cos(pose.th));
  pose.y  = pose.y + (forward_contribution * sin(pose.th));
  //向右或者向左的情况

  if(pose.r_delta_angle<0 || pose.l_delta_angle<0)
  {
    pose.th = pose.th - theta_contribution;
  }
  else
  {
    pose.th = pose.th + 0;
  }

  if(pose.th > (2*M_PI))
  {
    pose.th = pose.th - (2*M_PI);
  }
  printf("小车横坐标X为 %.2lf，小车纵坐标Y为 %.2lf，小车的转角为 %.2lf\n", pose.x, pose.y, pose.th);
  
}

