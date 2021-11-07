
/*********************************************
 Library Includes
   You can use other C libraries
**********************************************/
#include <stdio.h>                   
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/distance_sensor.h> 
#include <webots/position_sensor.h>


/*********************************************
 Global variables.  
   Declare any global variables here to keep
   your code tidy.
**********************************************/

// simulation time step is 32ms
#define TIME_STEP 32  

// 3 IR ground sensors
#define NB_GROUND_SENS 3
#define GS_LEFT 0
#define GS_CENTER 1
#define GS_RIGHT 2

// robot speed
#define Max_speed 1.5

// Robot state 
#define STATE_INITIAL        0
#define STATE_FIND_LINE      1
#define STATE_FOUND_LINE     2
#define STATE_OBS_R          3
#define STATE_OBS_L          4


// geometry value
#define R_pi_2 1900/Max_speed
#define R_pi 3800/Max_speed
#define half_rob 800/Max_speed
#define OBS 120
#define OBS_E 30

WbDeviceTag gs[NB_GROUND_SENS]; /* ground sensors */
float gs_value[NB_GROUND_SENS] = {0, 0, 0};

// Motors
WbDeviceTag left_motor, right_motor;

// LEDs
#define NB_LEDS 8
WbDeviceTag led[NB_LEDS];

// Proximity Sensors
#define NB_PS 8
WbDeviceTag ps[NB_PS];
float ps_value[NB_PS] = {0, 0, 0, 0, 0, 0, 0, 0};


// Position Sensors (encoders)
WbDeviceTag left_position_sensor, right_position_sensor;

int state; // Use to track current FSM state.


/*********************************************
 Function prototypes.  
   Tells the C compiler what functions we have 
   written within this source file
**********************************************/

// One time setup, at beginning of simulation
void setup( );

// Controller to be called again and again.
void loop( );

// An example function to cause a delay
void delay_ms( float ms );

// movement functions
void moving_forwards();
void stop_moving();
void rotate_right(float time);
void rotate_left(float time);


// A function to return an error signal representative
// of the line placement under the ground sensor.
float getLineError();
float dis_error();

// Robot mode functions
int init_state();
int join_line();
int follow_line(float e_line);
int obs_onr(float e_obs);
int obs_onl(float e_obs);


/*********************************************
 Main loop. 
   This is the main thread of our C program.
   For this unit of study, it is recommended
   you do not alter the structure of main()
   below.  Instead, you should work from the
   functions setup() and loop().  You are 
   free to create your own useful functions.
**********************************************/
int main(void) {

  // Initialise Webots - must be done!
  wb_robot_init();

  // Code called here happens before the 
  // simulator begins running properly.
  setup();

  // Run the simulator forever.
  while( wb_robot_step(TIME_STEP) != -1 ) {
      
    // Call robot controller loop
    loop();
  }


  return 0;
}


/*********************************************
 setup()
   Use this function to setup any variables
   or robot devices prior to the start of your
   simulation.
**********************************************/
void setup() {

  // Initialise simulation devices.
  char name[20];
  int i;
  
  // Setup LEDs
  for (i = 0; i < NB_LEDS; i++) {
    sprintf(name, "led%d", i);
    led[i] = wb_robot_get_device(name); /* get a handler to the sensor */
  }
  
  // Setup Ground Sensors
  for (i = 0; i < NB_GROUND_SENS; i++) {
    sprintf(name, "gs%d", i);
    gs[i] = wb_robot_get_device(name); /* ground sensors */
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
  
  // Setup proximity sensors
  for (i = 0; i < 8; i++) {
    
    // get distance sensors 
    sprintf(name, "ps%d", i);
    ps[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(ps[i], TIME_STEP);
  }

  state = STATE_INITIAL;
  
  // You can add your own initialisation code here:
}


/*********************************************
 loop()
   Use this function to build the structure of 
   your robot controller.  Remember that we 
   expect this function to be called again and
   again (iteratively).  
**********************************************/
void loop() {
  
  extern int follow_line(float e_line);

  // Report current time.
  printf("Loop at %.2f (secs)\n", wb_robot_get_time() );

  // Get latest ground sensor readings
  gs_value[0] = wb_distance_sensor_get_value(gs[0]);
  gs_value[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_value[2] = wb_distance_sensor_get_value(gs[2]);

  //printf("%.2f\n%.2f\n%.2f\n",gs_value[0],gs_value[1],gs_value[2]);

  // Get latest distance sensor readings
  for(int i = 0; i < NB_PS; i++ ) 
  {
    // read value from sensor
    ps_value[i] =  wb_distance_sensor_get_value(ps[i]);
  }

    printf("state = %d\n",state);
    // global state determination
    if(state == 0)
        state = init_state();
    else if(state == 1)
        state = join_line();
    else if(state == 2)
        state = follow_line(getLineError());
    else if(state == 3)
        state = obs_onr(dis_error());
    else if (state == 4)
        state = obs_onl(dis_error());
          

  // Call a delay function
  delay_ms( 50 );
}


/*********************************************
 void delay_ms( float ms )
  This simple function causes the simulator to continue
  advancing for a requested amount of milliseconds.  
  Note that, the wb_robot_step is advancing at TIME_STEP
  and so this function causes an innaccurate delay.
**********************************************/
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
    wb_motor_set_velocity(left_motor, Max_speed);
    wb_motor_set_velocity(right_motor, Max_speed);
}

void stop_moving()
{
    wb_motor_set_velocity(left_motor, 0);
    wb_motor_set_velocity(right_motor, 0);
}

void rotate_right(float time)
{
    wb_motor_set_velocity(left_motor, Max_speed);
    wb_motor_set_velocity(right_motor, -Max_speed);
    delay_ms(time);
}

void rotate_left(float time )
{
    wb_motor_set_velocity(left_motor, -Max_speed);
    wb_motor_set_velocity(right_motor, Max_speed);
    delay_ms(time);
}


// A function to return an error signal representative
// of the line placement under the ground sensor.
float getLineError() 
{
  float e_line;

  // Sum ground sensor activation
  float sum_gs ;
  sum_gs = gs_value[0] + gs_value[1] + gs_value[2];

  // Calculated error signal
  float w_left;
  w_left = gs_value[0] + (gs_value[1] * 0.5);
  float w_right;
  w_right = (gs_value[1]*0.5) + gs_value[2];
  e_line = (w_left - w_right)/sum_gs;

  // Return result
  printf("e_line = %f\n",e_line);
  return e_line;
}




// join line mode

int join_line()
{
    if(gs_value[0] + gs_value[1] + gs_value[2] > 2400)
    {
        moving_forwards();
        return STATE_FIND_LINE;
    }
    else
        return STATE_FOUND_LINE;

}

int init_state()
{
    stop_moving();
    delay_ms(5000);
    return STATE_FIND_LINE;
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


// follow line mode
int follow_line(float e_line)
{
    if (ps_value[0]>OBS)
    {
        rotate_left(R_pi_2/2);
        return STATE_OBS_R;
    }
    else if (ps_value[7]>OBS)
    {
        rotate_right(R_pi_2/2);
        return STATE_OBS_L;
    }

    static int j = 0;
    if (gs_value[0] + gs_value[1] + gs_value[2] < 1000)
    {
        j = 0;
        moving_forwards();
        delay_ms(half_rob);
        rotate_left(R_pi_2);
    }
    else if(gs_value[0] + gs_value[1] + gs_value[2] > 2500)
    {
        if (j/10)
        {
            rotate_left(R_pi);
            j = 0;
        }
        else
        {
            moving_forwards();
            j++;
        }
    }
    else
    {
        j = 0;
        // Use Weight Measurement to follow line
        // Determine a proportional rotation speed
        float turn_velocity;
        turn_velocity = 9;  // What is a sensible maximum speed?
        turn_velocity = turn_velocity * e_line;

        // Set motor values.
        wb_motor_set_velocity(right_motor, Max_speed - turn_velocity);
        wb_motor_set_velocity(left_motor, Max_speed + turn_velocity);
    }
    return STATE_FOUND_LINE;
}


// When the obstacle is on the right
int obs_onr(float e_obs)
{
  static int count_r = 0;
  if (count_r > 8)
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


// Whent the obstacle is on the left
int obs_onl(float e_obs)
{
  static int count_l = 0;
  if (count_l > 8)
  {
    if (gs_value[0]<500||gs_value[2]<500)
    {
      count_l = 0;
      return STATE_FOUND_LINE;
    }
  }

  // Determine a proportional rotation speed
  float turn_velocity;
  turn_velocity = 0.1;  // What is a sensible maximum speed?
  turn_velocity = turn_velocity * (e_obs + OBS_E);

  if (turn_velocity > 6.2 -Max_speed)
    turn_velocity = 6.2 - Max_speed;
  else if (turn_velocity < -6.2 - Max_speed)
    turn_velocity = -6.2 - Max_speed;

  printf("vl = %2f\n",turn_velocity);

  // Set motor values.
  wb_motor_set_velocity(right_motor, Max_speed + turn_velocity);
  wb_motor_set_velocity(left_motor, Max_speed - turn_velocity);
  count_l++;
  return STATE_OBS_L;
}