
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
#define Max_speed 1

// rotate angle
#define R_pi_2 1400
#define R_pi 2800

WbDeviceTag gs[NB_GROUND_SENS]; /* ground sensors */
unsigned short gs_value[NB_GROUND_SENS] = {0, 0, 0};

// Motors
WbDeviceTag left_motor, right_motor;

// LEDs
#define NB_LEDS 8
WbDeviceTag led[NB_LEDS];

// Proximity Sensors
#define NB_PS 8
WbDeviceTag distance_sensor[NB_PS];

// Position Sensors (encoders)
WbDeviceTag left_position_sensor, right_position_sensor;


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
void moving_backwards();
void stop_moving();
void rotate_right();
void rotate_left();
void turn_left(float e_line);
void turn_right(float e_line);


// A function to return an error signal representative
// of the line placement under the ground sensor.
float getLineError();

// Robot mode functions
bool mode_chose();
int join_line(int i);
void follow_line(float e_line);



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
    distance_sensor[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(distance_sensor[i], TIME_STEP);
  }
  
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
  
  // Report current time.
  printf("Loop at %.2f (secs)\n", wb_robot_get_time() );
  
  // Get latest ground sensor readings
  gs_value[0] = wb_distance_sensor_get_value(gs[0]);
  gs_value[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_value[2] = wb_distance_sensor_get_value(gs[2]);
  
  // Report ground sensor values
  printf("Ground sensor values: \n");
  printf(" 0: %d\n", gs_value[0] );
  printf(" 0: %d\n", gs_value[1] );
  printf(" 0: %d\n\n", gs_value[2] );

  // When all 3 gs are black
  if (gs_value[0] + gs_value[1] + gs_value[2] < 1000)
  {
    moving_forwards();
    delay_ms(600);
    rotate_right();
    delay_ms(R_pi_2);
  }

  // A counter for turning mode
  static int i = 0;
  if (mode_chose())
  {
    follow_line(getLineError());
    i = 0;
  }
  else
  {
    i = join_line(i);
  }

  // Call a delay function
  delay_ms( 200 );
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

void moving_backwards()
{
    wb_motor_set_velocity(left_motor, -Max_speed);
    wb_motor_set_velocity(right_motor, -Max_speed);
}

void stop_moving()
{
    wb_motor_set_velocity(left_motor, 0);
    wb_motor_set_velocity(right_motor, 0);
}

void rotate_right()
{
    wb_motor_set_velocity(left_motor, Max_speed);
    wb_motor_set_velocity(right_motor, -Max_speed);
}

void rotate_left()
{
    wb_motor_set_velocity(left_motor, -Max_speed);
    wb_motor_set_velocity(right_motor, Max_speed);
}


// A function to return an error signal representative
// of the line placement under the ground sensor.
float getLineError() {
  float e_line;

  // Read ground sensor, store result
  float gs_valuef[3];
  gs_valuef[0] = wb_distance_sensor_get_value(gs[0]);
  gs_valuef[1] = wb_distance_sensor_get_value(gs[1]);  
  gs_valuef[2] = wb_distance_sensor_get_value(gs[2]);

  // Sum ground sensor activation
  float sum_gs ;
  sum_gs = gs_valuef[0] + gs_valuef[1] + gs_valuef[2];

  // Normalise individual sensor readings
  gs_valuef[0] = gs_valuef[0]/sum_gs;
  gs_valuef[1] = gs_valuef[1]/sum_gs;
  gs_valuef[2] = gs_valuef[2]/sum_gs;

  // Calculated error signal
  float w_left;
  w_left = gs_valuef[0] + (gs_valuef[1] * 0.5);
  float w_right;
  w_right = (gs_valuef[1]*0.5) + gs_valuef[2];
  e_line = w_left - w_right;

  // Return result
  printf("e_line = %f\n",e_line);
  return e_line;
}


// follow line mode
void follow_line(float e_line)
{

  
  if(e_line > -0.01 && e_line < 0.01)
  {
    moving_forwards();
  }
  else
  {
    printf("turning\n");
    // Determine a proportional rotation speed
    float turn_velocity;
    turn_velocity = 10;  // What is a sensible maximum speed?
    turn_velocity = turn_velocity * e_line;

    // Set motor values.
    // What does "0 -" and "0 +" achieve here?
    wb_motor_set_velocity(right_motor, Max_speed - turn_velocity);
    wb_motor_set_velocity(left_motor, Max_speed + turn_velocity);
  }
}

// join line mode

int join_line(int i)
{
  if (i/20)
  {
    rotate_right();
    delay_ms(R_pi);
    i = 0;
  }
  else
  {
    moving_forwards();
    i++;
  }
  // The robot can only moving forwards for 8 loop time in join line mode
  return i;
}

// Setup motion mode
bool mode_chose()
{
  bool state;
  // If all 3 gs are white the robot is in join line mode.
  if (gs_value[0]> 800 && gs_value[1] > 800 && gs_value[2] > 800)
    state = 0;
  // Otherwise, it is in follow line mode.
  else
    state= 1;
  return state;
}