/*******************************************************************************
* drive_simple.c
*
* 
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <getopt.h>

#include <lcm/lcm.h>
#include "../lcmtypes/mbot_encoder_t.h"
#include "../lcmtypes/simple_motor_command_t.h"

#include <rc/start_stop.h>
#include <rc/encoder_eqep.h>
#include <rc/time.h>
#include <rc/motor.h>

//LCM
lcm_t * lcm;
#define MBOT_ENCODER_CHANNEL                "MBOT_ENCODERS"
#define MBOT_MOTOR_COMMAND_SIMPLE_CHANNEL   "MBOT_MOTOR_COMMAND_SIMPLE"

//global watchdog_timer to cut off motors if no lcm messages recieved
float watchdog_timer;

//functions
void publish_encoder_msg();
void print_answers();


int mot_l_pol;
int mot_r_pol;
int enc_l_pol;
int enc_r_pol;

mbot_encoder_t encoder_msg_old;



/*******************************************************************************
*  simple_motor_command_handler()
*
*  sets motor PWMS from incoming lcm message
*
*******************************************************************************/
//////////////////////////////////////////////////////////////////////////////
/// TODO: Create a handler that receives lcm message simple_motor_command_t and
/// sets motor PWM according to the recieved message.
/// command the motor using the command: rc_motor_set(channel, polarity * pwm);
/// for now the pwm value should be proportional to the velocity you send, 
/// the sign of the velocity should indicate direction, and angular velocity 
//  indicates turning rate. 
//////////////////////////////////////////////////////////////////////////////
// void simple_motor_command_handler...

static void simple_motor_command_handler(const lcm_recv_buf_t *rbuf, const char * channel, 
         const simple_motor_command_t* cmd, void * user){

    float FWD_PWM_CMD = 0.3;
    float TURN_PWM_CMD = 0.3;

    float l_pwm = cmd->forward_velocity * FWD_PWM_CMD - cmd->angular_velocity * TURN_PWM_CMD;
    float r_pwm = cmd->forward_velocity * FWD_PWM_CMD + cmd->angular_velocity * TURN_PWM_CMD;

    rc_motor_set(1, mot_l_pol * l_pwm);
    rc_motor_set(2, mot_r_pol * r_pwm);

}
/*******************************************************************************
* int main() 
*
*******************************************************************************/
int main(int argc, char *argv[]){
    //check args
    if( argc != 5 ) {
        printf("Usage: ./drive_simple {left motor polarity} {right motor polarity} {left encoder polarity} {right encoder polarity}\n");
        printf("Example: ./drive_simple -1 1 -1 1\n");
        return 0;
    }
    
    mot_l_pol = atoi(argv[1]);
    mot_r_pol = atoi(argv[2]);
    enc_l_pol = atoi(argv[3]);
    enc_r_pol = atoi(argv[4]);

    if( ((mot_l_pol != 1)&(mot_l_pol != -1)) |
        ((mot_r_pol != 1)&(mot_r_pol != -1)) |
        ((enc_l_pol != 1)&(enc_l_pol != -1)) |
        ((enc_r_pol != 1)&(enc_r_pol != -1))){
        printf("Usage: polarities must be -1 or 1\n");
        return 0;
      }

	// make sure another instance isn't running
    if(rc_kill_existing_process(2.0)<-2) return -1;

	// start signal handler so we can exit cleanly
    if(rc_enable_signal_handler()==-1){
        fprintf(stderr,"ERROR: failed to start signal handler\n");
        return -1;
    }

	if(rc_motor_init()<0){
        fprintf(stderr,"ERROR: failed to initialze motors\n");
        return -1;
    }

    lcm = lcm_create("udpm://239.255.76.67:7667?ttl=1");

    // make PID file to indicate your project is running
	// due to the check made on the call to rc_kill_existing_process() above
	// we can be fairly confident there is no PID file already and we can
	// make our own safely.
	// rc_make_pid_file();

	// done initializing so set state to RUNNING
    rc_encoder_eqep_init();
	rc_set_state(RUNNING);
    
    // Subscribe to custom channel
    simple_motor_command_t_subscribe(lcm, "MBOT_MOTOR_COMMAND_SIMPLE", &simple_motor_command_handler, NULL);

    watchdog_timer = 0.0;
    printf("Running...\n");

    encoder_msg_old.utime = rc_nanos_since_epoch();
    encoder_msg_old.leftticks = enc_l_pol * rc_encoder_eqep_read(1);
    encoder_msg_old.rightticks = enc_r_pol * rc_encoder_eqep_read(2);

    while(rc_get_state()==RUNNING){
        lcm_handle(lcm);
        watchdog_timer += 0.01;
        if(watchdog_timer >= 0.25)
        {
            rc_motor_set(1,0.0);
            rc_motor_set(2,0.0);
            printf("timeout...\r");
        }
        watchdog_timer = 0;
        publish_encoder_msg();
		// define a timeout (for erroring out) and the delay time
        lcm_handle_timeout(lcm, 1);
        rc_nanosleep(1E9 / 100); //handle at 10Hz
	}
    rc_motor_cleanup();
    rc_encoder_eqep_cleanup();
    lcm_destroy(lcm);
	// rc_remove_pid_file();   // remove pid file LAST

    print_answers();
	return 0;
}




/*******************************************************************************
* void publish_encoder_msg()
*
* publishes LCM message of encoder reading
* 
*******************************************************************************/
void publish_encoder_msg(){
    //////////////////////////////////////////////////////////////////////////////
    /// TODO: update this fuction by calculating and printing the forward speeds(v) 
    ///     and angular speeds (w).
    //////////////////////////////////////////////////////////////////////////////
    mbot_encoder_t encoder_msg;
    encoder_msg.utime = rc_nanos_since_epoch();
    encoder_msg.leftticks = enc_l_pol * rc_encoder_eqep_read(1);
    encoder_msg.rightticks = enc_r_pol * rc_encoder_eqep_read(2);
    encoder_msg.left_delta = encoder_msg.leftticks - encoder_msg_old.leftticks;
    encoder_msg.right_delta = encoder_msg.rightticks - encoder_msg_old.rightticks;

    float delta_t = (encoder_msg.utime - encoder_msg_old.utime) * (1e-9);

    float circ_r = 1.6916268e-4;

    float delta_s = ((encoder_msg.left_delta + encoder_msg.right_delta) / 2 ) * circ_r;

    float delta_theta = ((encoder_msg.right_delta - encoder_msg.left_delta) / 0.11) * circ_r;

    float v = delta_s / delta_t;
    float w = delta_theta / delta_t;

    printf(" ENC: %lld | %lld  - v: %6.3f | w: %6.3f \r", encoder_msg.leftticks, encoder_msg.rightticks, v, w);
    mbot_encoder_t_publish(lcm, MBOT_ENCODER_CHANNEL, &encoder_msg);
    encoder_msg_old = encoder_msg;
}

