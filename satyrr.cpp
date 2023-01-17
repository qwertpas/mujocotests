/*  Copyright © 2018, Roboti LLC

    This file is licensed under the MuJoCo Resource License (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.roboti.us/resourcelicense.txt
*/

#include <GLFW/glfw3.h>
#include "mujoco/mujoco.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "stdio.h"
#include "stdlib.h"
#include <string>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <thread>
#include <cstdio>
#include <ctime>
#include "string.h"
#include <Eigen/Dense>



using namespace Eigen;

#define Hip 1
#define Knee 2
// #define M_PI 3.14159265358979323846
using namespace std;

#define q_free_NUM 6
#define q_NUM 12
#define actuator_NUM 6
#define Hip 1
#define Knee 2
#define Yaw_left 1
#define Yaw_right 2
#define SATYRR_leg  0.55
#define SATYRR_r  0.06
#define SATYRR_length  0.8

// ******* FUNCTION DECLARATIONS **************//
void SATYRR_Init(const mjModel* m, mjData* d);


//****************** INPUT PARAMS ***************//
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;
double wheel_torque = 0.0;
double yaw_set = 0.0;
double yaw_damp = 0.0;


//************** SIM PARAMS *******************//
float_t ctrl_update_freq = 1000; //Hz
mjtNum last_update = 0.0;
int torso_Pitch, torso_Roll, torso_Yaw, torso_X, torso_Z, j_hip_l, j_hip_r, j_knee_l, j_knee_r, j_wheel_l, j_wheel_r,imu_gyro,
     sh_yaw_L, sh_roll_L, sh_pitch_L, elbow_L, sh_yaw_R, sh_roll_R, sh_pitch_R, elbow_R;

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


//************************************************************* CLASSES


class SATYRR_STATE
{
    public:        
        float q_free[q_free_NUM];
        float q[6];
        float dq[6];

        float des_state_[4];
        float state_[4];
        float CoM_[2];
        float CoM_R[2];
        float CoM_L[2];

        //States of robot 
        float x = 0.0;
        float x_old = 0.0;
        float dx = 0.0;
        float pitch = 0.0;
        float pitch_old = 0.0;
        float dpitch = 0.0;
        float psi = 0.0; //yaw
        float psi_old = 0.0;
        float dpsi = 0.0;
        float dyaw = 0.0;

        //Force sensors at end effector
        float touch_L = 0.0;
        float touch_R = 0.0;

        //Desired old params
        float x_des_prev = 0.0;
        float dx_des_prev = 0.0;
        float th_H_old = 0.0;
        float dth_H_old = 0.0;
        float DCM_H = 0.0;
        float th_H = 0.0;
        float dth_H = 0.0;
 
        //Fixed Params
        const float desHip = 0.45588;
        const float mass_wheel = 0.525989*2;
        const float mass_leg = (0.904864 + 0.394882) * 2; // thigh + shin
        const float mass_torso = 4.05;
        const float mass_upper = (0.2256 + 0.58497 + 0.111225 + 0.24689) * 2; // sholder + arm + shank + shoulder_u
        const float dt = 0.001;

        //Human Parameters
        const float g = 9.81;
        float h_H=1.2;//Human CoM height (m)
        float m_H=85.0;//Human mass (kg)
        float omega_H=sqrt(g/h_H); //Human natural freq.


        SATYRR_STATE()
        {
            for(int i=0;i<q_free_NUM;i++)
            {
                q_free[i] = 0.0;
            }

            for(int i=0;i<q_NUM;i++)
            {
                q[i] = 0.0;
            }

            for(int i=0;i<4;i++)
            {
                des_state_[i] = 0.0;
                state_[i] = 0.0;
            }

            for(int i=0;i<2;i++)
            {
                CoM_R[i] = 0.0;
                CoM_L[i] = 0.0;
                CoM_[i] = 0.0;
            }

        }

    bool getCOM(double q_hip_l, double q_hip_r, double pitch)
    {
        CoM_R[0] = -.01884*cos(pitch) + .07329*sin(pitch) - SATYRR_leg*sin(q_hip_r - pitch) + SATYRR_leg*sin(q_hip_r + pitch);
        CoM_R[1] = .07329*cos(pitch) + .01884*sin(pitch) + SATYRR_leg*cos(q_hip_r+ pitch) + SATYRR_leg*cos(q_hip_r - pitch); 

        CoM_L[0] = -.01884*cos(pitch) + .07329*sin(pitch) - SATYRR_leg*sin(q_hip_l - pitch) + SATYRR_leg*sin(q_hip_l + pitch);
        CoM_L[1] = .07329*cos(pitch) + .01884*sin(pitch) + SATYRR_leg*cos(q_hip_l + pitch) + SATYRR_leg*cos(q_hip_l - pitch);

        CoM_[0] = (CoM_R[0] + CoM_L[0])/2;
        CoM_[1] = (CoM_R[1] + CoM_L[1])/2; 

        return true;
    }

    /*
    * velocityMapping() maps from human LIP pitch to some desired velocity 
    */
    void velocityMapping(float *q_H, float*q_R, float* tgt)
    {
        float x_H = q_H[0];
        float px_H = 0.0; //CoP position
        // float x_H = 0;

        //Robot Parameters
        float h_R = 0.4253; 
        float omega_R = sqrt((9.81/h_R));

        //Human Parameters
        float h_H=1.2;//Human CoM height (m)
        float m_H=93.0;//Human mass (kg)
        float omega_H=sqrt(9.81/h_H); //Human natural freq.
        float th_H = (x_H - px_H)/h_H; //assuming CoP is at zero and not moving

        float max_pitch = 20 * (3.14159/180);
        float max_des_vel_R = 1.5; // in m/s (close to the real robots max before it slips)
        float slope = max_des_vel_R/max_pitch;
        //Mapping from pitch_h -> vel_r 
        float des_dx = slope*th_H;
        float des_x = x_des_prev + des_dx*dt;

        float dth_R_des = q_H[3]*(omega_R/omega_H);

        // Set target vel
        // tgt[0] = des_x;
        tgt[0] = q_R[0]; // x_des (curr disabled)
        tgt[1] = q_H[2]; // th_des
        tgt[2] = q_R[2]; // dx_des (curr disabled)
        tgt[3] = q_H[3]; // dth_des

        //Old des x is current 
        x_des_prev = des_x;
    }

    void dcmMapping(float *q_H, float*q_R, float* tgt)
    {
        float x_H = q_H[0];
        float px_H = q_H[1];
        float th_H = q_H[2];
        float dth_H = q_H[3];

        //Manual calculation (less preferred)
        // th_H = (x_H - 0)/h_H; //assuming CoP is at zero and not moving
        // dth_H = (th_H - th_H_old)/dt;
        // dth_H = dth_H*0.1 + dth_H_old*0.9; // lpf the velocity 

        DCM_H = th_H + dth_H/omega_H; //unitless CP DCM

        //Attempting solving for x_des 
        float ddx_des = g*q_H[2];
        float dx_des = dx_des_prev + ddx_des*dt;
        float x_des = x_des_prev + dx_des*dt;

        dx_des_prev = dx_des;
        x_des_prev = x_des;

        // printf("%.2f | %.2f\r\n", x_des, dx_des);
        tgt[0] = q_R[0]; // This effectively disables any form of positon tracking 
        tgt[1] = q_R[2]; // This disables any form of velocity tracking 
        // tgt[0] = x_des; // 
        // tgt[1] = dx_des; // 
        tgt[2] = 0;
        tgt[3] = DCM_H;

        th_H_old = th_H;
        dth_H_old = dth_H;
    }

};



class SATYRR_controller
{
    public:
        float FxR;
        float applied_torq[actuator_NUM];
        float wheel_torque;
        float yaw_torq;
        
        float q_DCM[4];
        
        float joint_torq_out[2];

        //Robot params 
        float h_R = 0.4253; // full body satyrr com height 
        float m_R = 0.4297; //mass of wheel
        float M_R = 7.16; // mass of body 
        float omega_R = sqrt((9.81/h_R));
        // float K_l2cp[4] ={-27.7356, -116.4209, 0, -264.7271};
        float K_l2cp[4] ={-27.7356, -116.4209, 0, -470};

        SATYRR_controller()
        {
            for(int i=0;i<actuator_NUM;i++)
            {
                applied_torq[i] = 0.0;
            }
            FxR = 0.0;
            wheel_torque = 0.0;
            yaw_torq = 0.0;
        }

        void lqrController(float *states, float *tgt)
        {
            float K_xW = 180; 
            float K_pitch = 640;
            float K_dxW = 120;
            float K_dpitch = 70;

            // K_pitch = 470;
            // K_dpitch = 470/(9.81/0.45);

            // tgt[3] = (1.2/0.45)*tgt[3];

            FxR = K_xW *(states[0] - tgt[0]) + K_pitch*(states[1]  - tgt[1]) + K_dxW*(states[2]  - tgt[2]) + K_dpitch*(states[3]  - tgt[3]);
        }

        void l2cpController(float *states, float *tgt)
        {
            //************************************ ROBOT CART-POLE ROM ****************************/

            //Robot states
            float x_R = states[0];
            float th_R = states[1];
            float dx_R = states[2];
            float dth_R = states[3];

            //Robot ref L2CP DCM
            float L2CP_DCM_R = th_R + dth_R/omega_R; //unitless CP DCM

            //Robot DCM state transform
            q_DCM[0] = x_R; // cart position
            q_DCM[1] = dx_R; // cart velocity
            // q_DCM[0] = 0; // cart position
            // q_DCM[1] = 0; // cart velocity
            q_DCM[2] = x_R + h_R*th_R; // CoM position
            q_DCM[3] = L2CP_DCM_R; // Cart-Pole DCM

            //Testing LQR feedback controller
            float q_err[4];
            q_err[0] = q_DCM[0] - tgt[0];
            q_err[1] = q_DCM[1] - tgt[1];
            q_err[2] = q_DCM[2] - tgt[2];
            q_err[3] = q_DCM[3] - tgt[3];

            // printf("%.2f\r\n", q_err[3]);

            FxR =  -1*(K_l2cp[0]*q_err[0]+ K_l2cp[1]*q_err[1] + K_l2cp[2]*q_err[2] + K_l2cp[3]*q_err[3]);
        }
}; 

// Set height data of an existing heightfield
bool set_heightfield(mjModel* m_, int id, MatrixXd& map, double radius_x, double radius_y, double base_z)
{

    const double max_height = map.maxCoeff();

	if (max_height > 1e-5)
	{
		map = map * (1.0 / max_height); // Normalize matrix
	}

	if (id >= m_->nhfield)
	{
		// hfield does not exist
        printf("hfield no exist");
		return false;
	}
	int nele = m_->hfield_ncol[id] * m_->hfield_nrow[id];
	if (nele != map.size())
	{
		// Incorrect size
        printf("incorrect size");
        printf("model size %d", nele);
        printf("geiven sieze %d", map.size());
		return false;
	}

	int h_i = m_->hfield_adr[id];

	// HField data is row-major
	for (int r = 0; r < map.rows(); r++)
	{
		for (int c = 0; c < map.cols(); c++)
		{
			m_->hfield_data[h_i++] = map(r, c);
            printf("%f", map(r, c));
		}
	}

	if (max_height > 0.0)
	{
		m_->hfield_size[id * 4 + 2] = max_height;
	}
	if (radius_x > 0.0 && radius_y > 0.0)
	{
		m_->hfield_size[id * 4 + 0] = radius_x;
		m_->hfield_size[id * 4 + 1] = radius_y;
	}
	if (base_z > 0.0)
	{
		m_->hfield_size[id * 4 + 3] = base_z;
	}

	return true;
}




//******************* FILE PARAMS *********//
SATYRR_controller SATYRR_Cont;
SATYRR_STATE SATYRR_S;
clock_t completion_time_clock;
float des_pitch = 0;
int counter = 0;
ofstream myfile;

int disturbance_time = 0;
float tgt[4] ={0, 0, 0, 0}; // xW, theta, dxW, dtheta


int data_save_flag = 1;
string file_name = "test5_chris_DCM_Fb_Fff_No_FFMom_No_CoP_20_Hz_FeetTog_neg75dist_FextIncl";
int cnt = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation

    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}



void SATYRR_state_update(const mjModel* m, mjData* d)
{
    //q_hip_left & right
    SATYRR_S.q[0] = d->qpos[m->jnt_qposadr[j_hip_l]];
    SATYRR_S.q[1] = d->qpos[m->jnt_qposadr[j_hip_r]];
    //q_knee_left & right
    SATYRR_S.q[2] = d->qpos[m->jnt_qposadr[j_knee_l]];
    SATYRR_S.q[3] = d->qpos[m->jnt_qposadr[j_knee_r]];
    //q_hip_wheel & right
    SATYRR_S.q[4] = d->qpos[m->jnt_qposadr[j_wheel_l]];
    SATYRR_S.q[5] = d->qpos[m->jnt_qposadr[j_wheel_r]];

    //q_vel_hip_left & right
    SATYRR_S.dq[0] = d->qvel[m->jnt_dofadr[j_hip_l]];
    SATYRR_S.dq[1] = d->qvel[m->jnt_dofadr[j_hip_r]];
    //q_vel_knee_left & right
    SATYRR_S.dq[2] = d->qvel[m->jnt_dofadr[j_knee_l]];
    SATYRR_S.dq[3] = d->qvel[m->jnt_dofadr[j_knee_r]];
    //q_vel_wheel_left & right
    SATYRR_S.dq[4] = d->qvel[m->jnt_dofadr[j_wheel_l]];
    SATYRR_S.dq[5] = d->qvel[m->jnt_dofadr[j_wheel_r]];

    //body state
    SATYRR_S.x_old = SATYRR_S.x; // Set current to previous
    SATYRR_S.x = d->qpos[m->jnt_qposadr[torso_X]]; 
    SATYRR_S.dx = (SATYRR_S.x - SATYRR_S.x_old) / (1/ctrl_update_freq);

    SATYRR_S.pitch_old = SATYRR_S.pitch;    
    SATYRR_S.pitch = d->qpos[m->jnt_qposadr[torso_Pitch]];
    //SATYRR_S.dpitch = (SATYRR_S.pitch - SATYRR_S.pitch_old) / (1/ctrl_update_freq);
    SATYRR_S.dpitch = d->sensordata[1];

    SATYRR_S.psi_old = SATYRR_S.psi;
    SATYRR_S.psi = d->qpos[m->jnt_qposadr[torso_Yaw]];
    SATYRR_S.dyaw = d->sensordata[2];

    SATYRR_S.touch_L = d->sensordata[3];
    SATYRR_S.touch_R = d->sensordata[4];

    // printf("h_l = %.2f | h_R = %.2f\r\n", SATYRR_S.touch_L, SATYRR_S.touch_R);
}

void mycontroller(const mjModel* m, mjData* d)
{
    //state update
    SATYRR_state_update(m,d);
    // tgt[1] = 0.1*sin(0.001*counter);
    float wh_r = .06;
    counter++;

    float states[4] = {SATYRR_S.x, SATYRR_S.pitch, SATYRR_S.dx, SATYRR_S.dpitch};
    //Robot knee,hip joint angle positions and velocities
    float *th_ = SATYRR_S.q;
    float *dth_ = SATYRR_S.dq;

    float test_qH[4] = {0.0, 0.0, 0.0, 0.0};
    //Planner for desired states (directly modify tgt vec)
    // tgt[0] = 0; tgt[1] = 0; tgt[2] = 0; tgt[3] = 0.0;
    // SATYRR_S.velocityMapping(udp_obj.q_H, states, tgt);
    // SATYRR_S.dcmMapping(udp_obj.q_H, states, tgt);
    SATYRR_S.dcmMapping(test_qH, states, tgt);


    //Stabilization controller
    // printf("%.2f | %.2f | %.2f | %.2f\r\n", tgt[0], tgt[1], tgt[2], tgt[3]);
    // SATYRR_Cont.lqrController(states, tgt);
    SATYRR_Cont.l2cpController(states, tgt);
    float wheel_torque =  (wh_r*SATYRR_Cont.FxR)/2;

    //External contact force 
    // float x_H = udp_obj.q_H[0];
    // float px_H = udp_obj.q_H[1];
    float x_H = 0;
    float px_H = 0;
    float m_H =  SATYRR_S.m_H;
    float h_H=1.2;//Human CoM height (m)
    float omega_H = SATYRR_S.omega_H;
    float h_R = 0.4253; // full body satyrr com height 
    float m_R = 0.4297; //mass of wheel
    float M_R = 7.16; // mass of body 
    float omega_R = sqrt((9.81/h_R));


    //Disturbance applied
    int satyrr_torso_id = mj_name2id(m, mjOBJ_BODY, "torso");
    float distForce = 0;
    mjtNum wrench_ext[6] = {distForce, 0, 0, 0, 0, 0};
    mjtNum wrench_zero[6] = {0, 0, 0, 0, 0, 0};
    // if((disturbance_time>22000) && (disturbance_time < 23600)){
    //     distForce = -75.0;
    //     wrench_ext[0] = distForce;
    //     mju_copy(&(d->xfrc_applied[6*satyrr_torso_id]), wrench_ext, 6);
    // }
    // else{
    //     distForce = 0;
    //     mju_copy(&(d->xfrc_applied[6*satyrr_torso_id]), wrench_zero, 6);
    // }
    disturbance_time++;
    
    //Contact Force
    // float F_touch = -(m_H*h_H*omega_H*omega_H)/(M_R*h_R*omega_R*omega_R)*(SATYRR_S.touch_L + SATYRR_S.touch_R);
    float F_touch = -(m_H*h_H*omega_H*omega_H)/(M_R*h_R*omega_R*omega_R)*(SATYRR_S.touch_L + SATYRR_S.touch_R) + (m_H*h_H*omega_H*omega_H)/(M_R*h_R*omega_R*omega_R)*distForce;

    //Virtual Inertial Human Force
    // float K_spring = 4.94117647; //420/85kg
    // float K_spring = 7.647059; //650/85kg
    // float K_spring = 3.52941176; // 300/85
    float K_spring = 0.0;
    float Fx_H = m_H*omega_H*omega_H*(x_H - px_H); //
    float Fx_H_spring =  (m_H*omega_H*omega_H*(x_H-px_H)) - (m_H*K_spring*x_H);
    
    //Feedback Term
    // float F_fb = omega_H*omega_H*m_H*h_H*((SATYRR_S.dpitch/omega_R) - (udp_obj.q_H[3]/omega_H));
    // float F_fb =   omega_H*omega_H*m_H*h_H*((SATYRR_S.dpitch/omega_R) - (udp_obj.q_H[3]/omega_H)) +  omega_H*omega_H*m_H*h_H*(SATYRR_S.pitch -  udp_obj.q_H[2]);
    float F_fb =   omega_H*omega_H*m_H*h_H*((SATYRR_S.dpitch/omega_R) - (0/omega_H)) +  omega_H*omega_H*m_H*h_H*(SATYRR_S.pitch -  0);
    // float F_fb = 0;


    //Zerod ff term (no dyn. similarity)
    // float F_ff = 0;

    // Normal ff term 
    // float F_ff = (SATYRR_S.pitch - udp_obj.q_H[2])*omega_R*omega_R*M_R*h_R; 

    // Term with spring enabled & F_fb to human
    // float F_ff = (SATYRR_S.pitch - (Fx_H_spring/(omega_H*omega_H*m_H*h_H)))*omega_R*omega_R*M_R*h_R; 

    // Term with spring & NO F_fb to human
    // float F_ff = omega_R*omega_R*M_R*h_R*((SATYRR_S.dpitch/omega_R) - (udp_obj.q_H[3]/omega_H)) + omega_R*omega_R*M_R*h_R*(SATYRR_S.pitch - (Fx_H_spring/(omega_H*omega_H*m_H*h_H))); 
    
    // W CoP in Fx_H(NOT WORK: Cant lean off of box)
    // float F_ff = (SATYRR_S.pitch - (Fx_H/(omega_H*omega_H*m_H*h_H)))*omega_R*omega_R*M_R*h_R; 

    //New feedforward with actuated human model adjustment and no feedback
    // float F_ff =  omega_R*omega_R*M_R*h_R*((SATYRR_S.dpitch/omega_R) - (udp_obj.q_H[3]/omega_H)) +  omega_R*omega_R*M_R*h_R*(SATYRR_S.pitch -  udp_obj.q_H[2]) + omega_R*omega_R*M_R*h_R*(px_H/h_H);

    //New feedforward with actuated human model adjustment and feedback
    // float F_ff =  omega_R*omega_R*M_R*h_R*(SATYRR_S.pitch -  udp_obj.q_H[2]) + omega_R*omega_R*M_R*h_R*(px_H/h_H);

    float F_ff = omega_R*omega_R*M_R*h_R*(px_H/h_H);

    //Feedforward moment from human
    float M_H_ff = (h_H* Fx_H) * (M_R*h_R*h_R*omega_R*omega_R)/(m_H*h_H*h_H*omega_H*omega_H); // Feedforward term of human moment scaled to robot
    float F_mom_ff = M_H_ff/h_R; // Convert feedforward moment from human to robot wheel force
    F_mom_ff = 0;

    float tau_ff = (F_ff + F_mom_ff)*wh_r;
    //float tau_ff = F_ff*wh_r;
    // float tau_ff = 0;

    float F_ext = F_fb + F_touch;
    // float F_ext = F_touch;
    // float F_ext = 0;

    //Yaw Controller
    float Kp_yaw = -2;
    float Kd_yaw = -0.2;
    float yaw_torq = Kp_yaw *(SATYRR_S.psi - 0) + Kd_yaw * (SATYRR_S.dyaw - 0);

    //Hip Controller
    float Kp_hip = 500;
    float Kd_hip = 5;
    SATYRR_Cont.applied_torq[0] = Kp_hip*(th_[0] - SATYRR_S.desHip) + Kd_hip*(dth_[0] - 0); //Left hip
    SATYRR_Cont.applied_torq[3] = Kp_hip*(th_[1] - SATYRR_S.desHip) + Kd_hip*(dth_[1] - 0); //Right hip

    //Knee Controller
    float Kp_knee = 400;
    float Kd_knee = 2;
    SATYRR_Cont.applied_torq[1] = Kp_knee*(th_[2] - (-SATYRR_S.desHip*2)) + Kd_knee*(dth_[2] - 0); //Left knee
    SATYRR_Cont.applied_torq[4] = Kp_knee*(th_[3] - (-SATYRR_S.desHip*2)) + Kd_knee*(dth_[3] - 0); //Right knee

    //2D Cartesian space controller combination
    SATYRR_Cont.applied_torq[2] =  wheel_torque - yaw_torq + tau_ff; // Left wheel_torque;
    SATYRR_Cont.applied_torq[5] =  wheel_torque + yaw_torq + tau_ff; //Right wheel_torque;

    //Applied torque
    if (d->time - last_update > 1.0/ctrl_update_freq)
    {
        //Lower body control
        d->ctrl[0] = -SATYRR_Cont.applied_torq[0];
        d->ctrl[1] = -SATYRR_Cont.applied_torq[1];
        d->ctrl[2] = -SATYRR_Cont.applied_torq[2];
        d->ctrl[3] = -SATYRR_Cont.applied_torq[3];
        d->ctrl[4] = -SATYRR_Cont.applied_torq[4];
        d->ctrl[5] = -SATYRR_Cont.applied_torq[5];

        //Left shoulder yaw control
        d->ctrl[6] = 0;
        d->ctrl[7] = 0;
        //Left shoulder roll control
        d->ctrl[8] = 0;
        d->ctrl[9] = 0;
        //Left shoulder pitch control
        d->ctrl[10] = 0;
        d->ctrl[11] = 0;
        //Left elbow Control
        d->ctrl[12] = 0;
        d->ctrl[13] = 0;

        //Right shoulder yaw control
        d->ctrl[14] = 0;
        d->ctrl[15] = 0;
        //Right shoulder roll control
        d->ctrl[16] = 0;
        d->ctrl[17] = 0;
        //Right shoulder pitch control
        d->ctrl[18] = 0;
        d->ctrl[19] = 0;
        //Right elbow control
        d->ctrl[20] = 0;
        d->ctrl[21] = 0;

        //Fixed positon for pushing if needed
        // d->ctrl[10] = -0.75;
        // d->ctrl[12] = -3.14;
        // d->ctrl[18] = 0.75;
        // d->ctrl[20] = 3.14;
    }

    //Update data to send back 
    // udp_obj.Robot_Data[0] = SATYRR_S.DCM_H; // desired dcm, human
    // udp_obj.Robot_Data[1] = SATYRR_Cont.q_DCM[3]; // actual robot dcm 
    // udp_obj.Robot_Data[2] = F_ext;
    // udp_obj.Robot_Data[3] = SATYRR_S.dpitch;
    // udp_obj.Robot_Data[4] = udp_obj.q_H[3];
    // udp_obj.Robot_Data[5] = SATYRR_S.pitch;
    // udp_obj.Robot_Data[6] = udp_obj.q_H[2];

    //Data logging 
    if (data_save_flag){
        if(cnt % 5 == 0){
            myfile << d->time
            << ", " << SATYRR_S.DCM_H
            << ", " << SATYRR_Cont.q_DCM[3]
            // << ", " << udp_obj.q_H[3]
            << ", " << SATYRR_S.dpitch
            // << ", " << udp_obj.q_H[2]
            << ", " << SATYRR_S.pitch
            << ", " << F_ext
            << ", " << F_fb
            << ", " << F_ff
            << ", " << F_touch
            << ", " << F_mom_ff
            // << ", " << udp_obj.q_H[1] 
            << ", " << (wheel_torque + tau_ff)
            << ", " << SATYRR_S.x
            << ", " << SATYRR_S.dx
            << "\n";
        }
    }
    cnt++;
            
            

}

void SATYRR_Init(const mjModel* m, mjData* d)
{
    // Convert actuator, sensor, and joint names to ID. The ids will be used in the controller function above
    torso_Pitch = mj_name2id(m, mjOBJ_JOINT, "rotate_pitch");
    torso_Roll = mj_name2id(m, mjOBJ_JOINT, "rotate_roll");
    torso_Yaw = mj_name2id(m, mjOBJ_JOINT, "rotate_yaw");
    torso_X = mj_name2id(m, mjOBJ_JOINT, "move_x");
    torso_Z = mj_name2id(m, mjOBJ_JOINT, "move_z");
    j_hip_l = mj_name2id(m, mjOBJ_JOINT, "Hip_L");
    j_hip_r = mj_name2id(m, mjOBJ_JOINT, "Hip_R");
    j_knee_l = mj_name2id(m, mjOBJ_JOINT, "Knee_L");
    j_knee_r = mj_name2id(m, mjOBJ_JOINT, "Knee_R");
    j_wheel_l = mj_name2id(m, mjOBJ_JOINT, "Ankle_L");
    j_wheel_r = mj_name2id(m, mjOBJ_JOINT, "Ankle_R");

    sh_yaw_L = mj_name2id(m, mjOBJ_JOINT, "Shoulder_yaw_L");
    sh_roll_L = mj_name2id(m, mjOBJ_JOINT, "Shoulder_roll_L");
    sh_pitch_L = mj_name2id(m, mjOBJ_JOINT, "Shoulder_pitch_L");
    elbow_L = mj_name2id(m, mjOBJ_JOINT, "Elbow_L");

    sh_yaw_R = mj_name2id(m, mjOBJ_JOINT, "Shoulder_yaw_R");
    sh_roll_R = mj_name2id(m, mjOBJ_JOINT, "Shoulder_roll_R");
    sh_pitch_R = mj_name2id(m, mjOBJ_JOINT, "Shoulder_pitch_R");
    elbow_R = mj_name2id(m, mjOBJ_JOINT, "Elbow_R");

    //Init states of robot position
    // d->qpos[m->jnt_qposadr[torso_Z]] = -0.5;  //Initial Height Position of the Robot ... why is this -.5?
    d->qpos[m->jnt_qposadr[torso_Z]] = -0.5;  //Initial Height Position of the Robot ... why is this -.5?
    d->qpos[m->jnt_qposadr[torso_X]] = 0.0; 
    d->qpos[m->jnt_qposadr[torso_Pitch]] = 0; 
    d->qpos[m->jnt_qposadr[j_hip_l]] = SATYRR_S.desHip; 
    d->qpos[m->jnt_qposadr[j_hip_r]] = SATYRR_S.desHip; 
    d->qpos[m->jnt_qposadr[j_knee_l]] = -SATYRR_S.desHip*2; 
    d->qpos[m->jnt_qposadr[j_knee_r]] = -SATYRR_S.desHip*2; 

    d->qpos[m->jnt_qposadr[sh_yaw_L]] = 0; 
    d->qpos[m->jnt_qposadr[sh_roll_L]] = 0; 
    d->qpos[m->jnt_qposadr[sh_pitch_L]] = 0; 
    d->qpos[m->jnt_qposadr[elbow_L]] = 0; 
    d->qpos[m->jnt_qposadr[sh_yaw_R]] = 0; 
    d->qpos[m->jnt_qposadr[sh_roll_R]] = 0; 
    d->qpos[m->jnt_qposadr[sh_pitch_R]] = 0; 
    d->qpos[m->jnt_qposadr[elbow_R]] = 0; 

    d->qpos[m->jnt_qposadr[sh_pitch_L]] = -0.75; 
    d->qpos[m->jnt_qposadr[elbow_L]] = -3.14;
    d->qpos[m->jnt_qposadr[sh_pitch_R]] = 0.75; 
    d->qpos[m->jnt_qposadr[elbow_R]] = 3.14; 
}

// main function
int main(int argc, const char** argv)
{
    // activate software
    mj_activate("mjkey.txt");

    // load and compile model
    char error[1000] = "Could not load binary model";
    // m = mj_loadXML("../src/map1.xml", 0, error, 1000);
    // m = mj_loadXML("../src/data/satyrr_wholebody.xml", 0, error, 1000);
    m = mj_loadXML("data/satyrr_wholebody.xml", 0, error, 1000);


    if( !m ) mju_error_s("Load model error: %s", error);


    // // Create a random terrain heightmap (10 by 10 nodes)
	// Eigen::MatrixXd terrain(10, 10);
	// for (int i = 0; i < terrain.size(); i++)
	// {
	// 	terrain(i) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 1.0;
	// }

	// // Create the heightmap in MuJoCo
	// set_heightfield(m, 0, terrain, 5.0, 5.0, 1.0);

    printf("%d", m->nhfielddata);


    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // SATYRR Init
    SATYRR_Init(m, d);

    //Data logging check 
    if (data_save_flag)
        myfile.open("C:/Users/roman/Documents/Marty/MuJoCo_SATYRR/SATYYR_Old_FullBody/mujoco200_win64/data_log/" + file_name + ".txt",ios::out);
    
    // controller setup: install control callback
    mjcb_control = mycontroller;

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjtNum timezero = d->time;
    double_t update_rate = 0.001;
    last_update = timezero-1.0/ctrl_update_freq;

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

   // UDP threading 
//    UDP_Data *udp_obj;
//    udp_obj = new UDP_Data();

    // std::thread udp(&UDP_Data::udp_recieve, &udp_obj);
    // udp.detach();

    // UDP send setup (client)
    // struct sockaddr_in s_other_send; 
    // int s_send, i_send;
    // int slen=sizeof(s_other_send);
	// WSADATA wsa;
	// //Initialise winsock
	// printf("\nInitialising Winsock for send...");
	// if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
	// {
	// 	printf("Failed. Error Code : %d",WSAGetLastError());
	// 	exit(EXIT_FAILURE);
	// }
	// printf("Initialised.\n");
	// //create socket
    // if ( (s_send=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    // {
	// 	printf("socket() failed with error code : %d" , WSAGetLastError());
	// 	exit(EXIT_FAILURE);
    // }
	// //setup address structure
    // memset((char *) &s_other_send, 0, sizeof(s_other_send));
    // s_other_send.sin_family = AF_INET;
    // s_other_send.sin_port = htons(PORT_SEND);
	// s_other_send.sin_addr.S_un.S_addr = inet_addr(SERVER);

    // Timer for path starts
    completion_time_clock = clock();

    //Main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;

        while( d->time - simstart < 1.0/60.0 )
        {
            mj_step(m, d);
        }

        //Send data back to MuJoCo
        // char data_to_hmi[4];
        // memcpy(data_to_hmi, &(speed_test), 4*sizeof(char));
        // if (sendto(s_send, data_to_hmi, 4*sizeof(char), 0 , (struct sockaddr *) &s_other_send, slen) == SOCKET_ERROR)
        // {
        //     printf("sendto() failed with error code : %d" , WSAGetLastError());
        //     exit(EXIT_FAILURE);
        // }
        
        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }
    //Close data logging file

    if (data_save_flag){
        myfile.close();
        printf("close file!! \n");
    }


    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}




