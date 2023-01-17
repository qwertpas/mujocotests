#ifndef BDH_SATYRR_CONTROLLER_HPP_
#define BDH_SATYRR_CONTROLLER_HPP_

#include "mujoco/mujoco.h"

#include <math.h>
#include "string.h"
#include <vector>

#define q_free_NUM 6
#define q_NUM 12
#define actuator_NUM 6

using namespace std;
#define Hip 1
#define Knee 2
#define Yaw_left 1
#define Yaw_right 2

#define SATYRR_leg  0.55
#define SATYRR_r  0.06
#define SATYRR_length  0.8

class SATYRR_STATE
{
    public:
        SATYRR_STATE();
        
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
        
        bool getCOM(float q_hip_l, float q_hip_r, float pitch);
        void velocityMapping(float *q_H, float *q_R, float* tgt);
        void dcmMapping(float *q_H, float *q_R, float* tgt);

};



class SATYRR_controller
{
    public:
        float FxR;
        float applied_torq[actuator_NUM];
        float wheel_torque;
        float yaw_torq;
        
        float q_DCM[4];

        SATYRR_controller();
        void lqrController(float *states, float *tgt);
        void l2cpController(float *states, float *tgt);
        
        float joint_torq_out[2];

        //Robot params 
        float h_R = 0.4253; // full body satyrr com height 
        float m_R = 0.4297; //mass of wheel
        float M_R = 7.16; // mass of body 
        float omega_R = sqrt((9.81/h_R));
        // float K_l2cp[4] ={-27.7356, -116.4209, 0, -264.7271};
        float K_l2cp[4] ={-27.7356, -116.4209, 0, -470};
}; 


#endif