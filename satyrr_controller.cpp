#include "satyrr_controller.hpp"


using namespace std;


SATYRR_STATE::SATYRR_STATE()
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

bool SATYRR_STATE::getCOM(double q_hip_l, double q_hip_r, double pitch)
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
void SATYRR_STATE::velocityMapping(float *q_H, float*q_R, float* tgt)
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

void SATYRR_STATE::dcmMapping(float *q_H, float*q_R, float* tgt)
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


/*
*******************************************************************************************************************
*******************************************************************************************************************
*******************************************************************************************************************

*/

SATYRR_controller::SATYRR_controller()
{
    for(int i=0;i<actuator_NUM;i++)
    {
        applied_torq[i] = 0.0;
    }
    FxR = 0.0;
    wheel_torque = 0.0;
    yaw_torq = 0.0;

}

void SATYRR_controller::lqrController(float *states, float *tgt)
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

void SATYRR_controller::l2cpController(float *states, float *tgt)
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
