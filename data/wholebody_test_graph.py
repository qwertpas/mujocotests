import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
from matplotlib.pyplot import plot, figure, legend
import matplotlib.pyplot as plt
import mujoco_py as mj
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from math import cos, sin, pi
import os
import copy

from utilss import rotation
# from utils.rotation_old import *
from models import *
import scipy.linalg
from time import sleep
#from utilss.controller import *
from pynput import keyboard
import time, os, fnmatch, shutil
import threading
import utilss.config
import threading

HMI_mode = False #obtain control from HMI device
target = np.array([0.0,0,0,0])
x_set = target[0]
yaw_set = 0.0   
x_step = 0.5
yaw_step = 0.5
bthread = False

global old_theta, time_t, des_x, x_dot_old
control_rate = 1. / 0.001
time_t = 0.0
des_x, des_dx, des_y = 0.0, 0.0, 0.0
x_dot_old = 0.0
theta = 0.0
old_theta = 0.0
theta_dot_old = 0.0
theta_dot_ = 0.0
desHip = 0.45588
sampling_time = 0.0
dCutoffreq = 0.7

def current_milli_time():
    return round(time.time() * 1000)

class satyrr_wholebody:
    gravity = 9.81
    mass_wheel = 0.525989*2
    mass_leg = (0.904864 + 0.394882) * 2 # thigh + shin
    mass_torso = 4.05
    mass_upper = (0.2256 + 0.58497 + 0.111225 + 0.24689) * 2# sholder + arm + shank + shoulder_u
    r = 0.12/2
    leg = 0.55
    dt = 0.002
    length = 0.8

    pitch_actual_old = 0.0
    pitch_actual_dot = 0.0

def getCOM(q_hip, pitch):
    """ CoM calculation per leg
    @param q_hip = [left, right], theta of hip
    @param pitch of the robot
    """
    
    CoM_R0 = -.01884*cos(pitch) + .07329*sin(pitch) - satyrr_wholebody.leg*sin(q_hip[1] - pitch) + satyrr_wholebody.leg*sin(q_hip[1] + pitch)
    CoM_R1 = .07329*cos(pitch) + .01884*sin(pitch) + satyrr_wholebody.leg*cos(q_hip[1]+ pitch) + satyrr_wholebody.leg*cos(q_hip[1] - pitch) 

    CoM_L0 = -.01884*cos(pitch) + .07329*sin(pitch) - satyrr_wholebody.leg*sin(q_hip[0] - pitch) + satyrr_wholebody.leg*sin(q_hip[0] + pitch)
    CoM_L1 = .07329*cos(pitch) + .01884*sin(pitch) + satyrr_wholebody.leg*cos(q_hip[0]+ pitch) + satyrr_wholebody.leg*cos(q_hip[0] - pitch)

    CoM0 = (CoM_R0 + CoM_L0)/2
    CoM1 = (CoM_R1 + CoM_L1)/2 
    return np.array([CoM0, CoM1],dtype=np.float32)

def stabilizationControl(tgt, state, pitch_actual):
    """ stabilization contrl
    @param tgt = [x,  theta, x_dot, theta_dot]
    @param state = [x,  theta, x_dot, theta_dot]
    @param CoM center of mass
    """
    K_xW, K_pitch, K_dxW, K_dpitch  = -180, -640, -120, -70 , #-100, -315, -40, -40
    # pitch_actual = np.arctan2(CoM[0], CoM[1])
    
    #satyrr_wholebody.pitch_actual_dot = (pitch_actual - satyrr_wholebody.pitch_actual_old) / 0.001
    #satyrr_wholebody.pitch_actual_old =  pitch_actual
    # print(f"pitch_actual: {pitch_actual}, q_pitch: {state[1]}")# similar actually
    FxR = K_xW *(tgt[0] - state[0]) + K_dxW*(tgt[2] - state[2]) + K_pitch*(0 - pitch_actual) + K_dpitch*(0 - state[3])
    #FxR = K_xW *(tgt[0] - state[0]) + K_dxW*(tgt[2] - state[2]) + K_pitch*(0 - pitch_actual) + K_dpitch*(0 - satyrr_wholebody.pitch_actual_dot)
    # wheel_torque = FxR * satyrr_wholebody.r
    wheel_torque = FxR * satyrr_wholebody.r /2
    return wheel_torque


def yawControl(tgt, state):
    """ yaw contrl
    @param target [yaw,yaw_vel], yaw_vel is better to be filtered
    @param state [yaw, yaw_vel]
    """
    Kp_yaw = 1.9
    Kd_yaw = 0.4
    yaw_damp = Kp_yaw *(tgt[0] - state[0]) + Kd_yaw * (tgt[1] - state[1])
    return yaw_damp

def jointContrl(q, q_vel, tgt, K):
    """control on knee joint
    @ param theta on left and right joint with respect to parent [q_joint_l, q_joint_r]
    @ param theta vel on left and right joint with respect to torso [q_joint_vel_l, q_joint_vel_r]
    @ param tgt theta
    # param K = [Kp_r, Kd_r, Kp_l, Kd_l]
    """
    # print("thetaDes: ", thetaDes)
    joint_torque_l = K[2]*(q[0] - tgt) + K[3]*(q_vel[0] - 0)
    joint_torque_r = K[0]*(q[1] - tgt) + K[1]*(q_vel[1] - 0)
    return joint_torque_l, joint_torque_r

def on_release(key):
    global x_set, yaw_set
    if key == keyboard.Key.down:
        x_set -= x_step
        print("add -{0}m to reference x ".format(x_step))

    if key == keyboard.Key.up:
        x_set += x_step
        print("add {0}m to reference x ".format(x_step))

    if key == keyboard.Key.left:
        yaw_set += yaw_step
        print("add {0} radian to reference psi ".format(yaw_step))

    if str(key) == "'z'":
        yaw_set -= yaw_step
        print("add -{0} radian to reference psi ".format(yaw_step))

    if key == keyboard.Key.esc:
        # Stop listener
        return False


def traj(tt):
    magnitude = 1
    T = magnitude * 5
    Steady_T = 3  
    S_T = T + Steady_T
    
    pos_prev, vel_prev, acc_prev = [0, 0, 0]
    pos_curr, vel_curr, acc_curr = [1, 0, 0]

    a0 = pos_prev;
    a1 = vel_prev;
    a2 = acc_prev / 2;

    a3 = (20 * pos_curr - 20 * pos_prev - (8 * vel_curr + 12 * vel_prev) * T - (
          3 * acc_prev - acc_curr) * T ** 2) / (2 * T ** 3);
    a4 = (-30 * pos_curr + 30 * pos_prev + (14 * vel_curr + 16 * vel_prev) * T + (
          3 * acc_prev - 2 * acc_curr) * T ** 2) / (2 * T ** 4);
    a5 = (12 * pos_curr - 12 * pos_prev - (6 * vel_curr + 6 * vel_prev) * T - (
         acc_prev - acc_curr) * T ** 2) / (2 * T ** 5);

    if (tt < T):
       traj_des_x = magnitude * (a0 + a1 * tt + a2 * (tt ** 2) + a3 * (tt ** 3) + a4 * (tt ** 4) + a5 * (tt ** 5))
       traj_des_dx = 0.0  # magnitude * (
                # a1 + 2 * a2 * tt + 3 * a3 * (tt ** 2) + 4 * a4 * (tt ** 3) + 5 * a5 * (
                #   tt ** 4))
    elif (tt >= T and tt < S_T):
       traj_des_x = magnitude * (a0 + a1 * T + a2 * (T ** 2) + a3 * (T ** 3) + a4 * (T ** 4) + a5 * (T ** 5))
       traj_des_dx = 0.0  # magnitude * (
                # a1 + 2 * a2 * T + 3 * a3 * (T ** 2) + 4 * a4 * (T ** 3) + 5 * a5 * (
                #  T ** 4))
    elif (tt >= S_T and tt < S_T + T):
       traj_des_x = -magnitude * (a0 + a1 * (tt - S_T) + a2 * ((tt - S_T) ** 2) + a3 * ((tt - S_T) ** 3) + a4 * ((tt - S_T) ** 4) + a5 * ((tt - S_T) ** 5)) + magnitude * (
                                     a0 + a1 * T + a2 * (T ** 2) + a3 * (T ** 3) + a4 * (
                                         T ** 4) + a5 * (T ** 5))
       traj_des_dx = 0.0  # -magnitude * (a1 + 2 * a2 * (tt - S_T) + 3 * a3 * ((tt - S_T) ** 2) + 4 * a4 * (
                # (tt - S_T) ** 5) + 5 * a5 * ((tt - S_T) ** 5))
    else:
       traj_des_x = 0
       traj_des_dx = 0.0

    traj_des_y = 0.0

    return traj_des_x, traj_des_y, traj_des_dx

def run_callback(_mutex ):
    while True:
        now_c = time.time()

        if bthread:
            _mutex.acquire()
            run_func()
            _mutex.release()

        elapsed_c = time.time() - now_c
        sleep_time_c = (1. / control_rate) - elapsed_c
        if sleep_time_c > 0.0:
            time.sleep(sleep_time_c)

        tf = time.time()
        print("time = %f" % (tf - now_c))
        if time_t > 13:
            break
#
# def run_func():
#     global x_dot_old, old_theta, time_t ,des_x
#
#     #########################Get state#####################
#     sim_state = sim.get_state()
#     q_hip = [sim_state.qpos[10], sim_state.qpos[7]]
#     q_vel_hip = [sim_state.qvel[9], sim_state.qvel[6]]
#     q_knee = [sim_state.qpos[11], sim_state.qpos[8]]
#     q_vel_knee = [sim_state.qvel[10], sim_state.qvel[7]]
#
#     quat = sim_state.qpos[3:7]
#     angular_vel = sim_state.qvel[3:6]  # angular velocity: av_x, av_y, av_z
#
#     quatObject = rotation.Quaternion(*quat)
#     euler = quatObject.toEuler("zyx")
#     angles = euler.asArray()  # angles: yaw, pitch, roll
#
#     R = quatObject.toRotationMatrix()
#     angular_vel = R @ angular_vel
#     euler_dot = rotation.Misc.EulerAngleDerivative(angles, angular_vel, 2, 1, 0)  # angular velocity
#
#     x, theta = sim_state.qpos[0], angles[1]
#     # x_dot, theta_dot = sim_state.qvel[0], euler_dot[1]
#
#     x_dot = sim_state.qvel[0]
#     dCutoffreq = 0.7
#     x_dot = 2 * pi * dCutoffreq * dt * x_dot + (1 - 2 * pi * dCutoffreq * dt) * x_dot_old
#     x_dot_old = x_dot
#
#     theta_dot = (theta - old_theta) / dt
#     old_theta = theta
#     # theta_dot = 2 * pi * dCutoffreq * dt * theta_dot + (1 - 2 * pi * dCutoffreq * dt) * theta_dot_old
#     # theta_dot_old = theta_dot
#
#     state = [x, theta, x_dot, theta_dot]
#
#     ############Desired ############################
#     # target[0]= x_set
#     des_dx, des_y, _ = traj(time_t)
#     des_x += des_dx * dt
#
#     target[0] = des_x
#     target[2] = des_dx
#     yaw_set = des_y
#
#     if HMI_mode:
#         target[0] = utils.config.dSet
#         print(f"target x: {target[0]}")
#
#     CoM = getCOM(q_hip, angles[1])
#     pitch_actual = np.arctan2(CoM[0], CoM[1])
#
#     wheel_torque = stabilizationControl(target, state, pitch_actual)
#     yaw_damp = yawControl([yaw_set, 0], [angles[0], euler_dot[0]])
#     # yaw_damp = 0
#     wheel_torque_l = wheel_torque - yaw_damp
#     wheel_torque_r = wheel_torque + yaw_damp
#
#     # hip_torque_l, hip_torque_r = jointContrl(q_hip, q_vel_hip)
#
#     hip_torque_l, hip_torque_r = jointContrl(q_hip, q_vel_hip, desHip, [500, 5, 500, 5])
#     knee_torque_l, knee_torque_r = jointContrl(q_knee, q_vel_knee, -desHip * 2, [200, 2, 200, 2])
#
#     # print('current target x:{0}'.format( x_set))
#     # hip_l, knee_l, wheel_l, hip_r, knee_r, wheel_r
#     ctrl = -np.array([hip_torque_l, knee_torque_l, wheel_torque_l, hip_torque_r, knee_torque_r, wheel_torque_r])
#     # print("torque: ", ctrl)
#     sim.data.ctrl[:] = ctrl
#     sim.step()
#     viewer.render()
#
#     time_t += dt
#     f_result.write("%f, %f, %f, %f, %f, %f, %f, %f, %f \n" % (
#     time_t, target[0], state[0], state[1], pitch_actual, state[2], state[3], euler_dot[1], wheel_torque))




if __name__ == "__main__":
    model_xml = open("../data/satyrr_wholebody.xml").read()
    model = mj.load_model_from_xml(model_xml)
    sim = mj.MjSim(model)
    #viewer = mj.MjViewer(sim)
    dt = model.opt.timestep

    f_result = open('/home/baek/Desktop/saytrr_wholebody.txt', 'w')


    if bthread:
        t_mutex = threading.Lock()
        t = threading.Thread(target=run_callback(t_mutex))
        t.start()

    # controller = CartPoleController(target)
    # print(controller.get_lqr_gain()) #[[  8.76095426 188.84008759  14.05866576  28.67560703]]


    quat = (rotation.Euler(0, theta, 0).toQuaternion().asArray())
    sim_state = sim.get_state()

    # Initial pos for standing upright
    sim_state.qpos[0:3] = [0.0, 0.0,-0.04]
    sim_state.qpos[3:7] = quat
    sim_state.qvel[0:3] = [0.0, 0.0, 0.0]
    sim_state.qvel[3:6] = [0.0, 0.0, 0.0]
    # hip and knee angle
    sim_state.qpos[10] = desHip
    sim_state.qpos[11] = -desHip*2
    sim_state.qpos[7] = desHip
    sim_state.qpos[8] = -desHip*2

    # initial pose for testing sit to stand
    # sim_state.qpos[:] = [ 4.20782829e-02, -1.28548787e-05, -2.12818501e-01,  9.99633435e-01,
    #                     -1.96259443e-05,  2.70734912e-02,  1.45473736e-04,  6.56669914e-01,
    #                     -2.39245103,  1.51862399,  6.56579115e-01, -2.39256709, 1.52015096]
    sim.set_state(sim_state)
    sim.step()

    if not bthread:
        while True:
            ti = time.time()
            #########################Get state#####################
            sim_state = sim.get_state()
            # print("state: ", sim_state.qpos)
            q_hip = [sim_state.qpos[10], sim_state.qpos[7]]
            q_vel_hip = [sim_state.qvel[9], sim_state.qvel[6]]
            q_knee = [sim_state.qpos[11], sim_state.qpos[8]]
            q_vel_knee = [sim_state.qvel[10], sim_state.qvel[7]]
            q_wheel = [sim_state.qpos[12], sim_state.qpos[9]]
            q_vel_wheel = [sim_state.qpos[11], sim_state.qpos[8]]
            # print("q_hip: ", q_hip, q_vel_hip)
            # print("left hip: ", sim_state.qpos[10] * 57)
            # print("left knee: ", sim_state.qpos[11] * 57)

            quat = sim_state.qpos[3:7]
            angular_vel = sim_state.qvel[3:6]  # angular velocity: av_x, av_y, av_z


            quatObject = rotation.Quaternion(*quat)
            euler = quatObject.toEuler("zyx")
            angles = euler.asArray()  # angles: yaw, pitch, roll
            # print("euler angles: ", angles)

            R = quatObject.toRotationMatrix()
            angular_vel = R @ angular_vel
            euler_dot = rotation.Misc.EulerAngleDerivative(angles, angular_vel, 2, 1, 0) # angular velocity

            x, theta = sim_state.qpos[0], angles[1]
            #x_dot, theta_dot = sim_state.qvel[0], euler_dot[1]

            #theta = 2 * pi * 70 * dt * theta + (1 - 2 * pi * 70 * dt) * theta_dot_old
            #theta_dot_old = theta

            x_dot = sim_state.qvel[0]

            x_dot = 2 * pi * dCutoffreq * dt * x_dot + (1 - 2 * pi * dCutoffreq * dt) * x_dot_old
            x_dot_old = x_dot

            theta_dot = (theta - old_theta) / dt
            old_theta = theta


            state = [x,  theta, x_dot, theta_dot]

            ############Desired ############################
            #target[0]= x_set
            des_dx, des_y, _ = traj(time_t)
            des_x += des_dx*dt


            target[0] = des_x
            target[2] = des_dx
            yaw_set = des_y

            if HMI_mode:
                target[0] = utils.config.dSet
                print(f"target x: {target[0]}")

            CoM = getCOM(q_hip, angles[1])
            pitch_actual = np.arctan2(CoM[0], CoM[1])

            wheel_torque = stabilizationControl(target, state, pitch_actual)
            yaw_damp = yawControl([yaw_set,0], [angles[0], euler_dot[0]])
            # yaw_damp = 0
            wheel_torque_l = wheel_torque - yaw_damp
            wheel_torque_r = wheel_torque + yaw_damp

            # hip_torque_l, hip_torque_r = jointContrl(q_hip, q_vel_hip)

            hip_torque_l, hip_torque_r = jointContrl(q_hip, q_vel_hip, desHip, [500,5,500,5])
            knee_torque_l, knee_torque_r = jointContrl(q_knee, q_vel_knee, -desHip * 2, [200,2,200,2])

            # print('current target x:{0}'.format( x_set))
            # hip_l, knee_l, wheel_l, hip_r, knee_r, wheel_r
            ctrl = -np.array([hip_torque_l, knee_torque_l, wheel_torque_l, hip_torque_r, knee_torque_r, wheel_torque_r])
            #ctrl = -np.array([hip_torque_l, knee_torque_l, 0.0, hip_torque_r, knee_torque_r, 0.0])
            # print("torque: ", ctrl)
            sim.data.ctrl[:] = ctrl
            sim.step()
            #viewer.render()

            time_t += dt
            f_result.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n" %(time_t, target[0],state[0],state[1], pitch_actual, state[2], state[3], euler_dot[1], wheel_torque, q_wheel[0], q_wheel[1]))



            elapsed_c = time.time() - ti
            sleep_time_c = (1. / control_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)

            sampling_time = time.time() - ti
            print("time = %f" % (sampling_time))

            if time_t > 10:
               break



