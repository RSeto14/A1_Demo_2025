import socket
import struct
import numpy as np
import glob
import sys
import os

from xbox360_controller import Xbox360Controller
from policy import Policy

import subprocess

"""
このスクリプトはPC側のサーバーを定義しています。
このスクリプトを実行するとPC側のサーバーとUnitree A1のクライアントが接続します。
"""


script_dir = os.path.dirname(__file__)    
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)

script_name = os.path.basename(__file__)[: -len(".py")]




class Server:
    def __init__(self):
        
        ####################################################################################
        # self.host = "169.254.250.232"
        # self.host = "169.254.6.104"
        self.host = "192.168.12.240"
        self.port = 12345

        self.dt = 0.02

        self.mode = 0
        self.use_residual = False

        self.policy_id = 0
        self.LR_conversion = False
        self.command = np.array([0.0,0.0,0.0])

        self.stop_step = 0

        
        
        self.controller = Xbox360Controller()

        print("Controller OK")

        # self.stand_policy = StandPolicy()
        self.policy = Policy()

        print("Policy OK")

            
        # ソケットの作成
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server is listening on {self.host}:{self.port}")

        self.Kp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.target_q = np.array([0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6])

        
        
        """
        Joint Index
        0: FR_hip, 1: FR_thigh, 2: FR_calf
        3: FL_hip, 4: FL_thigh, 5: FL_calf
        6: RR_hip, 7: RR_thigh, 8: RR_calf
        9: RL_hip, 10: RL_thigh, 11: RL_calf
        """
        self.controlstep = 0
        self.print_controlstep = 0
        self.joint_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.foot_force = np.array([0.0, 0.0, 0.0, 0.0])
        self.euler = np.array([0.0, 0.0])
        self.angular_vel = np.array([0.0, 0.0, 0.0])
        
        self.hat_vert_push = False
        self.hat_horz_push = False
        # self.start_push = False
        # self.home_push = False
        

    def get_data(self, data):
        array = np.array(struct.unpack('d' * 34, data[:34*8]))

        self.controlstep = array[0]
        self.joint_q = array[1:13]
        self.joint_dq = array[13:25] 
        # self.foot_force = array[25:31] # FR, FL, RR, RL
        self.foot_force = (array[25:29] > 0.1).astype(int)
        self.euler = np.degrees(array[29:31]) # roll, pitch
        self.angular_vel = array[31:34] # roll, pitch, yaw
    
    def print_info(self,hz):
        if (self.controlstep - self.print_controlstep)/500 >= 1/hz:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.print_controlstep = self.controlstep

            print(f"controlstep: {self.controlstep}")
            print(f"mode: {self.mode}")
            print(f"Kp: {self.Kp}")
            print(f"Kd: {self.Kd}")
            print(f"target_q: {self.target_q}")
            print(f"joint_q: {self.joint_q}")
            print(f"joint_dq: {self.joint_dq}")
            print(f"foot_force: {self.foot_force}")
            print(f"euler: {self.euler}")
            print(f"angular_vel: {self.angular_vel}")


    def set_param(self):

        
        if np.abs(self.controller.hat_vert) > 0.5 and self.hat_vert_push == False:
            # print(f"self.controller.hat_vert: {self.controller.hat_vert}")
            self.Kp = self.controller.hat_vert * 5.0 + self.Kp
            self.Kp = np.round(self.Kp,3)
            self.Kp = np.clip(self.Kp, 0.0, 60.0)
            self.hat_vert_push = True

        elif np.abs(self.controller.hat_vert) > 0.5:

            self.hat_vert_push = True

        else:
            self.hat_vert_push = False

        # if self.controller.start > 0.1 and self.start_push == False:
        #     self.Kp += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        #     self.start_push = True

        # else:
        #     self.start_push = False

        # if self.controller.home > 0.1 and self.home_push == False:
        #     self.Kp -= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        #     self.home_push = True

        # else:
        #     self.home_push = False



            

        if np.abs(self.controller.hat_horz) > 0.5 and self.hat_horz_push == False:
            self.Kd = self.controller.hat_horz * 0.5 + self.Kd
            self.Kd = np.round(self.Kd,3)
            self.Kd = np.clip(self.Kd, 0.0, 2.0)
            self.hat_horz_push = True
        elif np.abs(self.controller.hat_horz) > 0.5:


            self.hat_horz_push = True
        else:
            self.hat_horz_push = False

    def mode_change(self):
        if self.controller.RB > 0.1:
            self.mode = 1
            self.use_residual = True

        if self.controller.LB > 0.1:
            self.mode = 1
            self.use_residual = False

        if self.controller.LT > 0.1:
            self.mode = 0
            self.policy.reset()
        if self.controller.RT > 0.1:
            self.mode = 0
            self.policy.reset()
            self.Kp = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
    
    def command_change(self):

        if self.controller.Ljoy_vert >= 0.5 and np.abs(self.controller.Rjoy_horz) < 0.5:
            self.command = np.array([0.3, 0.0, 0.0])
            self.policy_id = 1
            self.LR_conversion = False

        elif self.controller.Ljoy_vert >= 0.5 and self.controller.Rjoy_horz >= 0.5:
            self.command = np.array([0.3, 0.0, 0.5])
            self.policy_id = 2
            self.LR_conversion = False

        elif self.controller.Ljoy_vert >= 0.5 and self.controller.Rjoy_horz <= -0.5:
            self.command = np.array([0.3, 0.0, -0.5])
            self.policy_id = 2
            self.LR_conversion = True

        elif self.controller.Ljoy_vert < 0.5 and self.controller.Rjoy_horz >= 0.5:
            self.command = np.array([0.0, 0.0, 1.0])
            self.policy_id = 3
            self.LR_conversion = False

        elif self.controller.Ljoy_vert < 0.5 and self.controller.Rjoy_horz <= -0.5:
            self.command = np.array([0.0, 0.0, -1.0])
            self.policy_id = 3
            self.LR_conversion = True
        
        elif self.controller.Ljoy_vert < 0.5 and self.controller.Ljoy_horz >= 0.5:
            self.command = np.array([0.0, 0.3, 0.0])
            self.policy_id = 4
            self.LR_conversion = False

        elif self.controller.Ljoy_vert < 0.5 and self.controller.Ljoy_horz <= -0.5:
            self.command = np.array([0.0, -0.3, 0.0])
            self.policy_id = 4
            self.LR_conversion = True

        else:
            self.command = np.array([0.0, 0.0, 0.0])
            self.policy_id = 0
            self.LR_conversion = False



    def target_q_change(self):
        if self.mode == 1:
            if self.policy_id != 0:
                target_q, res = self.policy.get_target_q(self.joint_q, self.joint_dq, self.euler[0:2], self.angular_vel, self.foot_force, commands=self.command,policy_id=self.policy_id,LR_conversion=self.LR_conversion)
                if self.use_residual:
                    self.target_q = target_q + res
                else:
                    self.target_q = target_q
                self.stop_step = 0
            else:
                if self.stop_step < 30:
                    target_q, res = self.policy.get_target_q(self.joint_q, self.joint_dq, self.euler[0:2], self.angular_vel, self.foot_force, commands=np.array([0.0, 0.0, 0.0]),policy_id=0)
                    if self.use_residual:
                        self.target_q = target_q + res
                    else:
                        self.target_q = target_q
                    
                else:
                    target_q, res = self.policy.get_target_q(self.joint_q, self.joint_dq, self.euler[0:2], self.angular_vel, self.foot_force, commands=np.array([0.0, 0.0, 0.0]),policy_id=0)
                    if self.use_residual:
                        self.target_q = np.array([0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597]) + res
                    else:
                        self.target_q = np.array([0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597])
                    self.policy.reset(residual_reset= not self.use_residual)

                self.stop_step += 1
            

        else:
            # self.target_q = np.array([0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.75976, -1.5195, 0.0, 0.75976, -1.5195])
            # self.target_q = np.array([0.0, 0.82983, -1.8, 0.0, 0.82983, -1.6597, 0.0, 0.75976, -1.5195, 0.0, 0.75976, -1.5195])
            self.target_q = np.array([0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597, 0.0, 0.82983, -1.6597])
            self.stop_step = 0

            

if __name__ == "__main__":


    env_name = "demo2025"  # conda環境名
    script_path = rf"{script_dir}\PD.py"
    subprocess.Popen(f'start cmd /C "conda activate {env_name} && python {script_path}"', shell=True)

    try:
        server = Server()

    
        while True:
            client_socket, client_address = server.server_socket.accept()
            # print(f"Connection from {client_address} has been established.")
            
            data = client_socket.recv(1024)
            if len(data) < 34*8:  # 36 doubles * 8 bytes each
                    print(f"Received data ({len(data)}) is less than expected(680)")
                    client_socket.close()
                    continue
            
            server.get_data(data)
            server.controller.check_buttons(threshold=0.2)
            server.set_param()
            server.print_info(hz=1)

            server.mode_change()
            server.command_change()
            server.target_q_change()


            response = struct.pack('d' * 36, *(server.Kp.tolist() + server.Kd.tolist() +server.target_q.tolist()))
            client_socket.send(response)
            client_socket.close()

    except KeyboardInterrupt:
        print("Script interrupted by user.")
        sys.exit(0)


