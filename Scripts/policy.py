import os
import sys
import torch
import json
import math
import numpy as np
import re
import random
# import genesis as gs


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)

from rsl_rl import runners


"""
このスクリプトはPolicyクラスを定義しています。
このスクリプトを実行するとシミュレーションテストができます。
"""

##### シミュレーションテスト設定 #####
# test_id_list = [0,1,2,3,4] # テストするポリシーのID 0: Stop, 1: Forward, 2: Forward Turn Left, 3: Turn Left, 4: Left
test_id_list = [2]
test_LR_conversion = False # テストするポリシーのLR変換の有無
test_residual = True # 残差ポリシーの有無
terrain_type = "box" # "flat" or "box"
n_rows = 20 # boxの行数
n_cols = 20 # boxの列数
box_noise = 0.04 # boxの最大高さの半分の値を設定 [m]
test_commands_list = [np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]), np.array([0.3, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.3, 0.0])]
########################################
    
class Policy:
    def __init__(self):
        self.dt = 0.02
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dirs = [
            
            # os.path.join(parent_dir, "Logs", "CPG-RL", "250216_161050"), # Stop [0.0, 0.0, 0.0]
            os.path.join(parent_dir, "Logs", "CPG-RL", "250219_032206"), # Stop [0.0, 0.0, 0.0]
            # os.path.join(parent_dir, "Logs", "CPG-RL", "250215_215635"), # Forward [0.3, 0.0, 0.0]
            os.path.join(parent_dir, "Logs", "CPG-RL", "250219_021030"), # Forward [0.3, 0.0, 0.0]
            # os.path.join(parent_dir, "Logs", "CPG-RL", "250216_115210"), # Forward Turn Left [0.3, 0.0, 0.5]
            os.path.join(parent_dir, "Logs", "CPG-RL", "250219_025005"), # Forward Turn Left [0.3, 0.0, 0.5]
            # os.path.join(parent_dir, "Logs", "CPG-RL", "250216_111642"), # Turn Left [0.0, 0.0, 1.0]
            os.path.join(parent_dir, "Logs", "CPG-RL", "250219_032033"), # Turn Left [0.0, 0.0, 1.0]
            # os.path.join(parent_dir, "Logs", "CPG-RL", "250216_131807"), # Left [0.0, 0.3, 0.0]
            os.path.join(parent_dir, "Logs", "CPG-RL", "250219_034230"), # Left [0.0, 0.3, 0.0]
            
        ]

        self.residual_log_dirs = [
            os.path.join(parent_dir, "Logs", "Res4CPG-RL", "250219_051757"), # Residual for All
            # os.path.join(parent_dir, "Logs", "Res4CPG-RL", "250219_042228"), # Residual Stop [0.0, 0.0, 0.0]
            # os.path.join(parent_dir, "Logs", "Res4CPG-RL", "250219_034409"), # Residual Forward [0.3, 0.0, 0.0]
            # os.path.join(parent_dir, "Logs", "Res4CPG-RL", "250219_043633"), # Residual Forward [0.3, 0.0, 0.0]
            # os.path.join(parent_dir, "Logs", "Res4CPG-RL", "250218_192605"), # Residual 
        ]
        


        self.model_paths = []
        for self.log_dir in self.log_dirs:
            network_dir = os.path.join(self.log_dir, "Networks")
            if not os.path.exists(network_dir):
                print(f"Network directory not found: {network_dir}")
                continue
            model_files = [f for f in os.listdir(network_dir) if re.match(r"model_\d+\.pt", f)]
            if not model_files:
                print(f"No model files found in: {network_dir}")
                continue
            # 数字部分を抽出して最大のものを探す
            max_model = max(model_files, key=lambda f: int(re.search(r"\d+", f).group()))
            self.model_paths.append(os.path.join(network_dir, max_model))

        self.residual_model_paths = []
        for self.residual_log_dir in self.residual_log_dirs:
            network_dir = os.path.join(self.residual_log_dir, "Networks")
            if not os.path.exists(network_dir):
                print(f"Network directory not found: {network_dir}")
                continue
            model_files = [f for f in os.listdir(network_dir) if re.match(r"model_\d+\.pt", f)]
            if not model_files:
                print(f"No model files found in: {network_dir}")
                continue
            # 数字部分を抽出して最大のものを探す
            max_model = max(model_files, key=lambda f: int(re.search(r"\d+", f).group()))
            self.residual_model_paths.append(os.path.join(network_dir, max_model))
            
    
        self.train_cfgs = []
        self.env_cfgs = []
        self.obs_cfgs = []
        for log_dir in self.log_dirs:
            with open(f"{log_dir}/config.json", 'r') as json_file:
                loaded_config = json.load(json_file)

            self.train_cfgs.append(loaded_config["train_cfg"])
            self.env_cfgs.append(loaded_config["env_cfg"])
            self.obs_cfgs.append(loaded_config["obs_cfg"])

        self.residual_train_cfgs = []
        self.residual_env_cfgs = []
        self.residual_obs_cfgs = []
        for residual_log_dir in self.residual_log_dirs:
            with open(f"{residual_log_dir}/config.json", 'r') as json_file:
                loaded_config = json.load(json_file)

            self.residual_train_cfgs.append(loaded_config["train_cfg"])
            self.residual_env_cfgs.append(loaded_config["env_cfg"])
            self.residual_obs_cfgs.append(loaded_config["obs_cfg"])

        self.envs = [A1Env(
        num_envs=1,
        env_cfg=self.env_cfgs[i],
        obs_cfg=self.obs_cfgs[i],
        device=self.device
        ) for i in range(len(self.env_cfgs))]

        self.res_envs = [A1Env(
        num_envs=1,
        env_cfg=self.residual_env_cfgs[i],
        obs_cfg=self.residual_obs_cfgs[i],
        device=self.device
        ) for i in range(len(self.residual_env_cfgs))]

        self.obs_scales_list = [self.obs_cfgs[i]["obs_scales"] for i in range(len(self.env_cfgs))]
        self.commands_scale_list = [torch.tensor(
            [self.obs_scales_list[i]["lin_vel"], self.obs_scales_list[i]["lin_vel"], self.obs_scales_list[i]["ang_vel"]],
            device=self.device,
            dtype=torch.float32,
        ) for i in range(len(self.env_cfgs))]

        self.residual_obs_scales_list = [self.residual_obs_cfgs[i]["obs_scales"] for i in range(len(self.residual_env_cfgs))]
        self.residual_commands_scale_list = [torch.tensor(
            [self.residual_obs_scales_list[i]["lin_vel"], self.residual_obs_scales_list[i]["lin_vel"], self.residual_obs_scales_list[i]["ang_vel"]],
            device=self.device,
            dtype=torch.float32,
        ) for i in range(len(self.residual_env_cfgs))]


        self.default_dof_pos_list = [
            torch.tensor([self.env_cfgs[i]["default_joint_angles"][name] for name in self.env_cfgs[i]["dof_names"]],device=self.device,dtype=torch.float32) 
            for i in range(len(self.env_cfgs))]
        
        self.residual_default_dof_pos_list = [
            torch.tensor([self.residual_env_cfgs[i]["default_joint_angles"][name] for name in self.residual_env_cfgs[i]["dof_names"]],device=self.device,dtype=torch.float32) 
            for i in range(len(self.residual_env_cfgs))]
        
        self.a_list = [self.env_cfgs[i]["a"] for i in range(len(self.env_cfgs))]
        self.d_list = [self.env_cfgs[i]["d"] for i in range(len(self.env_cfgs))]
        self.h_list = [torch.tensor(self.env_cfgs[i]["h"][0], device=self.device, dtype=torch.float32).repeat(4) for i in range(len(self.env_cfgs))]
        self.gc_list = [torch.tensor(self.env_cfgs[i]["gc"][0], device=self.device, dtype=torch.float32).repeat(4) for i in range(len(self.env_cfgs))]
        self.gp_list = [torch.tensor(self.env_cfgs[i]["gp"][0], device=self.device, dtype=torch.float32).repeat(4) for i in range(len(self.env_cfgs))]

        self.mu = torch.ones(4, device=self.device, dtype=torch.float32)
        self.omega = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.psi = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.r = torch.ones(4, device=self.device, dtype=torch.float32)
        self.r_dot = torch.zeros( 4, device=self.device, dtype=torch.float32)
        self.r_ddot = torch.zeros( 4, device=self.device, dtype=torch.float32)
        self.theta = torch.tensor([0, torch.pi, torch.pi, 0], device=self.device, dtype=torch.float32)
        # self.theta = torch.tensor([torch.pi, 0,  0, torch.pi], device=self.device, dtype=torch.float32)
        self.phi = torch.zeros(4, device=self.device, dtype=torch.float32)

        self.res = torch.zeros(12, device=self.device, dtype=torch.float32)
        self.last_res = torch.zeros(12, device=self.device, dtype=torch.float32)
        self.res_history = torch.zeros(3,12, device=self.device, dtype=torch.float32)
        self.d_res = torch.zeros(12, device=self.device, dtype=torch.float32)



        self.residual_res_scale_list = [torch.tensor(self.residual_env_cfgs[i]["res_scale"], device=self.device, dtype=torch.float32) for i in range(len(self.residual_env_cfgs))]
        self.residual_d_res_scale_list = [torch.tensor(self.residual_env_cfgs[i]["d_res_scale"], device=self.device, dtype=torch.float32) for i in range(len(self.residual_env_cfgs))]

        self.policy = []
        for i in range(len(self.log_dirs)):
            runner = runners.OnPolicyRunner(self.envs[i], self.train_cfgs[i], self.log_dirs[i], device=self.device)
            runner.load(self.model_paths[i])
            self.policy.append(runner.get_inference_policy(device=self.device))

        self.residual_policy = []
        for i in range(len(self.residual_log_dirs)):
            runner = runners.OnPolicyRunner(self.res_envs[i], self.residual_train_cfgs[i], self.residual_log_dirs[i], device=self.device)
            runner.load(self.residual_model_paths[i])
            self.residual_policy.append(runner.get_inference_policy(device=self.device))

    
    def reset(self,residual_reset=True):
        self.r = torch.ones(4, device=self.device, dtype=torch.float32)
        self.r_dot = torch.zeros( 4, device=self.device, dtype=torch.float32)
        self.r_ddot = torch.zeros( 4, device=self.device, dtype=torch.float32)
        self.mu = torch.ones(4, device=self.device, dtype=torch.float32)
        self.omega = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.psi = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.theta = torch.tensor([0, torch.pi, torch.pi, 0], device=self.device, dtype=torch.float32)
        self.phi = torch.zeros(4, device=self.device, dtype=torch.float32)

        if residual_reset:
            self.res = torch.zeros(12, device=self.device, dtype=torch.float32)
            self.last_res = torch.zeros(12, device=self.device, dtype=torch.float32)
            self.d_res = torch.zeros(12, device=self.device, dtype=torch.float32)


    # def dof_limit(self,target_dof_pos,dof_pos):
    #     return torch.clip(target_dof_pos, -self.env_cfgs[0]["dq_limit"] * self.env.dt + dof_pos, self.env_cfgs[0]["dq_limit"] * self.env.dt + dof_pos)

    def get_target_q(self, dof_pos, dof_vel, base_euler, base_ang_vel, foot_contact, commands=np.array([0.3, 0.0, 0.0]), policy_id=0, LR_conversion=False):
        obs = self.obs(dof_pos, dof_vel, base_euler, base_ang_vel, foot_contact, commands, policy_id, LR_conversion=LR_conversion)
        actions = self.policy[policy_id](obs)
        if LR_conversion:
            actions = self.action_conversion(actions)
        actions = torch.clip(actions, -self.env_cfgs[policy_id]["clip_actions"], self.env_cfgs[policy_id]["clip_actions"])
        self.mu, self.omega, self.psi, self.r, self.r_dot, self.r_ddot, self.theta, self.phi = self.envs[policy_id].CPG(actions, self.a_list[policy_id], self.r, self.r_dot, self.r_ddot, self.theta, self.phi, self.env_cfgs[policy_id], dt=self.dt)
        foot_pos = self.envs[policy_id].Trajectory(self.d_list[policy_id], self.h_list[policy_id], self.gc_list[policy_id], self.gp_list[policy_id], self.env_cfgs[policy_id].get("x_offset", [0.0, 0.0, 0.0, 0.0]), self.env_cfgs[policy_id].get("y_offset", [0.0, 0.0, 0.0, 0.0]), self.env_cfgs[policy_id].get("h_offset", [0.0, 0.0, 0.0, 0.0]), self.r, self.theta, self.phi, use_eth_z=self.env_cfgs[policy_id].get("eth_z", False))
        target_dof_pos = self.envs[policy_id].Inverse_Kinematics(foot_pos)



        policy_id = 0

        res_obs = self.obs(dof_pos, dof_vel, base_euler, base_ang_vel, foot_contact, commands, policy_id, LR_conversion=LR_conversion, residual=True)
        res_action = self.residual_policy[policy_id](res_obs)
        if LR_conversion:
            res_action = self.joint_conversion(res_action)
        self.res = res_action * self.residual_res_scale_list[policy_id]
        self.res = torch.clip(self.res, -self.residual_d_res_scale_list[policy_id]*self.dt + self.last_res, self.residual_d_res_scale_list[policy_id]*self.dt + self.last_res)
        self.res = torch.clip(self.res, -self.residual_res_scale_list[policy_id], self.residual_res_scale_list[policy_id])
        self.d_res = (self.res - self.last_res) / self.dt
        self.last_res = self.res

        self.res_history = torch.cat([self.res_history[1:], self.res.unsqueeze(0)], dim=0)

        exec_res = torch.mean(self.res_history, dim=0)



        return target_dof_pos.detach().to("cpu").numpy(), exec_res.detach().to("cpu").numpy()
    

    def obs(self, _dof_pos, _dof_vel, _base_euler, _base_ang_vel, _foot_contact, _commands, policy_id, LR_conversion=False, residual=False):

        if residual:
            obs_cfg = self.residual_obs_cfgs[policy_id]
            obs_scales = self.residual_obs_scales_list[policy_id]
            commands_scale = self.residual_commands_scale_list[policy_id]
        else:
            obs_cfg = self.obs_cfgs[policy_id]
            obs_scales = self.obs_scales_list[policy_id]
            commands_scale = self.commands_scale_list[policy_id]
        if LR_conversion:
            dof_pos = self.joint_conversion(_dof_pos)
            dof_vel = self.joint_conversion(_dof_vel)
            base_euler = self.roll_pitch_yaw_conversion([_base_euler[0],_base_euler[1],0.0])
            base_ang_vel = self.roll_pitch_yaw_conversion(_base_ang_vel)
            foot_contact = self.leg_conversion(_foot_contact)
            commands = torch.tensor([_commands[0],-_commands[1],-_commands[2]], device=self.device, dtype=torch.float32)

            mu = self.leg_conversion(self.mu)
            omega = self.leg_conversion(self.omega)
            psi = self.leg_conversion(-self.psi)
            r = self.leg_conversion(self.r)
            theta = self.leg_conversion(self.theta)
            phi = self.leg_conversion(-self.phi)
            phi = phi % (2*torch.pi)

            res = self.joint_conversion(self.res)
            d_res = self.joint_conversion(self.d_res)

            

        else:
            dof_pos = torch.tensor(_dof_pos, device=self.device, dtype=torch.float32)
            dof_vel = torch.tensor(_dof_vel, device=self.device, dtype=torch.float32)
            base_euler = torch.tensor(_base_euler, device=self.device, dtype=torch.float32)
            base_ang_vel = torch.tensor(_base_ang_vel, device=self.device, dtype=torch.float32)
            foot_contact = torch.tensor(_foot_contact, device=self.device, dtype=torch.float32)
            commands = torch.tensor(_commands, device=self.device, dtype=torch.float32)

            mu = self.mu
            omega = self.omega
            psi = self.psi
            r = self.r
            theta = self.theta
            phi = self.phi

            res = self.res
            d_res = self.d_res


        # Observations
        obs_components = []
        if obs_cfg.get("include_dof_pos", False):
            obs_components.append((dof_pos - self.default_dof_pos_list[policy_id]) * obs_scales["dof_pos"])  # 12

        if obs_cfg.get("include_dof_vel", False):
            obs_components.append(dof_vel * obs_scales["dof_vel"])  # 12

        if obs_cfg.get("include_euler", False):
            obs_components.append(base_euler[0:2] * obs_scales["euler"])  # 2

        if obs_cfg.get("include_ang_vel", False):
            obs_components.append(base_ang_vel * obs_scales["ang_vel"])  # 3

        if obs_cfg.get("include_foot_contact", False):
            obs_components.append(foot_contact * obs_scales["foot_contact"])  # 4

        # CPG state
        if obs_cfg.get("include_mu", False):
            obs_components.extend([
                mu * obs_scales["mu"],  # 4
            ])
        if obs_cfg.get("include_omega", False):
            obs_components.extend([
                omega * obs_scales["omega"],  # 4
            ])
        if obs_cfg.get("include_psi", False):
            obs_components.extend([
                psi * obs_scales["psi"],  # 4
            ])
        if obs_cfg.get("include_r", False):
            obs_components.extend([
                r * obs_scales["r"],  # 4
            ])
        if obs_cfg.get("include_theta", False):
            obs_components.extend([
                theta * obs_scales["theta"],  # 4
            ])
        if obs_cfg.get("include_phi", False):
            obs_components.extend([
                phi * obs_scales["phi"],  # 4
            ])

        # command
        if obs_cfg.get("include_commands", False):
            obs_components.append(commands * commands_scale)  # 3

        

        if obs_cfg.get("include_res", False):
            obs_components.extend([
                res * obs_scales["res"],  # 4
            ])
        
        if obs_cfg.get("include_d_res", False):
            obs_components.extend([
                d_res * obs_scales["d_res"],  # 4
            ])

        obs_buf = torch.cat(obs_components, axis=-1)

        return obs_buf
    
    def joint_conversion(self, joint_data):
        conversioned = torch.tensor([-joint_data[3], joint_data[4], joint_data[5],
                                 -joint_data[0], joint_data[1], joint_data[2],
                                 -joint_data[9], joint_data[10], joint_data[11],
                                 -joint_data[6], joint_data[7], joint_data[8]], device=self.device, dtype=torch.float32)
        return conversioned
    
    def roll_pitch_yaw_conversion(self, roll_pitch_yaw):
        conversioned = torch.tensor([-roll_pitch_yaw[0], roll_pitch_yaw[1], -roll_pitch_yaw[2]], device=self.device, dtype=torch.float32)
        return conversioned
    
    def leg_conversion(self, leg_data):
        conversioned = torch.tensor([leg_data[1], leg_data[0], leg_data[3],leg_data[2]], device=self.device, dtype=torch.float32)
        return conversioned
    
    def action_conversion(self, action):
        conversioned = torch.tensor([action[1], action[0], action[3], action[2], action[5], action[4], action[7], action[6], -action[9], -action[8], -action[11], -action[10]], device=self.device, dtype=torch.float32)
        return conversioned
        



class A1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, device="cuda:0",eval=False):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.eval = eval
        

        
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device, dtype=torch.float32)

        

    def CPG(self, actions, a, r, r_dot, r_ddot, theta, phi, env_cfg, dt=0.02):

        mu = (env_cfg["action_scale_mu"][1] - env_cfg["action_scale_mu"][0])/2 * (actions[:4]) + (env_cfg["action_scale_mu"][0]+env_cfg["action_scale_mu"][1])/2
        omega = (env_cfg["action_scale_omega"][1] - env_cfg["action_scale_omega"][0])/2 * (actions[4:8]) + (env_cfg["action_scale_omega"][0]+env_cfg["action_scale_omega"][1])/2
        psi = (env_cfg["action_scale_psi"][1] - env_cfg["action_scale_psi"][0])/2 * (actions[8:12]) + (env_cfg["action_scale_psi"][0]+env_cfg["action_scale_psi"][1])/2


        mu = torch.clip(mu, env_cfg["action_scale_mu"][0], env_cfg["action_scale_mu"][1])
        omega = torch.clip(omega, env_cfg["action_scale_omega"][0], env_cfg["action_scale_omega"][1])
        psi = torch.clip(psi, env_cfg["action_scale_psi"][0], env_cfg["action_scale_psi"][1])
    

        r_ddot = a*(a*(mu - r)/4 - r_dot)*dt
        r_dot = r_dot + r_ddot*dt
        r = r + r_dot*dt
        theta = theta + 2*torch.pi*omega*dt
        phi = phi + 2*torch.pi*psi*dt
        theta = theta % (2*torch.pi)
        phi = phi % (2*torch.pi)

        return mu, omega, psi, r, r_dot, r_ddot, theta, phi

    def Trajectory(self, d, h, gc, gp, x_offset, y_offset, h_offset, r,theta, phi, use_eth_z=False):
        # Determine g based on the sign of sin(theta)
        g = torch.where(torch.sin(theta) > 0, gc, gp)

        # Calculate x, y, and z positions
        x = -d * (r-1) * torch.cos(theta) * torch.cos(phi)
        y = -d * (r-1) * torch.cos(theta) * torch.sin(phi)
        # A1
        yr = -0.0838
        yl = 0.0838
        # Go2
        # yr = -0.094
        # yl = 0.094
        
        if use_eth_z:

            k = 2 * theta / np.pi

            # 条件に基づいてzを計算
            condition1 = (0 <= theta) & (theta <= np.pi / 2)
            condition2 = (np.pi / 2 < theta) & (theta <= np.pi)

            z = torch.where(
                condition1,
                -h + gc * (-2 * k**3 + 3 * k**2),
                torch.where(
                    condition2,
                    -h + gc * (2 * (k - 1)**3 - 3 * (k - 1)**2 + 1),
                    -h
                )
            )

        else:
            z = -h + g * torch.sin(theta)


        # Construct the trajectory positions
        foot_pos = torch.stack([
            torch.stack([x[0]+x_offset[0], y[0] + yr, z[0]+h_offset[0]], dim=-1),
            torch.stack([x[1]+x_offset[1], y[1] + yl, z[1]+h_offset[1]], dim=-1),
            torch.stack([x[2]+x_offset[2], y[2] + yr, z[2]+h_offset[2]], dim=-1),
            torch.stack([x[3]+x_offset[3], y[3] + yl, z[3]+h_offset[3]], dim=-1)
        ], dim=0)

        return foot_pos 
    

    def Inverse_Kinematics(self, target_positions: torch.Tensor) -> torch.Tensor:
        # A1
        L1 = 0.0838
        L2 = 0.2
        L3 = 0.2
        # Go2
        # L1 = 0.094
        # L2 = 0.21
        # L3 = 0.21

        # Initialize a tensor to store joint angles
        joint_angles = torch.zeros(target_positions.shape[:-1] + (3,), device=target_positions.device)
        
        # Calculate th1 for right and left legs
        th_f_yz_right = torch.atan2(-target_positions[0::2, 2], -target_positions[0::2, 1])
        th1_right = th_f_yz_right - torch.acos(L1 / torch.sqrt(target_positions[0::2, 1]**2 + target_positions[0::2, 2]**2))
        
        th_f_yz_left = torch.atan2(target_positions[1::2, 2], target_positions[1::2, 1])
        th1_left = th_f_yz_left + torch.acos(L1 / torch.sqrt(target_positions[1::2, 1]**2 + target_positions[1::2, 2]**2))
        
        # Assign th1 to joint_angles
        joint_angles[0::2, 0] = th1_right
        joint_angles[1::2, 0] = th1_left
        
        # Calculate rotated target positions
        cos_th1 = torch.cos(joint_angles[:, 0])
        sin_th1 = torch.sin(joint_angles[:, 0])
        
        rotated_target_pos = torch.stack([
            target_positions[:, 0],
            cos_th1 * target_positions[:, 1] + sin_th1 * target_positions[:, 2],
            -sin_th1 * target_positions[:, 1] + cos_th1 * target_positions[:, 2]
        ], dim=-1)
        
        # Calculate phi
        phi = torch.acos((L2**2 + L3**2 - rotated_target_pos[:, 0]**2 - rotated_target_pos[:, 2]**2) / (2 * L2 * L3))
        
        # Calculate th2 and th3
        th_f_xz = torch.atan2(-rotated_target_pos[:, 0], -rotated_target_pos[:, 2])
        th2 = th_f_xz + (torch.pi - phi) / 2
        th3 = -torch.pi + phi
        
        # Assign th2 and th3 to joint_angles
        joint_angles[:, 1] = th2
        joint_angles[:, 2] = th3

        # print(joint_angles)
        
        return joint_angles.view(12)
    
    def reset(self):
        return self.obs_buf, None

        
if __name__ == "__main__":
    # Sim to Sim
    import time
    import numpy as np
    import mujoco
    import mujoco.viewer
    from Sim2Sim.model_xml import FlatXml, BoxXml
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    colors = ["#fafcfc","#74878f",]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom",colors)

    test_id = test_id_list[0]

    policy = Policy()
    Kp = np.array([policy.env_cfgs[test_id]["kp"]])
    Kd = np.array([policy.env_cfgs[test_id]["kd"]])

    if terrain_type == "flat":
        xml = FlatXml(friction=1.2,skelton=False)
    elif terrain_type == "box":
        xml = BoxXml(friction=0.8,box_size=0.3,height=0.0,num_row=n_rows,num_col=n_cols,noise=box_noise,down=True,max_noise=0.06,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=False,cmap=cmap,vmin=0.0,vmax=1.0)


    def PD_control(Kp: np.ndarray, Kd: np.ndarray,target_q: np.ndarray, joint_q: np.ndarray, joint_dq: np.ndarray) -> np.ndarray:
        torques = Kp * (target_q -joint_q)  - Kd * joint_dq
        return torques
    
    def Joint_q_dq(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:

        j_names = ["FR_hip_", "FR_thigh_", "FR_calf_", "FL_hip_", "FL_thigh_", "FL_calf_",
                "RR_hip_", "RR_thigh_", "RR_calf_", "RL_hip_", "RL_thigh_", "RL_calf_"]

        joint_q = np.array([data.sensor(j_name + "pos").data[0] for j_name in j_names])
        joint_dq = np.array([data.sensor(j_name + "vel").data[0] for j_name in j_names])

        return joint_q, joint_dq
    
    def IMU(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quaternion = np.array(data.sensor("Body_Quat").data)
        angular_vel = np.array(data.sensor("Body_Gyro").data)
        linear_acc = np.array(data.sensor("Body_Acc").data)
        linear_vel = np.array(data.sensor("Body_Vel").data)
        return quaternion, angular_vel, linear_acc, linear_vel

    def Foot_force(data: mujoco.MjData) -> np.ndarray:
        foot_force = [1 if data.sensor("FR_foot").data[0] > 0 else 0,
                        1 if data.sensor("FL_foot").data[0] > 0 else 0,
                        1 if data.sensor("RR_foot").data[0] > 0 else 0,
                        1 if data.sensor("RL_foot").data[0] > 0 else 0]
        # foot_force = [data.sensor("FR_foot").data[0],
        #                 data.sensor("FL_foot").data[0],
        #                 data.sensor("RR_foot").data[0],
        #                 data.sensor("RL_foot").data[0]]
        return np.array(foot_force)
    
    def Quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        R = np.array([
            [1 - 2 * (q[2]**2 + q[3]**2), 2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[1]*q[3] + q[0]*q[2])],
            [2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[1]**2 + q[3]**2), 2 * (q[2]*q[3] - q[0]*q[1])],
            [2 * (q[1]*q[3] - q[0]*q[2]), 2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2)]
        ])
        return R

    def Rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        # ヨー角を計算
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # ピッチ角を計算
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))

        # ロール角を計算
        roll = np.arctan2(R[2, 1], R[2, 2])

        return np.array([roll, pitch, yaw])

    
    model = mujoco.MjModel.from_xml_string(xml)

    data = mujoco.MjData(model)

    data.joint("free").qpos = [0.0,0,0.30,1,0,0,0]

    for leg in ["FR", "FL", "RR", "RL"]:
        for joint in ["hip", "thigh", "calf"]:
            if joint == "hip":
                data.joint(f"{leg}_{joint}_joint").qpos = [0.0]
            elif joint == "thigh":
                data.joint(f"{leg}_{joint}_joint").qpos = [0.8]
            elif joint == "calf":
                data.joint(f"{leg}_{joint}_joint").qpos = [-1.6]


    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0
        start_time = time.time()

        command_change_time = 5
        command_change_count = 0

        while viewer.is_running():
            
            if sim_time > command_change_time:
                if command_change_count % 2 == 0:
                    command_change_count += 1
                    command_change_time += 5
                    
                    if (len(test_id_list) > 1):
                        test_id = random.choice(test_id_list[1:])
                        print(f"test_id: {test_id}, command: {test_commands_list[test_id]}")
                    else:
                        test_id = test_id_list[0]
                    
                else:
                    command_change_count += 1
                    command_change_time += 3
                    test_id = 0
                    # policy.reset(residual_reset=True)


            sim_time += model.opt.timestep*20

            joint_q, joint_dq = Joint_q_dq(data)
            quaternion, angular_vel, linear_acc, linear_vel = IMU(data)
            foot_contact = Foot_force(data)
            R = Quaternion_to_rotation_matrix(quaternion)
            base_euler = Rotation_matrix_to_euler(R)
            base_euler = np.degrees(base_euler)

            target_q, res = policy.get_target_q(joint_q, joint_dq, base_euler[0:2], angular_vel, foot_contact, commands=test_commands_list[test_id], policy_id=test_id, LR_conversion=test_LR_conversion)

            for i in range(20):
                joint_q, joint_dq = Joint_q_dq(data)
                if test_residual:
                    torques = PD_control(Kp, Kd, target_q+res, joint_q, joint_dq)
                else:
                    torques = PD_control(Kp, Kd, target_q, joint_q, joint_dq)
                data.ctrl[:] = torques

                mujoco.mj_step(model, data)
            
            viewer.sync()

            time_until_next_step = sim_time - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
