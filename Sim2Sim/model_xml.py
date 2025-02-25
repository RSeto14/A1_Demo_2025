import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


script_dir = os.path.dirname(__file__)    # .../A1_WS/Envs
parent_dir = os.path.dirname(script_dir)  # .../A1_WS
sys.path.append(parent_dir)

from Sim2Sim.a1_xml import a1_xml
# from a1_xml_brax import a1_xml



def FlatXml(friction=0.8,skelton=True):
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_xml(skeleton=skelton)}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>-->
            <!-- <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>-->
            <!--<texture type="2d" name="groundplane" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.99 0.99 0.99"  width="300" height="300"/>-->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.8157 0.8314 0.8392" rgb2="0.8157 0.8314 0.8392"  width="500" height="500" mark="edge" markrgb="0.1 0.1 0.1" />
            
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}" />
        </worldbody>

    </mujoco>
    """
    return xml

def SlopeXml(friction=0.8,skelton=True,roll=0.0,pitch=0.0,yaw=0.0):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    quat = [np.cos(yaw/2)*np.cos(pitch/2)*np.cos(roll/2) + np.sin(yaw/2)*np.sin(pitch/2)*np.sin(roll/2),
            np.cos(yaw/2)*np.cos(pitch/2)*np.sin(roll/2) - np.sin(yaw/2)*np.sin(pitch/2)*np.cos(roll/2),
            np.cos(yaw/2)*np.sin(pitch/2)*np.cos(roll/2) + np.sin(yaw/2)*np.cos(pitch/2)*np.sin(roll/2),
            np.sin(yaw/2)*np.cos(pitch/2)*np.cos(roll/2) - np.cos(yaw/2)*np.sin(pitch/2)*np.sin(roll/2)]
    
    xml = f"""
    <mujoco model="a1 Slope Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_xml(skeleton=skelton)}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>-->
            <!-- <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>-->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.99 0.99 0.99"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}" rgba="1.0 1.0 1.0 1" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>
        </worldbody>

    </mujoco>
    """
    return xml

def DekobokoXml(height,png_id,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,friction=0.8):
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
        
    xml = f"""
    <mujoco model="a1 Dekoboko">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_model}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
            <hfield name="uneven"  file="{script_dir}/terrain/dekoboko{png_id}.png" nrow="0" ncol="0" size="30 30 {height} 0.05" />
        </asset>

        <worldbody>
            <geom conaffinity="1" condim="3" hfield="uneven" material="groundplane" name="floor" pos="0 0 0" friction="{friction}" rgba="0.8 0.9 0.8 1" type="hfield" />
        </worldbody>

    </mujoco>
    """
    return xml

def MazeXml(friction=0.8,maze=[[0,0,1,0,0,0],[0,0,0,1,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0]],box_size=0.2):
    num_col = len(maze[0])
    num_row = len(maze)
    boxes_xml = ""
    for i in range(num_col):
        for j in range(num_row):
            if maze[j][i] == 0:
                continue
            else:
                pos_x = -num_col/2 * box_size + box_size/2 + box_size * i
                pos_y = -num_row/2 * box_size + box_size/2 + box_size * j
                rgba = f"{0.5 * i/num_col + 0.1} {0.5 * j/num_row + 0.1} 0.0 1"
                boxes_xml += f"""
                <body name="box{i*num_col+j+1}_parent" pos="0 0 0.05">
                    <body name="box{i*num_col+j+1}_body" pos="{pos_x} {pos_y} 0.0">
                        <geom name="box{i*num_col+j+1}" type="box" size="{box_size/2-0.001} {box_size/2-0.001} 0.05" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
                    </body>
                </body>
                """
      
    boxes_xml +=f"""
        <body name="box_parent" pos="0 0 0.05">
            <geom name="_box1" type="box" size="{box_size*num_col/2+0.01} 0.005 0.05" pos="0 {box_size*num_row/2 + 0.005} 0.0" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box2" type="box" size="{box_size*num_col/2+0.01} 0.005 0.05" pos="0 {-box_size*num_row/2 - 0.005} 0.0" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box3" type="box" size="0.005 {box_size*num_row/2} 0.05" pos="{box_size*num_col/2 + 0.005} 0 0.0" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box4" type="box" size="0.005 {box_size*num_row/2} 0.05" pos="{-box_size*num_col/2 - 0.005} 0 0.0" rgba="0.2 0.2 0.2 1"/>
        </body>
    """
    
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->


        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <light name="spotlight1" pos="-5 5 10" diffuse="0.8 0.8 0.8"/>
            <light name="spotlight2" pos="3 -5 10" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}" rgba="0.8 0.9 0.8 1"/>
            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml

def MoveXml(friction=0.8, num_col=10, num_row=10, box_size=0.2):

    
    # Generate the 64 boxes with slide joints
    boxes_xml = ""
    for i in range(num_col):
        for j in range(num_row):
            # pos_x = -1.05 + 0.3 * i
            # pos_y = -1.05 + 0.3 * j
            pos_x = -num_col/2 * box_size + box_size/2 + box_size * i
            pos_y = -num_row/2 * box_size + box_size/2 + box_size * j
            rgba = f"{0.5 * i/num_col + 0.1} {0.5 * j/num_row + 0.1} 0.0 1"
            box_id = i * num_row + j + 1  # Unique ID for each box
            boxes_xml += f"""
            <body name="box{box_id}_parent" pos="0 0 -0.1">
                <body name="box{box_id}_body" pos="{pos_x} {pos_y} 0.0">
                    <geom name="box{box_id}" type="box" size="{box_size/2-0.001} {box_size/2-0.001} 0.1" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
                    <joint name="box{box_id}_slide" type="slide" axis="0 0 1" range="-0.1 0.1"/>
                </body>
            </body>
            """
    # 
    boxes_xml +=f"""
        <body name="box_parent" pos="0 0 0.1">
            <geom name="_box1" type="box" size="{box_size*num_col/2+0.01} 0.005 0.1" pos="0 {box_size*num_row/2 + 0.005} -0.1" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box2" type="box" size="{box_size*num_col/2+0.01} 0.005 0.1" pos="0 {-box_size*num_row/2 - 0.005} -0.1" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box3" type="box" size="0.005 {box_size*num_row/2} 0.1" pos="{box_size*num_col/2 + 0.005} 0 -0.1" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box4" type="box" size="0.005 {box_size*num_row/2} 0.1" pos="{-box_size*num_col/2 - 0.005} 0 -0.1" rgba="0.2 0.2 0.2 1"/>
            <geom name="_box5" type="box" size="{box_size*num_col/2+0.01} 0.2 0.2" pos="0 {box_size*num_row/2 + 0.01 + 0.4*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" quat="{np.cos(np.pi/8)} {np.sin(np.pi/8)} 0 0"/>
            <geom name="_box6" type="box" size="{box_size*num_col/2+0.01} 0.2 0.2" pos="0 {-box_size*num_row/2 - 0.01 - 0.4*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" quat="{np.cos(np.pi/8)} {np.sin(np.pi/8)} 0 0"/>
            <geom name="_box7" type="box" size="0.2 {box_size*num_row/2+0.01} 0.2" pos="{box_size*num_col/2 + 0.01 + 0.4*np.sin(np.pi/4)} 0 0.0" rgba="0.2 0.2 0.2 1" quat="{np.cos(np.pi/8)} 0 {np.sin(np.pi/8)} 0"/>
            <geom name="_box8" type="box" size="0.2 {box_size*num_row/2+0.01} 0.2" pos="{-box_size*num_col/2 - 0.01 - 0.4*np.sin(np.pi/4)} 0 0.0" rgba="0.2 0.2 0.2 1" quat="{np.cos(np.pi/8)} 0 {np.sin(np.pi/8)} 0"/>
            <geom name="_box9" type="box" size="{0.2*np.sin(np.pi/4)} {0.2*np.sin(np.pi/4)} {0.4*np.sin(np.pi/4)}" pos="{box_size*num_col/2 + 0.01 + 0.2*np.sin(np.pi/4)} {box_size*num_row/2 + 0.01 + 0.2*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" />
            <geom name="_box10" type="box" size="{0.2*np.sin(np.pi/4)} {0.2*np.sin(np.pi/4)} {0.4*np.sin(np.pi/4)}" pos="{-box_size*num_col/2 - 0.01 - 0.2*np.sin(np.pi/4)} {box_size*num_row/2 + 0.01 + 0.2*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" />
            <geom name="_box11" type="box" size="{0.2*np.sin(np.pi/4)} {0.2*np.sin(np.pi/4)} {0.4*np.sin(np.pi/4)}" pos="{box_size*num_col/2 + 0.01 + 0.2*np.sin(np.pi/4)} {-box_size*num_row/2 - 0.01 - 0.2*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" />
            <geom name="_box12" type="box" size="{0.2*np.sin(np.pi/4)} {0.2*np.sin(np.pi/4)} {0.4*np.sin(np.pi/4)}" pos="{-box_size*num_col/2 - 0.01 - 0.2*np.sin(np.pi/4)} {-box_size*num_row/2 - 0.01 - 0.2*np.sin(np.pi/4)} 0.0" rgba="0.2 0.2 0.2 1" />
        </body>
    """
            # print(boxes_xml)
    
    xml = f"""
    <mujoco model="a1 Dekoboko">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
    
        </asset>

        <worldbody>
            <light name="spotlight1" pos="-3 3 2" diffuse="0.8 0.8 0.8"/>
            <light name="spotlight2" pos="3 -3 2" diffuse="0.8 0.8 0.8"/>
            <!-- Add 64 boxes of size 0.15m x 0.15m x 0.1m with different colors -->
            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml

# def StepXml(friction=0.8,start=0.5,num_step=10,width=0.2,height=0.1,length=5):
#     # a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
#     boxes_xml =""
#     for i in range(num_step):
#         pos_x = start + width/2 + i* width
#         poz_z = (i+1) * height/2 + 0.1
#         boxes_xml += f"""
#         <body name="box{i}_parent" pos="0 0 -0.1">
#             <body name="box{i}_body" pos="{pos_x} 0 {poz_z}">
#                 <geom name="box{i}" type="box" size="{width/2} {length/2} {(i+1)*height/2}" rgba="0.5 0.5 0.5 1" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
#             </body>
#         </body>
#         """
    
#     xml = f"""
#     <mujoco model="a1 Flat Ground">
#         <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->


#         <statistic center="0 0 0.1" extent="0.8"/>

#         <visual>
#             <rgba haze="0.15 0.25 0.35 1"/>
#             <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
#         </visual>

#         <asset>
#             <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
#             <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
#             <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
#         </asset>

#         <worldbody>
#             <light name="spotlight1" pos="-3 3 2" diffuse="0.8 0.8 0.8"/>
#             <light name="spotlight2" pos="3 -3 2" diffuse="0.8 0.8 0.8"/>
#             <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}" rgba="0.8 0.9 0.8 1"/>
#             {boxes_xml}
#         </worldbody>

#     </mujoco>
#     """
#     return xml

# def Step2Xml(friction=0.8,num_step=10,width=0.2,height=0.1,num_row=5,num_col=5,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True):
    
#     # a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
#     size = 2 * width * num_step
#     boxes_xml =""
#     for row in range(num_row):
#         for col in range(num_col):
#             for i in range(num_step):
#                 pos_x = 0
#                 pos_y = 0
#                 pos_x = -num_col/2 * size + size/2 + size * col
#                 pos_y = -num_row/2 * size + size/2 + size * row
#                 poz_z = (2*i+1) * height/2 + 0.1
#                 boxes_xml += f"""
#                 <body name="box{row}_{col}_{i}_parent" pos="0 0 -0.1">
#                     <body name="box{row}_{col}_{i}_body" pos="{pos_x} {pos_y} {poz_z}">
#                         <geom name="box{row}_{col}_{i}" type="box" size="{size/2 - i*width} {size/2 - i*width} {height/2}" rgba="0.5 0.5 0.5 1" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
#                     </body>
#                 </body>
#                 """
    
#     xml = f"""
#     <mujoco model="a1 Flat Ground">
#         <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

#         <statistic center="0 0 0.1" extent="0.8"/>

#         <visual>
#             <rgba haze="0.15 0.25 0.35 1"/>
#             <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
#         </visual>

#         <asset>
#             <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
#             <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
#             <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
#         </asset>

#         <worldbody>
#             <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}" rgba="0.8 0.9 0.8 1"/>
#             {boxes_xml}
#         </worldbody>

#     </mujoco>
#     """
#     return xml

def StepXml(friction=0.8,step_length=20,step_height=0.1,step_width=0.4,num_step=10,_num_step=5,down=True,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True):
    
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)

    step_xml = ""
    
    for i in range(num_step):
        pos_x = step_width * i
        pos_z = i * step_height
        color = f"{0.5 * (i+_num_step)/(num_step-1+_num_step) + 0.1} {0.5 - 0.5 * (i+_num_step)/(num_step-1+_num_step) + 0.1} 0.0 1"
        
        if down:
            pos_z *= -1
        step_xml += f"""
        <body name="box{i}_parent" pos="0 0 0">
            <body name="box{i}_body" pos="{pos_x} 0 {pos_z}">
                <geom name="box{i}" type="box" size="{step_width/2} {step_length/2} {step_height/2}" rgba="{color}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
            </body>
        </body>
        """
    for i in range(_num_step):
        pos_x = -step_width * (i+1)
        pos_z = (i+1) * step_height
        if not down:
            pos_z *= -1
        color = f"{0.5 * (-1-i+_num_step)/(num_step-1+_num_step) + 0.1} {0.5 - 0.5 * (-1-i+_num_step)/(num_step-1+_num_step) + 0.1} 0.0 1"
        step_xml += f"""
        <body name="box{-(i+1)}_parent" pos="0 0 0">
            <body name="box{-(i+1)}_body" pos="{pos_x} 0 {pos_z}">
                <geom name="box{-(i+1)}" type="box" size="{step_width/2} {step_length/2} {step_height/2}" rgba="{color}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
            </body>
        </body>
        """
    
    xml = f"""
    <mujoco model="a1 Step Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->
        
        {a1_model}


        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            {step_xml}
        </worldbody>

    </mujoco>
    """
    return xml
    
    

def BoxXml(friction=0.8,box_size=0.3,height=0.1,num_row=5,num_col=5,noise=0.01,down=True,max_noise=0.06,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,cmap=plt.get_cmap("viridis"),vmin=0.0,vmax=1.0):
    
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    max_height = max((num_col-1)/2, (num_row-1)/2)
    
    boxes_xml =""
    for i in range(num_col):
        for j in range(num_row):

            pos_x = -num_col/2 * box_size + box_size/2 + box_size * i
            pos_y = -num_row/2 * box_size + box_size/2 + box_size * j
            
            if down:
                h = height*(max_height - max(abs(i - (num_col-1)/2), abs(j - (num_row-1)/2))) + 0.01 + noise + noise * np.random.uniform(-1, 1)
                pos_z = h/2 - max_height*height - 0.01 -  noise
            else:
                h = height*max(abs(i - (num_col-1)/2), abs(j - (num_row-1)/2)) + 0.01 + noise + noise * np.random.uniform(-1, 1)
                pos_z = h/2 -  0.01 -  noise
            # rgba = f"{0.5 * i/num_col + 0.1} {0.5 * j/num_row + 0.1} 0.0 1"
            # rgba = f"{0.5 * (h-0.01)/(max_height*height+2*noise) + 0.1} {0.5 *(1-(h-0.01)/(max_height*height+2*noise)) + 0.1} 0.0 1"
            rgba = value_to_rgba((h-0.01)/(max_height*height+2*max_noise),cmap=cmap,vmin=vmin,vmax=vmax)
            color = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
            boxes_xml += f"""
            <body name="box_{i}_{j}_parent" pos="0 0 0.0">
                <body name="box_{i}_{j}_body" pos="{pos_x} {pos_y} {pos_z}">
                    <geom name="box_{i}_{j}" type="box" size="{box_size/2-0.001} {box_size/2-0.001} {h/2}" rgba="{color}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
                </body>
            </body>
            """
      
    if skeleton:
        light_xml = """
        <light name="spotlight_box1" pos="-5 5 10" diffuse="0.8 0.8 0.8"/>
        <light name="spotlight_box2" pos="3 -5 10" diffuse="0.8 0.8 0.8"/>
        """
    else:
        light_xml = ""
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->
        
        {a1_model}


        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            {light_xml}
            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml


def SoftBoxXml(friction=0.8,box_size=0.3,num_row=5,num_col=5,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,stiffness=2000,damping=1):
    
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    max_height = max((num_col-1)/2, (num_row-1)/2)
    
    boxes_xml =""
    for i in range(num_col):
        for j in range(num_row):

            pos_x = -num_col/2 * box_size + box_size/2 + box_size * i
            pos_y = -num_row/2 * box_size + box_size/2 + box_size * j
            
            # rgba = f"{0.5 * i/num_col + 0.1} {0.5 * j/num_row + 0.1} 0.0 1"
            rgba = "0.4549 0.5294 0.5608 1.0"
            
            boxes_xml += f"""
            <body name="box_{i}_{j}_parent" pos="0 0 0.0">
                <body name="box_{i}_{j}_body" pos="{pos_x} {pos_y} 0.0">
                    <geom name="box_{i}_{j}" type="box" size="{box_size/2-0.001} {box_size/2-0.001} {0.2/2}" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" mass="0.05"/>
                    <joint name="box_{i}_{j}_slide" type="slide" axis="0 0 1" range="-0.12 0.12" stiffness="{stiffness}" damping="{damping}"/>
                </body>
            </body>
            """
      
    
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->
        
        {a1_model}


        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <light name="fixed_light1" pos="-5 5 10" diffuse="0.8 0.8 0.8"/>
            <light name="fixed_light2" pos="3 -5 10" diffuse="0.8 0.8 0.8"/>
            
            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml

def OneStepXml(friction=0.8,down=True,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,h=0.12,
           max_height=0.12,cmap=mcolors.LinearSegmentedColormap.from_list("custom",["#d0d4d6","#74878f"]),vmin=0.0,vmax=1.0):
    # cmap=plt.get_cmap("viridis")
    
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    boxes_xml =""
    
    if down:
        _rgba = value_to_rgba(h/max_height,cmap=cmap,vmin=vmin,vmax=vmax)
        rgba = f"{_rgba[0]} {_rgba[1]} {_rgba[2]} {0.5}"
        
    else:
        _rgba = value_to_rgba(0/max_height,cmap=cmap,vmin=vmin,vmax=vmax)
        rgba = f"{_rgba[0]} {_rgba[1]} {_rgba[2]} {_rgba[3]}"
    

    
    boxes_xml += f"""
    <body name="box_1_parent" pos="0 0 0.0">
        <body name="box_1_body" pos="-4.5 0 0">
            <geom name="box_1" type="box" size="5.5 10 {h}" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
        </body>
    </body>
    """
    if down:
        _rgba = value_to_rgba(0/max_height,cmap=cmap,vmin=vmin,vmax=vmax)
        rgba = f"{_rgba[0]} {_rgba[1]} {_rgba[2]} {_rgba[3]}"
        pos_z = -h
    else:
        _rgba = value_to_rgba(h/max_height,cmap=cmap,vmin=vmin,vmax=vmax)
        rgba = f"{_rgba[0]} {_rgba[1]} {_rgba[2]} {0.5}"
        
        pos_z = h
    
    
    boxes_xml += f"""
    <body name="box_2_parent" pos="0 0 0.0">
        <body name="box_2_body" pos="5.5 0 {pos_z}">
            <geom name="box_2" type="box" size="4.5 10 {h}" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
        </body>
    </body>
    """
      
    
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_model}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <camera name="fixed1" mode="fixed" pos="1.172 -2.173 0.207" xyaxes="0.999 0.048 0.000 -0.007 0.136 0.991"/>
            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml

def value_to_rgba(value,cmap=plt.get_cmap("viridis"),vmin=0.0,vmax=1.0):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # 値が0から1の範囲内にあることを保証
    value = max(0, min(1, value))
    
    
    rgba = cmap(value*(1 - vmin - (1-vmax)) + vmin)
    
    # rgba = cmap(norm(value))
    # RGBAは0～1の範囲で返されるので、255スケールに変換する場合
    # rgba_255 = [int(255 * c) for c in rgba]
    
    return rgba

def cmap_bar(cmap=plt.get_cmap("viridis"), vmin=0.0, vmax=1.0, width=0.1, height=6):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # カラーマップを取得
    
    
    gradient = np.linspace(0, 1, 256)
    print(gradient)
    gradient = np.vstack((gradient, gradient))
    gradient = np.rot90(gradient,k=3)

    fig, ax = plt.subplots()
    # ax.set_xlim(vmin,vmax)
    ax.set_ylim(vmin*256,vmax*256)
    
    ax.imshow(gradient, aspect=0.2, cmap=cmap)
    ax.set_axis_off()

    plt.show()


def manipulation_xml():
    xml ="""
<mujoco model="6dof_manipulator">
    <compiler angle="degree"/>
    <option timestep="0.01"/>
    
    <worldbody>
        <!-- Base of the manipulator -->
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
            
            <!-- First joint and link -->
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.2 0.8 0.2 1"/>
                
                <!-- Second joint and link -->
                <body name="link2" pos="0 0 0.2">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-180 180"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.2 0.2 0.8 1"/>
                    
                    <!-- Third joint and link -->
                    <body name="link3" pos="0 0 0.2">
                        <joint name="joint3" type="hinge" axis="0 1 0" range="-180 180"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.8 0.8 0.2 1"/>
                        
                        <!-- Fourth joint and link -->
                        <body name="link4" pos="0 0 0.2">
                            <joint name="joint4" type="hinge" axis="0 1 0" range="-180 180"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.8 0.2 0.8 1"/>
                            
                            <!-- Fifth joint and link -->
                            <body name="link5" pos="0 0 0.2">
                                <joint name="joint5" type="hinge" axis="0 1 0" range="-180 180"/>
                                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.2 0.8 0.8 1"/>
                                
                                <!-- Sixth joint and link -->
                                <body name="link6" pos="0 0 0.2">
                                    <joint name="joint6" type="hinge" axis="0 1 0" range="-180 180"/>
                                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" rgba="0.8 0.5 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Position actuators for each joint -->
        <position joint="joint1" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
        <position joint="joint2" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
        <position joint="joint3" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
        <position joint="joint4" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
        <position joint="joint5" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
        <position joint="joint6" kp="100" ctrlrange="-180 180" ctrllimited="true"/>
    </actuator>
</mujoco>
    """
    return xml
