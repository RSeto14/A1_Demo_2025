import pygame
import os
import numpy as np
import sys

"""
このスクリプトはXbox360コントローラの入力を取得するクラスを定義しています。
"""

class Xbox360Controller:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        try:
            self.joy = pygame.joystick.Joystick(0)  # create a joystick instance
            self.joy.init()  # init instance
            print(f"Joystick name: {self.joy.get_name()}")
        except pygame.error:
            print("No joystick found.")
            sys.exit()
        
        
        self.RT = -1
        self.LT = -1
        self.RB = 0
        self.LB = 0
        self.Rjoy_vert = 0
        self.Rjoy_horz = 0
        self.Ljoy_vert = 0
        self.Ljoy_horz = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.hat_vert = 0
        self.hat_horz = 0
        self.start = 0
        self.home = 0


    def check_buttons(self,threshold=0.2):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 5:  # ZRボタンが押された
                    self.RT = event.value
                    # print("RT button is pressed")
                if event.axis == 4:  # ZLボタンが押された
                    self.LT = event.value
                    # print("LT button is pressed")
                
        
            # Left joy stick
            self.Ljoy_vert = -self.joy.get_axis(1)
            self.Ljoy_horz = -self.joy.get_axis(0)
                

            # Right joy stick
            self.Rjoy_vert = -self.joy.get_axis(3)
            self.Rjoy_horz = -self.joy.get_axis(2)
                
            
                
                
            # 方向ボタン (ハットスイッチ) の処理を追加
            self.hat_horz = -self.joy.get_hat(0)[0]
            self.hat_vert = self.joy.get_hat(0)[1]
                
            self.RB = self.joy.get_button(5)
            self.LB = self.joy.get_button(4)
            self.A = self.joy.get_button(0)
            self.B = self.joy.get_button(1)
            self.X = self.joy.get_button(2)
            self.Y = self.joy.get_button(3)
            self.start = self.joy.get_button(7)
            self.home = self.joy.get_button(6)
            
            # filter
            self.RT = self.filter(self.RT,threshold)
            self.LT = self.filter(self.LT,threshold)
            self.Rjoy_vert = self.filter(self.Rjoy_vert,threshold)
            self.Rjoy_horz = self.filter(self.Rjoy_horz,threshold)
            self.Ljoy_vert = self.filter(self.Ljoy_vert,threshold)
            self.Ljoy_horz = self.filter(self.Ljoy_horz,threshold)
            self.hat_horz = self.filter(self.hat_horz,threshold)
            self.hat_vert = self.filter(self.hat_vert,threshold)
        
        
    def filter(self,x,threshold):
        if x > threshold:
            return np.round((x - threshold)/(1-threshold),3)
        elif x < -threshold:
            return np.round((x + threshold)/(1-threshold),3)
        else:
            return 0.000
            
    def print_buttons(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"RT: {self.RT}, LT: {self.LT}, RB: {self.RB}, LB: {self.LB}, Rjoy_vert: {self.Rjoy_vert}, Rjoy_horz: {self.Rjoy_horz}, Ljoy_vert: {self.Ljoy_vert}, Ljoy_horz: {self.Ljoy_horz}, A: {self.A}, B: {self.B}, X: {self.X}, Y: {self.Y}, hat_horz: {self.hat_horz}, hat_vert: {self.hat_vert}, start: {self.start}, home: {self.home}")
            
if __name__ == "__main__":
    import time
    controller = Xbox360Controller()
    while True:
        controller.check_buttons()
        controller.print_buttons()
        time.sleep(0.5)
