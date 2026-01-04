import math

# 智能体
class Agent:
    def __init__(self, position=None, speed=None, heading_angle=None):
        self.position = position
        self.speed = speed
        self.heading_angle = heading_angle
        self.type = None # 智能体类型
        self.name = None
        self.obs_dim = None
        self.act_dim = None
        self.theta1 = None # 前向角差
        self.theta2 = None # 后向角差
        self.epsilon = 0 # 速度与航向角差
        self.last_vector = None  # 智能体与目标连线矢量
        self.target_idx = 0  # 选择目标的序号
        self.best_idx = 0  # 群体最优目标的序号

# 障碍物
class Obstacle:
    def __init__(self, position=None, radius=None, color=None):
        self.position = position
        self.radius = radius
        self.color = color

# 目标
class Target:
    def __init__(self, position=None):
        self.position = position