# 环境可视化设置
import matplotlib
# import cv2
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 解决中文显示和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为“黑体”
plt.rcParams['axes.unicode_minus'] = False   # 解决负号'-'显示为方块的问题
from typing import Tuple
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
from matplotlib.transforms import Affine2D

# import pygame
class MutiAgentEnv:
    metadata = {
        'render.modes': ['human', 'rgb_array']}

    def __init__(self, scene, reset_callback=None, reward_callback=None,
                 observation_callback=None, done_callback=None, info_callback=None):
        self.scene = scene
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.done_callback = done_callback
        self.info_callback = info_callback
        self.agents_num = self.scene.pursuer_num
        self.obs_n = [self.scene.pursuers[i].obs_dim for i in range(self.agents_num)]  # 这里设置观测空间的维度
        self.act_n = [self.scene.pursuers[i].act_dim for i in range(self.agents_num)]
        self.render_cnt = 0  # 只开一张画布
        self.ax = None
        self.fig = None
        self.cnt = 0  # 图像帧计数
        # 记录轨迹
        self.pursuers_x = [[] for _ in range(self.agents_num)]
        self.pursuers_y = [[] for _ in range(self.agents_num)]
        self.evaders_x = [[] for _ in range(self.scene.evader_num)]
        self.evaders_y = [[] for _ in range(self.scene.evader_num)]

    ## 更新环境
    def step(self, actions):
        obs_n = []
        reward_n = []
        done_n = []
        # 更新每个智能体的状态
        self.scene.world_update(actions)
        # 获取每个智能体的奖励和其它信息
        # 追击者奖励
        for i, agent in enumerate(self.scene.pursuers):
            if agent.type == 0:
                # 计算奖励
                if self.reward_callback is not None:
                    reward_n.append(self.reward_callback(agent))
                else:
                    reward_n.append(0.0)
                    print("pursuer reward error")
                # 获取观测值
                if self.observation_callback is not None:
                    obs_n.append(self.observation_callback(agent))
                else:
                    print("pursuers observation error")
                    obs_n.append(None)
                # 结束标志位
                if self.done_callback is not None:
                    done_n.append(self.done_callback(agent))
                else:
                    done_n.append(False)
            else:
                continue
        # 逃跑者奖励
        for i, agent in enumerate(self.scene.evaders):
            if agent.type == 0:
                # 计算奖励
                if self.reward_callback is not None:
                    reward_n.append(self.reward_callback(agent))
                else:
                    reward_n.append(0.0)
                    print("pursuer reward error")
                # 获取观测值
                if self.observation_callback is not None:
                    obs_n.append(self.observation_callback(agent))
                else:
                    print("evaders observation error")
                    obs_n.append(None)
                # 结束标志位
                if self.done_callback is not None:
                    done_n.append(self.done_callback(agent))
                else:
                    done_n.append(False)
            else:
                continue
        return obs_n, reward_n, done_n
    ## 重置环境
    def reset(self, mode, vel, scenario_id=0):
        obs_n = []
        # self.scene.reset_world(mode=mode, vel=vel)
        # 【修改】将 scenario_id 传递给底层的 reset_world
        self.scene.reset_world(mode=mode, vel=vel, scenario_id=scenario_id)
        # 获取每个智能体的奖励和其它信息
        for i, agent in enumerate(self.scene.pursuers):
            if agent.type == 0:
                # 获取观测值
                if self.observation_callback is not None:
                    obs_n.append(self.observation_callback(agent))
                else:
                    print("pursuers observation reset error")
                    obs_n.append(None)
            else:
                continue
        for i, agent in enumerate(self.scene.evaders):
            if agent.type == 0:
                # 获取观测值
                if self.observation_callback is not None:
                    obs_n.append(self.observation_callback(agent))
                else:
                    print("evaders observation reset error")
                    obs_n.append(None)
            else:
                continue
        # 重置轨迹参数
        self.pursuers_x = [[] for _ in range(self.agents_num)]
        self.pursuers_y = [[] for _ in range(self.agents_num)]

        self.evaders_x = [[] for _ in range(self.scene.evader_num)]
        self.evaders_y = [[] for _ in range(self.scene.evader_num)]
        self.cnt = 0  # 图像帧计数
        return obs_n
    def on_key_press(self, event):
        T = 0.2
        if event.key == 'up':
            v = 2
            self.scene.evaders[0].position[0] += v * np.cos(self.scene.evaders[0].heading_angle) * T
            self.scene.evaders[0].position[1] += v * np.sin(self.scene.evaders[0].heading_angle) * T
            print("up")
        if event.key == 'left':
            w = 1 / 3 * np.pi
            self.scene.evaders[0].heading_angle += w * T
            print("left")
        if event.key == 'right':
            w = 1 / 3 * np.pi
            self.scene.evaders[0].heading_angle -= w * T
            print("right")
        if event.key == 'up' and event.key == 'left':
            v = 2
            self.scene.evaders[0].position[0] += v * np.cos(self.scene.evaders[0].heading_angle) * T
            self.scene.evaders[0].position[1] += v * np.sin(self.scene.evaders[0].heading_angle) * T
            w = 1 / 3 * np.pi
            self.scene.evaders[0].heading_angle += w * T
        if self.scene.evaders[0].heading_angle > np.pi:
            self.scene.evaders[0].heading_angle -= 2 * np.pi
        elif self.scene.evaders[0].heading_angle < -np.pi:
            self.scene.evaders[0].heading_angle += 2 * np.pi

    ## 环境可视化
    def render(self, mode='human'):
        self.cnt += 1
        colors = ['red', 'green']
        # Load images for captor and runner agents
        captor_img = mpimg.imread('usv_p.png', 0)
        runner_img = mpimg.imread('usv_e.png', 0)
        captor_height, captor_width = captor_img.shape[:2]
        runner_height, runner_width = runner_img.shape[:2]
        # Calculate the aspect ratio
        captor_aspect_ratio = captor_width / captor_height  # 2
        runner_aspect_ratio = runner_width / runner_height
        if self.render_cnt == 0:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(6, 6), dpi=150) # figsize控制窗口大小, dpi控制清晰度
            # 连接键盘事件
            self.render_cnt += 1
            # self.ax = ax
        else:
            fig = self.fig
            ax = self.ax

        # 绑定键盘事件处理函数
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # Create figure and axis
        # _, ax = plt.subplots()
        # Set axis limits
        ax.set_xlim(-10, 120)
        ax.set_ylim(-10, 120)
        # plt.axis('equal')
        # plt.axis([0, 100, 0, 100])
        # Plot captor agents
        captor_scale = 3.5  # 可视化自适应参数  -1.5
        for k, captor in enumerate(self.scene.pursuers):
            width = captor_scale * captor_aspect_ratio
            (x, y) = captor.position
            # 【取消注释】
            self.pursuers_x[k].append(x)
            self.pursuers_y[k].append(y)
            # circle = plt.Circle((x,y), 4, fill=False, linestyle='dashed', edgecolor='blue')
            # ax.add_artist(circle)
            captor_angle = np.degrees(captor.heading_angle)
            height = captor_scale
            trans_data = Affine2D().rotate_deg_around(x, y, captor_angle) + ax.transData
            ax.imshow(captor_img, extent=[x - width / 2, x + width / 2, y - height / 2, y + height / 2],
                      transform=trans_data, aspect='auto')
            # 【取消注释】并美化轨迹线条
            plt.plot(self.pursuers_x[k], self.pursuers_y[k], alpha=0.5, linewidth=1.5, color='#c6225f')
            # 【新增】为每个追捕者绘制其捕获范围圈
            capture_circle = patches.Circle(
                (x, y), self.scene.collision_distance,
                edgecolor='#D62728', facecolor='none', alpha=0.4,
                linewidth=1.0, linestyle='--'
            )
            ax.add_patch(capture_circle)

            # 【请用下面的代码块替换原有的逃逸者绘制部分】
            # 绘制逃逸者、轨迹、扇区和范围圈
            runner_scale = 3.5
            for k, runner in enumerate(self.scene.evaders):
                width = runner_scale * runner_aspect_ratio
                (x, y) = runner.position
                self.evaders_x[k].append(x)
                self.evaders_y[k].append(y)
                # 1. 先绘制背景元素
                # a. 绘制扇区的外部范围圈
                #    这个圆的半径就是协同奖励的有效半径 self.scene.rh
                if hasattr(self.scene, 'rh'):
                    sector_boundary_circle = patches.Circle(
                        (x, y), self.scene.rh,
                        edgecolor='gray', facecolor='none', alpha=0.7,
                        linewidth=1.0, linestyle='--'
                    )
                    ax.add_patch(sector_boundary_circle)
                # b. 绘制动态旋转的扇区划分线
                if hasattr(self.scene, 'num_sectors') and self.scene.num_sectors > 0:
                    num_sectors = self.scene.num_sectors
                    sector_angle_step = 2 * np.pi / num_sectors
                    base_rotation = runner.heading_angle
                    start_angle_offset = -np.pi / num_sectors
                    line_length = self.scene.rh

                    for s in range(num_sectors):
                        angle = base_rotation + start_angle_offset + s * sector_angle_step
                        end_x = x + line_length * np.cos(angle)
                        end_y = y + line_length * np.sin(angle)
                        ax.plot([x, end_x], [y, end_y], color='gray', linestyle=':', linewidth=1.0, alpha=0.8)

                # 2. 再绘制轨迹
                plt.plot(self.evaders_x[k], self.evaders_y[k], alpha=0.7, linewidth=1.5, color='#1F77B4',
                         label=f'Evader {k}' if self.cnt < 2 else "")

                # 3. 最后绘制智能体自身，确保它在最顶层
                runner_angle_degrees = np.degrees(runner.heading_angle)
                height = runner_scale
                trans_data = Affine2D().rotate_deg_around(x, y, runner_angle_degrees) + ax.transData
                ax.imshow(runner_img, extent=[x - width / 2, x + width / 2, y - height / 2, y + height / 2],
                          transform=trans_data, aspect='auto')

        # Plot obstacles
        for obs in self.scene.obstacles:
            obstacle = plt.Circle(obs.position, obs.radius, color='black')
            ax.add_patch(obstacle)
        # 在这里添加绘制功能区域的代码
        # ----------------------------------------------------------- #
        # 1. 绘制逃逸者目标区域 (美化已有功能)
        if hasattr(self.scene, 'target_area') and hasattr(self.scene, 'target_radius'):
            area_coords = self.scene.target_area
            target_center: Tuple[float, float] = (float(area_coords[0]), float(area_coords[1]))
            target_radius = self.scene.target_radius
            # 在这里创建圆形，移除了 label 属性
            target_zone = patches.Circle(
                target_center, target_radius,
                edgecolor='#33A7FF', facecolor='#8ECFC9', alpha=0.3,
                linewidth=1, linestyle='--'
                # label='目标区域'  <-- 已移除，因为 ax.text 更直接
            )
            ax.add_patch(target_zone)

            # 为文本标签增加一个小的垂直间距
            vertical_offset = 2.0
            ax.text(target_center[0], target_center[1] + target_radius + vertical_offset, '目标区域',
                    ha='center', va='bottom',  # va='bottom' 让文本基线在指定Y坐标之上
                    color='#004D99', fontsize=12, fontweight='bold')
        # 显示标签
        x1 = [120]
        y1 = [120]
        x2 = [130]
        y2 = [130]

        # 绘制两个散点系列，并为它们指定不同的标签
        plt.plot(x1, y1, marker='o', label='<Pursuer>', color='#c6225f')
        plt.plot(x2, y2, marker='o', label='<Evader>', color='#000000')
        # ax.text(90, 95, "-- pursuer", color="blue", fontsize=10, ha='center', va='center')
        # ax.text(90, 90, "-- evader", color="red", fontsize=10, ha='center', va='center')
        self.ax = ax
        self.fig = fig
        # Show plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        # 添加图例
        plt.legend(fontsize=9, shadow=True)
        # if self.cnt == 75:
        #     plt.savefig("./results/MADDPG/8/figure", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.pause(0.01)
        # 清除整个图像上的内容
        # plt.clf()
        # 清除当前轴上的内容
        plt.cla()

    def close(self):
        # 关闭图像窗口
        plt.close()

        # return None
