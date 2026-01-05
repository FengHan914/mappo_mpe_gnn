import numpy as np

from apf import vapf
from scenes.world import Agent, Obstacle


# 追捕者相关奖励设置,取消围捕奖励
# 计算智能体与同类型相邻智能体的角度差值
def diff_angle(agentsPosition, targetPosition, rc):
    n_a = agentsPosition.shape[0]
    diff_angles = 10 * np.ones((n_a, 2))

    distances = np.linalg.norm(agentsPosition - targetPosition, axis=1)

    m = np.count_nonzero(distances < rc)

    if m != 0:
        indices = np.where(distances < rc)[0]
        angles = np.arctan2(agentsPosition[indices, 1] - targetPosition[1],
                            agentsPosition[indices, 0] - targetPosition[0])
        angles_indices = np.column_stack((angles, indices))

        sorted_indices = np.argsort(angles_indices[:, 0])
        sorted_angles_indices = angles_indices[sorted_indices]
        theta1 = np.diff(sorted_angles_indices[:, 0], prepend=-2 * np.pi + sorted_angles_indices[-1, 0])
        theta2 = np.diff(sorted_angles_indices[:, 0], append=2 * np.pi + sorted_angles_indices[0, 0])

        sorted_angles_indices = np.column_stack((sorted_angles_indices, theta1, theta2))
        sorted_indices = np.argsort(sorted_angles_indices[:, 1])
        sorted_angles_indices = sorted_angles_indices[sorted_indices]

        # 角度差值 theta1 theta2
        diff_angles[indices, :] = sorted_angles_indices[:, 2:4]

    return diff_angles


class Entrap_Scene:

    ## 初始化
    def __init__(self, pursuer_num, evader_num):

        # 追击者数量
        self.pursuer_num = pursuer_num

        # 创建环境中追击者
        self.pursuers = [Agent() for i in range(self.pursuer_num)]

        # 逃跑者数量  -- 1
        self.evader_num = evader_num

        # 创建环境中逃跑者
        self.evaders = [Agent() for i in range(self.evader_num)]

        # # 智能体总数量
        agnet_num = self.pursuer_num + self.evader_num

        # 障碍物数量
        self.obstacle_num = 0
        # ----------------------------------------------------------
        self.collision_distance = 5.0  # 捕获判定距离阈值，比如 2.0
        self.episode_limit = 500  # 最大步数限制--应该和主函数保持一致
        self.current_step = 0  # 当前步数计数器

        self.target_area = [110, 50]  # 假设目标区域为(80, 80)
        self.target_radius = 9.0  # 目标区域的半径 --10

        # ----------------------------------------------------------
        # 创建障碍物
        self.obstacles = [Obstacle() for i in range(self.obstacle_num)]

        # 归一化
        self.dist_max = 200
        self.v_max = 5
        self.w_max = 4 * np.pi
        self.dg = 1.55  # 期望距离
        self.do = 3  # 避碰距离  --8  [稍微减小避碰距离,减小25%]
        # self.do = 10  # --原始
        self.dr = 20  # 群体跟踪距离
        self.rl = 1  # 编队最小距离
        self.rh = 12  # 编队最大距离
        self.T = 0.2

        # 设置智能体类型 0-强化学习 1-人工势场 2-键盘控制
        # 追击者
        for i, agent in enumerate(self.pursuers):
            agent.name = "pursuer"
            agent.type = 0
            agent.act_dim = 2
        # 逃跑者
        for i, agent in enumerate(self.evaders):
            agent.name = "evader"
            agent.type = 1
            agent.act_dim = 2

        # 群体最优智能体（距离目标最近）
        self.best_per_idx = None

        # 智能体与相邻智能体之间的角度差值
        self.pursuers_diff_angles = np.zeros((self.pursuer_num, 2))

        # 记录围捕成功等参数
        self.suceess = np.zeros(self.evader_num)
        self.time = np.zeros(self.evader_num)
        self.cnt = 0  # 记录迭代步数

        # 碰撞参数记录
        self.collision = np.ones(self.evader_num)

        # 【新增代码】
        self.capture_position = None  # 用于记录捕获发生时的位置
        self.pursuer_collision_occurred = False  # 用于记录追捕者之间是否发生碰撞

        # 课程学习参数
        self.vel = 0.0

        self.reset_world(mode="train", vel=self.vel)

        # 【新增】定义扇区数量为一个类属性
        self.num_sectors = 4

        # 追击者
        for i, agent in enumerate(self.pursuers):
            if agent.type == 0:
                agent.obs_dim = len(self.observation(agent))
        # 逃跑者
        for i, agent in enumerate(self.evaders):
            if agent.type == 0:
                agent.obs_dim = len(self.observation(agent))

    def reset_world(self, mode, vel=None, scenario_id=0):
        """
        根据场景ID重置世界。
        scenario_id 0: 原始训练场景 (默认)
        scenario_id 1: 追捕者中心包围场景
        scenario_id 2: 全局随机散布场景
        """
        # 首先重置所有通用状态
        self.suceess = np.zeros(self.evader_num)
        self.time = np.zeros(self.evader_num)
        self.cnt = 0
        self.collision = np.ones(self.evader_num)
        self.capture_position = None
        self.pursuer_collision_occurred = False
        self.vel = vel
        self.current_step = 0

        # --- 根据 scenario_id 设置不同的初始位置 ---
        if scenario_id == 1:
            # 场景1：追捕者在中心附近，逃逸者在外围随机位置
            print("[测试场景 1]: 中心集结")
            map_bounds = (0, 120)  # 地图边界
            # 逃逸者随机出现在左上角圆形区域
            for agent in self.evaders:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(1, 30)
                agent.position = np.array([30 + radius * np.cos(angle), 90 + radius * np.sin(angle)])
                # radius = np.random.uniform(1, 45)
                # agent.position = np.array([15 + radius * np.cos(angle), 105 + radius * np.sin(angle)])
                agent.speed = np.zeros(2)
                agent.heading_angle = np.random.uniform(-np.pi, np.pi)

            # 追捕者在中心区域集结
            for agent in self.pursuers:
                # agent.position = np.random.uniform(map_bounds[1] / 3, 3 * map_bounds[1] / 4, size=2)
                agent.position = np.array([np.random.uniform(52, 90), np.random.uniform(20, 68)])  # 测试实验并保存数据绘图
                agent.speed = np.zeros(2)
                agent.heading_angle = np.random.uniform(-np.pi, np.pi)

        elif scenario_id == 2: # 比较困难,随机性更大
            # 场景2：追捕者在距离目标范围比较近的区间随机生成,逃逸者在在距离目标范围比较近的区间随机生成
            # print("[测试场景 2]: 环形分层随机")

            # --- 1. 定义场景参数 ---
            target_center = np.array(self.target_area)
            map_bounds = (0, 120)

            # 定义追捕者的生成环带 (距离目标点较近)
            # 您可以调整这些半径来改变难度
            PURSERS_MIN_RADIUS = 40.0
            PURSERS_MAX_RADIUS = 60.0

            # 定义逃逸者的生成环带 (距离目标点较远)
            EVADER_MIN_RADIUS = 80.0
            EVADER_MAX_RADIUS = 120.0

            # --- 在“近环带”内生成追捕者 ---
            for agent in self.pursuers:
                # 使用循环确保生成的位置在地图边界内
                while True:
                    # 在环带内通过极坐标随机生成一个点
                    radius = np.random.uniform(PURSERS_MIN_RADIUS, PURSERS_MAX_RADIUS)
                    angle = np.random.uniform(0, 2 * np.pi)

                    pos_x = target_center[0] + radius * np.cos(angle)
                    pos_y = target_center[1] + radius * np.sin(angle)

                    # 检查坐标是否越界
                    if (map_bounds[0] <= pos_x <= map_bounds[1]) and \
                            (map_bounds[0] <= pos_y <= map_bounds[1]):
                        agent.position = np.array([pos_x, pos_y])
                        agent.speed = np.zeros(2)
                        agent.heading_angle = np.random.uniform(-np.pi, np.pi)
                        break  # 生成有效位置，退出循环

            # --- 在“远环带”内生成逃逸者 ---
            for agent in self.evaders:
                while True:
                    radius = np.random.uniform(EVADER_MIN_RADIUS, EVADER_MAX_RADIUS)
                    angle = np.random.uniform(0, 2 * np.pi)

                    pos_x = target_center[0] + radius * np.cos(angle)
                    pos_y = target_center[1] + radius * np.sin(angle)

                    if (map_bounds[0] <= pos_x <= map_bounds[1]) and \
                            (map_bounds[0] <= pos_y <= map_bounds[1]):
                        agent.position = np.array([pos_x, pos_y])
                        agent.speed = np.zeros(2)
                        agent.heading_angle = np.random.uniform(-np.pi, np.pi)
                        break

        elif scenario_id == 3: # 非常困难
            # 场景2：所有智能体在地图上完全随机分布
            print("[测试场景 2]: 全局随机")
            map_bounds = (0, 120)  # 地图边界区间
            for agent in self.pursuers + self.evaders:
                agent.position = np.random.uniform(map_bounds[0], map_bounds[1], size=2)
                agent.speed = np.zeros(2)
                agent.heading_angle = np.random.uniform(-np.pi, np.pi)

        else:  # scenario_id == 0 或 其他默认情况
            # 场景0：保持您原始的训练初始设定不变
            # if mode == "eval":  # 只在评估模式下打印，避免训练时刷屏
            #     print("[测试场景 0]: 原始训练布局")
            # print("[测试场景 0]: 阵列初始")
            # 初始化追击者位置
            x = np.random.uniform(10, 20)
            y = np.random.uniform(1, 15)
            for i, agent in enumerate(self.pursuers):
                agent.position = np.array([x + i * 30, y + np.random.uniform(0, 8)])
                agent.speed = np.random.uniform(0, 1) * np.ones(2)
                agent.heading_angle = np.random.uniform(-np.pi, +np.pi)
                agent.epsilon = 0
                agent.vd = np.zeros(2)

            # 初始化逃跑者位置
            for i, agent in enumerate(self.evaders):
                agent.position = np.array([np.random.uniform(10, 20), np.random.uniform(100, 110)])
                agent.speed = np.random.uniform(0, 1) * np.ones(2)
                agent.heading_angle = 0

    ## 速度限幅(仅对单个速度限幅 v = [1,6])
    def v_limited(self, Vraw, Vlim):
        Vlimit = np.zeros_like(Vraw)
        r = np.sqrt(Vraw[0] ** 2 + Vraw[1] ** 2)
        if r <= Vlim:
            Vlimit[0] = Vraw[0]
            Vlimit[1] = Vraw[1]
        else:
            Vlimit[0] = Vlim * Vraw[0] / r
            Vlimit[1] = Vlim * Vraw[1] / r
        return Vlimit

    def update_position_angle(self, agent, T):
        # 限制速度
        agent.speed = self.v_limited(agent.speed, self.v_max)

        # 计算航向与速度方向的偏差角
        delta = agent.heading_angle - np.arctan2(agent.speed[1], agent.speed[0])

        # 修正 delta 到 [-π, π] 范围
        delta = (delta + np.pi) % (2 * np.pi) - np.pi

        # 限制速度与航向角的偏差
        max_epsilon = 1 / 3 * np.pi  # 动态调整限制
        epsilon = np.clip(delta, -max_epsilon, max_epsilon)
        agent.epsilon = epsilon

        # 更新位置
        vd = np.sqrt(agent.speed[0] ** 2 + agent.speed[1] ** 2) * np.cos(epsilon)
        agent.position[0] += vd * np.cos(agent.heading_angle) * T
        agent.position[1] += vd * np.sin(agent.heading_angle) * T

        # 限制角速度更新
        max_angle_rate = 1 / 3 * np.pi  # 最大角速度
        w = np.clip(-delta / T, -max_angle_rate, max_angle_rate)  # 动态限制

        # 更新航向角
        agent.heading_angle += w * T

        # 保证航向角在 [-π, π] 范围内
        agent.heading_angle = (agent.heading_angle + np.pi) % (2 * np.pi) - np.pi

    ## 环境更新
    def world_update(self, actions):
        # --- 1. 更新所有智能体的位置和状态 ---
        T = self.T

        # 追击者更新
        for i, agent in enumerate(self.pursuers):
            # 获取更新前无人艇与目标连线的向量
            self.update_position_angle(agent, T)

            d_cur = np.sqrt(np.sum(np.square(agent.position - self.evaders[agent.target_idx].position)))
            if d_cur <= self.rh:
                vd = vapf.Vd(agent.position.reshape(1, -1), self.evaders[agent.target_idx].position)
                agent.vd = vd[0]

            # 强化学习
            if agent.type == 0:
                # # 更新速度
                agent.speed[0] += actions[i, 0] * T
                agent.speed[1] += actions[i, 1] * T
                '''done = self.get_done()  # 设置终止条件 --done变量
                if done:
                    return done  # 返回终止标志'''
            # 人工势场
            elif agent.type == 1:
                pass
            # 键盘控制
            elif agent.type == 2:
                pass

        # 逃跑者更新
        for i, agent in enumerate(self.evaders):
            if agent.type == 0:
                pass

            # 人工势场
            elif agent.type == 1:
                # ================================================================
                # =============初始版本的新版人工势场法===========================
                # ================================================================
                # # 1. 计算逃逸者与最近追捕者的方向向量
                # nearest_pursuer = min(self.pursuers, key=lambda p: np.linalg.norm(agent.position - p.position))
                # direction_away = agent.position - nearest_pursuer.position  # 从追捕者指向逃逸者的方向
                # distance_to_pursuer = np.linalg.norm(direction_away)
                #
                # # 2. 计算逃逸者朝目标区域的方向向量
                # target_position = np.array([110, 30])  # 假设目标区域在(80, 80)
                # direction_target = target_position - agent.position  # 从逃逸者指向目标区域的方向
                #
                # # 3. 综合方向向量
                # # 避让权重和目标方向权重
                # weight_away = 0.25  # 避让权重，离追捕者越近，避让权重越高
                # weight_target = 1 - weight_away  # 朝目标区域的权重
                # if distance_to_pursuer < 1e-5:  # 防止归一化错误
                #     direction_away = np.zeros(2)
                # else:
                #     direction_away = direction_away / distance_to_pursuer  # 单位化方向向量
                #
                # if np.linalg.norm(direction_target) < 1e-5:
                #     direction_target = np.zeros(2)
                # else:
                #     direction_target = direction_target / np.linalg.norm(direction_target)  # 单位化方向向量
                #
                # # 综合方向
                # combined_direction = weight_away * direction_away + weight_target * direction_target
                #
                # # 4. 计算速度，并限制最大速度
                # agent.speed = combined_direction / np.linalg.norm(combined_direction) * self.v_max * 0.8
                #
                # # 5. 更新逃逸者的位置和角度
                # self.update_position_angle(agent, self.T)

                # # ================================================================
                # # =============仅添加脱困的APF===========================
                # # ================================================================
                # 1. 计算逃逸者与最近追捕者的方向向量 (原始逻辑)
                # nearest_pursuer = min(self.pursuers, key=lambda p: np.linalg.norm(agent.position - p.position))
                # direction_away = agent.position - nearest_pursuer.position
                # distance_to_pursuer = np.linalg.norm(direction_away)
                #
                # # 2. 计算逃逸者朝目标区域的方向向量 (原始逻辑)
                # target_position = np.array(self.target_area)
                # direction_target = target_position - agent.position
                #
                # # 3. 综合方向向量 (使用固定权重) (原始逻辑)
                # weight_away = 0.25
                # weight_target = 1 - weight_away
                #
                # if distance_to_pursuer > 1e-6:
                #     direction_away = direction_away / distance_to_pursuer
                #
                # if np.linalg.norm(direction_target) > 1e-6:
                #     direction_target = direction_target / np.linalg.norm(direction_target)
                #
                # combined_direction = weight_away * direction_away + weight_target * direction_target
                #
                # # 4. 【新增优化】引入随机性以跳出局部最小
                # STUCK_THRESHOLD = 0.1  # 如果合力方向向量的模长小于此值，视为陷入僵局
                # if np.linalg.norm(combined_direction) < STUCK_THRESHOLD:
                #     # 添加一个小的随机扰动来“推”一下智能体
                #     random_perturbation = np.random.uniform(-1, 1, size=2)
                #     combined_direction += 0.5 * random_perturbation  # 0.5是扰动强度，可以调整
                #
                # # 5. 设置最终速度 (原始逻辑)
                # if np.linalg.norm(combined_direction) > 1e-6:
                #     agent.speed = (combined_direction / np.linalg.norm(combined_direction)) * self.v_max * 0.8
                # else:
                #     agent.speed = np.zeros(2)
                #
                # # 6. 更新逃逸者的位置和角度 (原始逻辑)
                # self.update_position_angle(agent, self.T)

                # # ================================================================
                # # =============经过优化的新版人工势场法(很强)===========================
                # # ================================================================
                # # --- 初始化参数 ---
                evader_pos = agent.position
                total_repulsive_force = np.zeros(2)

                # --- 1 & 3. 计算所有威胁的总斥力 (来自追捕者和障碍物) ---

                # a. 来自所有追捕者的斥力
                # PURSUER_INFLUENCE_RADIUS: 逃逸者能“感知”到追捕者的最大距离
                PURSUER_INFLUENCE_RADIUS = self.rh * 1.5  # 例如设为协同半径的1.5倍  -self.rh * 1.5
                min_dist_to_pursuer = float('inf')

                for p_agent in self.pursuers:
                    dist_vec = evader_pos - p_agent.position
                    dist = np.linalg.norm(dist_vec)
                    min_dist_to_pursuer = min(min_dist_to_pursuer, dist)  # 更新最近距离

                    if dist < PURSUER_INFLUENCE_RADIUS and dist > 1e-6:
                        # 斥力大小与距离成反比，方向为远离追捕者
                        repulsive_force = (1.0 / dist) * (dist_vec / dist)
                        total_repulsive_force += repulsive_force

                # b. 来自所有障碍物的斥力 (如果存在)
                OBSTACLE_INFLUENCE_RADIUS = 15.0  # 感知障碍物的最大距离 -15.0
                for obs in self.obstacles:
                    dist_vec = evader_pos - obs.position
                    dist = np.linalg.norm(dist_vec)
                    if dist < OBSTACLE_INFLUENCE_RADIUS and dist > 1e-6:
                        repulsive_force = (1.0 / dist) * (dist_vec / dist)
                        total_repulsive_force += repulsive_force

                # --- 2. 根据最近威胁动态计算权重 ---
                DANGER_ZONE_RADIUS = self.rh  # 当追捕者进入此范围，逃命优先 - self.rh*0.8

                if min_dist_to_pursuer >= DANGER_ZONE_RADIUS:
                    # 在安全距离外，主要想着去目标点
                    weight_away = 0.2
                    weight_target = 0.8
                else:
                    # 在危险区域内，距离越近，逃命权重越高
                    # 当距离为0时，权重为1.0；当距离为DANGER_ZONE_RADIUS时，权重为0.2
                    weight_away = 1.0 - 0.8 * (min_dist_to_pursuer / DANGER_ZONE_RADIUS)
                    weight_target = 1.0 - weight_away

                # --- 计算引力 (朝向目标点) ---
                target_position = np.array(self.target_area)
                attractive_force = target_position - evader_pos
                if np.linalg.norm(attractive_force) > 1e-6:
                    attractive_force /= np.linalg.norm(attractive_force)

                # --- 组合最终方向 ---
                # 注意：我们只对斥力方向的合力进行归一化（如果它不为零）
                if np.linalg.norm(total_repulsive_force) > 1e-6:
                    total_repulsive_force /= np.linalg.norm(total_repulsive_force)

                combined_direction = (weight_away * total_repulsive_force +
                                      weight_target * attractive_force)

                # --- 4. 引入随机性以跳出局部最小 ---
                STUCK_THRESHOLD = 0.1  # 如果合力方向向量的模长小于此值，视为陷入僵局
                if np.linalg.norm(combined_direction) < STUCK_THRESHOLD:
                    # 添加一个小的随机扰动
                    random_perturbation = np.random.uniform(-1, 1, size=2)
                    combined_direction += 0.5 * random_perturbation  # 0.5是扰动强度

                # --- 设置最终速度 ---
                if np.linalg.norm(combined_direction) > 1e-6:
                    agent.speed = (combined_direction / np.linalg.norm(combined_direction)) * self.v_max * 0.8

                # 更新逃逸者的位置和角度
                self.update_position_angle(agent, self.T)

            # 键盘控制
            elif agent.type == 2:
                pass
        done = self.check_capture_condition()  # 检查捕获条件
        if done:
            # print("捕获成功，终止当前episode")
            return True  # 返回终止标志

        # --- 2. 在一个时间步的最后，统一检查所有终止条件 ---

        # a. 检查碰撞 (最高优先级)
        if not self.pursuer_collision_occurred:
            for i in range(self.pursuer_num):
                for j in range(i + 1, self.pursuer_num):
                    p1 = self.pursuers[i]
                    p2 = self.pursuers[j]
                    dist = np.linalg.norm(p1.position - p2.position)
                    if dist < self.do:
                        self.pursuer_collision_occurred = True
                        break
                if self.pursuer_collision_occurred:
                    break

        # b. 检查其他条件
        is_captured = self.check_capture_condition()
        is_escaped = self.check_evader_escape_condition()
        is_timeout = self.current_step >= self.episode_limit

        # --- 3. 只要满足任意一个终止条件，就返回True ---
        #    这个 done 标志现在综合了所有情况
        done = is_captured or is_escaped or is_timeout or self.pursuer_collision_occurred
        # 步数加1
        self.cnt += 1

        return done

    # 追捕者奖励
    def pur_reward(self, agent):
        # 趋近奖励
        k1 = 10  # 降低系数以加重奖励
        d_g = np.sqrt(np.sum(np.square(agent.position - self.evaders[0].position)))  # 追击者到逃跑者的距离
        dg = self.dg  # 理想的目标距离
        rew1 = 1 / (1 + k1 / self.dist_max * np.sqrt((d_g - dg) ** 2))  # 趋近奖励

        # 避碰奖励（降低权重）
        k2 = 2  # 减小系数
        do = self.do  # 避碰距离
        rew2 = 0
        for i, other_agent in enumerate(self.pursuers):
            if other_agent is agent:
                continue
            dist_p = np.sqrt(np.sum(np.square(agent.position - other_agent.position)))
            if dist_p <= do:
                rew2 += np.exp(-k2 / do * (do - dist_p)) - 1

        # 捕获成功的正奖励（目标区域在逃跑者附近）
        capture_reward = 0
        capture_distance = 5.0  # 捕获的阈值距离
        # -------------------------------------
        if d_g <= capture_distance:  # 如果追击者靠近逃跑者
            capture_reward = 20.0  # 捕获成功的奖励

        # 逃跑者抵达目标区域的负奖励
        evader_escape_penalty = 0
        target_area = np.array(self.target_area)  # 目标区域中心
        target_radius = self.target_radius  # 目标区域半径
        for evader in self.evaders:
            distance_to_target = np.linalg.norm(evader.position - target_area)
            if distance_to_target <= target_radius:  # 逃跑者到达目标区域
                evader_escape_penalty = -10.0  # 追击失败的负奖励
                break  # 一个逃跑者成功即可结束

        # 协同围捕奖励
        rew_coop = 0
        effective_pursuit_radius = self.rh  # 假设使用编队最大距离 rh 作为有效追捕半径，当前值为12
        min_pursuers_for_coop = 2  # 至少需要多少个追捕者形成协同
        num_sectors = 4  # 将逃逸者周围划分为4个扇区 (前、后、左、右)
        pursuers_in_sectors = [[] for _ in range(num_sectors)]
        active_pursuers_for_coop = []  # 记录参与协同的追捕者

        evader_pos = self.evaders[0].position  # 假设只有一个逃逸者

        for p_idx, p_agent in enumerate(self.pursuers):
            dist_to_evader = np.linalg.norm(p_agent.position - evader_pos)
            if dist_to_evader < effective_pursuit_radius:
                angle_to_evader = np.arctan2(p_agent.position[1] - evader_pos[1],
                                             p_agent.position[0] - evader_pos[0])
                # 将角度映射到扇区索引 (0 to num_sectors-1)
                # 例如，简单地根据角度范围划分
                sector_idx = int(((angle_to_evader + np.pi) / (2 * np.pi)) * num_sectors) % num_sectors
                pursuers_in_sectors[sector_idx].append(p_idx)
                if p_agent is agent:  # 如果当前计算奖励的智能体是活跃的
                    active_pursuers_for_coop.append(True)
                else:
                    active_pursuers_for_coop.append(False)  # 只是为了占位

        occupied_sectors = 0
        for sector_pursuers in pursuers_in_sectors:
            if len(sector_pursuers) > 0:
                occupied_sectors += 1

        # 如果当前智能体参与了协同，并且形成了有效的扇区占领
        is_current_agent_active = False
        dist_to_evader_current = np.linalg.norm(agent.position - evader_pos)
        if dist_to_evader_current < effective_pursuit_radius:
            is_current_agent_active = True

        if is_current_agent_active and occupied_sectors >= min_pursuers_for_coop:  # 可以根据 occupied_sectors 的多少给予不同大小的奖励
            # 一个简单的奖励可以是占领的扇区数越多奖励越高
            rew_coop = occupied_sectors * (0.5)  # 例如每个额外占领的扇区奖励0.5，需要调参


        # # 【核心修改】协同围捕奖励 (基于逃逸者自身坐标系)
        # rew_coop = 0
        # effective_pursuit_radius = self.rh
        # min_pursuers_for_coop = 2
        #
        # # 使用 self.num_sectors (在__init__中定义的)
        # pursuers_in_sectors = [[] for _ in range(self.num_sectors)]
        # evader_pos = self.evaders[0].position
        #
        # # 【新增】获取逃逸者的朝向角
        # evader_heading = self.evaders[0].heading_angle
        #
        # for p_idx, p_agent in enumerate(self.pursuers):
        #     dist_to_evader = np.linalg.norm(p_agent.position - evader_pos)
        #     if dist_to_evader < effective_pursuit_radius:
        #         # 1. 计算追捕者相对于逃逸者的全局角度
        #         global_angle = np.arctan2(p_agent.position[1] - evader_pos[1],
        #                                     p_agent.position[0] - evader_pos[0])
        #
        #         # 2. 计算相对于逃逸者朝向的相对角度
        #         relative_angle = global_angle - evader_heading
        #
        #         # 3. 将相对角度归一化到 [-pi, pi] 范围
        #         relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        #
        #         # 4. 根据相对角度计算扇区索引
        #         sector_idx = int(((relative_angle + np.pi) / (2 * np.pi)) * self.num_sectors) % self.num_sectors
        #         pursuers_in_sectors[sector_idx].append(p_idx)
        #
        # occupied_sectors = sum(1 for sector in pursuers_in_sectors if sector)
        #
        # is_current_agent_active = d_g < effective_pursuit_radius
        # if is_current_agent_active and occupied_sectors >= min_pursuers_for_coop:
        #     rew_coop = occupied_sectors * 0.5

        # 综合奖励
        # rew = 1.0 * (
        #         0.7 * rew1  # 趋近奖励 0.85-->0.70
        #         + 0.1 * rew2  # 避碰奖励 0.05-->0.10
        #         + 0.1 * rew_coop
        #         + 0.05 * capture_reward  # 捕获成功奖励
        #         + 0.05 * evader_escape_penalty  # 逃跑者逃跑的负奖励
        # )

        # 【建议修改后】增大“避碰奖励”的权重:0.1-->0.2
        rew = 1.0 * (
                0.6 * rew1  # 可以适当降低趋近奖励的权重
                + 0.2 * rew2  # 【重点】将避碰权重从0.1提升到0.2
                + 0.1 * rew_coop  # 保持协同权重不变
                + 0.05 * capture_reward
                + 0.05 * evader_escape_penalty
        )

        return rew

    # 逃跑者奖励
    def eva_reward(self, agent):
        rew = 0
        return rew

    def reward(self, agent):
        main_reward = self.pur_reward(agent) if agent.name == "pursuer" else self.eva_reward(agent)
        return main_reward

    # 获取观测值
    def observation(self, agent):

        # 初始化观测值
        obs_s = []  # 智能体自身状态信息（相对逃跑者）
        obs_c = []  # 智能体相互之间状态信息（其它追击者智能体、障碍物）
        obs_p = []  # 集群态势信息

        # 获取距离逃跑者最近追捕者的索引
        min_dist = 200
        for i, ps in enumerate(self.pursuers):
            # dist = np.sqrt(np.sum(np.square(ps.position - self.evaders[0].position)))
            if ps.target_idx == agent.target_idx:
                dist = np.sqrt(np.sum(np.square(ps.position - self.evaders[agent.target_idx].position)))
                if dist < min_dist:
                    min_dist = dist
                    # self.best_per_idx = i
                    agent.best_idx = i

        # 追击者观测信息获取  --移除 obs_p 中与集群态势相关的信息，避免因队形奖励而产生围捕策略

        if agent.name == "pursuer":
            # dist_pe = np.sqrt(np.sum(np.square(agent.position - self.evaders[0].position)))
            dist_pe = np.sqrt(np.sum(np.square(agent.position - self.evaders[agent.target_idx].position)))
            # 自身信息
            obs_s.append((agent.position - self.evaders[agent.target_idx].position) / self.dist_max)  # 追击目标只有一个 1,2
            obs_s.append((agent.speed - self.evaders[agent.target_idx].speed) / self.v_max)  # 3,4
            obs_s.append((agent.heading_angle - self.evaders[agent.target_idx].heading_angle) / self.w_max)  # 5
            epsilon = agent.epsilon
            dg = self.dg  # 期望围捕距离
            obs_s.append((dist_pe - dg) / self.dist_max)  # 6
            # obs_s.append(dg / self.dist_max)
            obs_s.append(epsilon / self.w_max)  # 7

            # 集群态势信息
            rl = self.rl  # 内测半径 第一次的模型需注释
            rh = self.rh  # 外侧半径
            # obs_p.append(rl/self.dist_max)
            # obs_p.append(rh/self.dist_max)
            obs_p.append((dist_pe - rh) / self.dist_max)  # 11
            obs_p.append((dist_pe - rl) / self.dist_max)  # 12

            # 相对信息
            for i, other_agent in enumerate(self.pursuers):
                cur_idx = i  # 当前智能体索引
                if other_agent is agent or other_agent.target_idx != agent.target_idx:
                    continue

                dist_pp = np.sqrt(np.sum(np.square(agent.position - other_agent.position)))
                obs_c.append((dist_pp - self.do) / self.dist_max)
                obs_c.append((agent.position - other_agent.position) / self.dist_max)
                obs_c.append((agent.speed - other_agent.speed) / self.v_max)
                obs_c.append((agent.heading_angle - other_agent.heading_angle) / self.w_max)
        # 逃跑者观测信息获取
        if agent.name == "evader":
            pass
        out = np.hstack(obs_s + obs_p + obs_c)
        # print("New observation dimension:", len(out))  # out 是返回的观测向量

        return out

    # 判断围捕是否结束
    def get_done(self):
        """
        判断追捕是否结束
        Returns:
            done (bool): 是否满足终止条件
        """
        for pursuer in self.pursuers:
            for evader in self.evaders:
                # 计算追击者与逃逸者的距离
                distance = np.linalg.norm(pursuer.position - evader.position)

                # 如果距离小于碰撞距离，任务终止
                if distance <= self.collision_distance:
                    return True
        return False

    # 检查追捕者是否捕获了逃逸者
    # In entrap.py, class Entrap_Scene

    # 【修正】确保此函数在捕获时能记录位置
    def check_capture_condition(self):
        """
        检查追捕者是否捕获了逃逸者, 并在捕获时记录位置
        """
        for pursuer in self.pursuers:
            for evader in self.evaders:
                distance = np.linalg.norm(pursuer.position - evader.position)
                if distance <= self.collision_distance:  # 捕获成功
                    self.capture_position = evader.position  # 记录捕获时逃逸者的位置
                    return True
        return False

    # 检查逃跑者是否到达目标区域
    def check_evader_escape_condition(self):
        """
        检查逃跑者是否到达目标区域
        """
        for evader in self.evaders:
            distance_to_target = np.linalg.norm(evader.position - np.array(self.target_area))
            if distance_to_target <= self.target_radius:  # 逃跑成功
                return True
        return False
