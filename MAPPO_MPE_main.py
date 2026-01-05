# MAPPO_MPE_main.py (文件开头)
import argparse
import csv  # 确保此行存在
import os
# 必须在导入 torch 或 numpy 之前设置！
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
# matplotlib.use('Agg')  # <--- 设置非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# from make_env import make_env
# from env import mpe
from env.environment import MutiAgentEnv
from mappo_mpe import MAPPO_MPE
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from scenes.entrap import Entrap_Scene


# 曲线平滑函数
def smooth(data, weight=0.97):
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # GPU训练
        device = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')

        # Create env  --- 创建一个训练环境
        self.agent_num = 3
        self.scene = Entrap_Scene(self.agent_num, 1)  # --加载追逃场景--
        self.env = MutiAgentEnv(self.scene, reset_callback=self.scene.reset_world, reward_callback=self.scene.reward,
                                observation_callback=self.scene.observation)
        self.args.N = self.env.agents_num  # The number of agents
        self.env_evaluate = MutiAgentEnv(self.scene, reset_callback=self.scene.reset_world,
                                         reward_callback=self.scene.reward,
                                         observation_callback=self.scene.observation)
        self.args.obs_dim = self.env.obs_n[0]  # 设置观测向量维度  -self.env.obs_n[0]

        self.args.action_dim = self.env.act_n[0]
        self.args.state_dim = np.sum(
            self.env.obs_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        # print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim))
        # print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim))

        # =====================================
        # ============设置实验编号================
        # =====================================
        # 编辑保存的路径-编辑序号
        # -model -result -checkpoint -data_train
        self.train_num = number
        self.algorithm_name = "MAPPO"
        model_dir = os.path.join("./model", self.algorithm_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 当前训练模型需要保存的文件夹  -- file
        model_num = self.train_num  # ---self.number
        cur_model_dir = os.path.join(model_dir, model_num)
        if not os.path.exists(cur_model_dir):
            os.makedirs(cur_model_dir)

        # 创建结果文件夹  -- file
        result_dir = os.path.join("./results", self.algorithm_name, model_num)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 创建断点文件夹 -- file
        checkpoint_dir = os.path.join("./checkpoint", self.algorithm_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 创建训练数据文件夹 -- file
        checkpoint_dir = os.path.join("./data_train", self.algorithm_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 创建训练数据轮次文件夹 -- file
        checkpoint_dir = os.path.join("./data_train", self.algorithm_name, model_num)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 保存所有奖励 -- file
        self.rewards_all = []
        self.training_rewards_steps = []  # 【新增这一行】

        # 颜色---绘制奖励函数曲线
        self.colors = ['#8ECFC9', '#82B0D2', '#BEB8DC', '#E7DAD2']

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args, self.device)
        self.replay_buffer = ReplayBuffer(self.args, self.device)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.evaluation_record_steps = []

        self.capture_rates = []  # <--- NEW: For capture rates
        self.escape_rates = []  # <--- NEW: For escape rates
        self.timeout_rates = []  # <--- NEW: For timeout rates
        self.avg_capture_steps = []  # <--- NEW: For average steps to capture

        self.actor_loss_history = []  # <--- NEW: For actor loss
        self.critic_loss_history = []  # <--- NEW: For critic loss
        self.loss_record_steps = []

        self.total_steps = 0
        # 定义一个变量，用于存储当前训练应使用的场景ID
        self.current_training_scenario = 0

        # 初始化数据记录文件
        print("--- 初始化数据记录文件 ---")
        # 1. 定义文件名，确保每个实验有唯一的文件路径
        output_dir = f'./data_train/{self.algorithm_name}/{self.train_num}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. 创建并写入“评估结果文件”的表头
        self.results_filename = os.path.join(output_dir, 'evaluation_results.csv')
        try:
            with open(self.results_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # 动态创建每个智能体的奖励表头，以便扩展
                reward_headers = [f'agent_{i}_reward' for i in range(self.args.N)]
                # 最终的完整表头
                header = ['total_steps', 'avg_reward', *reward_headers,
                          'capture_rate', 'escape_rate', 'timeout_rate',
                          'avg_capture_steps', 'collision_rate']
                writer.writerow(header)
            print(f"评估结果将保存在: {self.results_filename}")
        except IOError as e:
            print(f"错误: 无法写入文件 {self.results_filename}. 原因: {e}")

        # 3. 创建并写入“损失函数文件”的表头
        self.loss_filename = os.path.join(output_dir, 'training_losses.csv')
        try:
            with open(self.loss_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['total_steps', 'actor_loss', 'critic_loss']
                writer.writerow(header)
            print(f"训练损失将保存在: {self.loss_filename}")
        except IOError as e:
            print(f"错误: 无法写入文件 {self.loss_filename}. 原因: {e}")

        # 4. 【新增】创建并写入“训练回合奖励文件”的表头
        self.training_rewards_filename = os.path.join(output_dir, 'training_episode_rewards.csv')
        try:
            with open(self.training_rewards_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # 创建包含 total_steps 的表头
                header = ['total_steps'] + [f'agent_{i}_reward' for i in range(self.args.N)]
                writer.writerow(header)
            print(f"训练回合奖励将保存在: {self.training_rewards_filename}")
        except IOError as e:
            print(f"错误: 无法写入文件 {self.training_rewards_filename}. 原因: {e}")
        # --------------------------------------------------- #

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)



    def run(self, ):
        # --- 初始化和路径设置---
        old_step = 0
        best_reward = -100
        name = self.algorithm_name
        train_num = self.train_num
        cur_model_dir = os.path.join("./model", name, train_num)
        if not os.path.exists(cur_model_dir):
            os.makedirs(cur_model_dir)
        cur_checkpoint_dir = os.path.join("./checkpoint", name, train_num)
        if not os.path.exists(cur_checkpoint_dir):
            os.makedirs(cur_checkpoint_dir)

        evaluate_num = 1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            # --- 运行一个回合 (此函数将使用我们刚刚确定的场景ID) ---
            rewards, episode_steps = self.run_episode_mpe(evaluate=False)

            self.total_steps += episode_steps

            # --- 后续的评估、保存、训练逻辑 (保持不变)
            # 每evaluate_freq进行一次评价
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            # --- 保持稀疏性，按 evaluate_freq 的频率记录训练奖励 ---
            if self.total_steps // self.args.evaluate_freq > (
                    self.total_steps - episode_steps) // self.args.evaluate_freq:

                # a. 将当前回合的奖励存入内存列表，用于后续Python绘图
                self.rewards_all.append(rewards)
                self.training_rewards_steps.append(self.total_steps)

                # b. 将该数据实时追加到CSV文件
                try:
                    with open(self.training_rewards_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # 将 total_steps 和 list形式的rewards合并成一行写入
                        row_data = [self.total_steps] + rewards.tolist()
                        writer.writerow(row_data)
                except IOError as e:
                    print(f"警告: 无法向文件 {self.training_rewards_filename} 写入回合奖励数据. 原因: {e}")

            # 模型保存
            if self.total_steps % 50000 == 0:
                ## 保存断点
                # path_checkpoint = "./checkpoint/MADDPG/last_model.pth"
                path_checkpoint = os.path.join(cur_checkpoint_dir, "last_model.pth")
                checkpoint = {  # "runner": self,
                    "total_steps": self.total_steps,
                    "agent_n": self.agent_n,
                    "best_reward": best_reward,
                    "rewards_all": self.rewards_all
                }
                torch.save(checkpoint, path_checkpoint)

                # 保存模型
                path_model = os.path.join(cur_model_dir, "model_{}.pth".format(self.total_steps))
                torch.save(self.agent_n, path_model)
                print("!!--Last Model has Update(-{})--!!".format(self.total_steps))

            if self.total_steps % 10000 == 0:
                path_best_model = os.path.join(cur_model_dir, "model_best.pth")
                rs = sum(rewards)
                if rs > best_reward:
                    torch.save(self.agent_n, path_best_model)
                    best_reward = rs
                print("step:{}/{} length:{} rewards:{}".format(self.total_steps, args.max_train_steps,
                                                               self.total_steps - old_step, rewards))

            if self.replay_buffer.episode_num == self.args.batch_size:
                # 1. 训练网络并获取损失
                avg_actor_loss, avg_critic_loss = self.agent_n.train(self.replay_buffer, self.total_steps)

                # 2. 训练完成后，立刻清空经验池
                self.replay_buffer.reset_buffer()

                # 3. 记录损失历史，用于后续绘图
                self.actor_loss_history.append(avg_actor_loss)
                self.critic_loss_history.append(avg_critic_loss)
                self.loss_record_steps.append(self.total_steps)  # 【新增这一行】


                # 4. 记录到 TensorBoard
                self.writer.add_scalar('loss/actor_loss', avg_actor_loss, global_step=self.total_steps)
                self.writer.add_scalar('loss/critic_loss', avg_critic_loss, global_step=self.total_steps)

                # 5. 【修正位置】将loss数据追加到CSV文件
                try:
                    with open(self.loss_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.total_steps, avg_actor_loss, avg_critic_loss])
                except IOError as e:
                    print(f"警告: 无法向文件 {self.loss_filename} 写入损失数据. 原因: {e}")

                # 6. (可选) 定期打印损失
                if self.total_steps % (self.args.evaluate_freq * 5) == 0:
                    print(
                        f"    - Training @ step {self.total_steps}: Actor Loss={avg_actor_loss:.4f}, Critic Loss={avg_critic_loss:.4f}")

            old_step = self.total_steps  # 这行保持在if代码块之外


        # 使用 self.training_rewards_steps 作为X轴数据
        x_axis_training_steps = np.array(self.training_rewards_steps)
        rewards_data = np.array(self.rewards_all)
        # s_rewards_all = smooth(rewards_data, weight=0.95)
        # 确保 s_rewards_all 是 numpy array
        s_rewards_all = np.array(smooth(rewards_data, weight=0.95))

        for a in range(self.args.N):
            plt.figure()
            # 传入X轴数据
            plt.plot(x_axis_training_steps, rewards_data[:, a], linestyle='-', color=self.colors[a],
                     label='Rrewards' + str(a))
            plt.plot(x_axis_training_steps, s_rewards_all[:, a], linestyle='--', color='#FA7F6F',
                     label='smoothed_Rewards' + str(a))

            plt.legend()
            plt.title('MAPPO for MutiAgent_PE')
            plt.xlabel('Total Steps')  # 【修改】X轴标签
            plt.ylabel('Rewards')
            plt.savefig("./results/{}/{}/agent_{}_Rewards_curve.png".format(self.algorithm_name, self.number, a))
        plt.show()

        # --- 绘制 Actor Loss ---
        if self.actor_loss_history:  # 检查列表是否非空
            plt.figure(figsize=(10, 7.5))  # 创建新画布4:3
            # iterations_x_axis = range(len(self.actor_loss_history))  # X轴是训练迭代次数
            x_axis_steps = self.loss_record_steps

            # 绘制平滑 Actor Loss
            smoothed_actor_loss = smooth(self.actor_loss_history)  # 使用 weight=0.97 (或您选择的值)
            if smoothed_actor_loss:  # 确保 smooth 函数返回有效结果
                plt.plot(x_axis_steps, smoothed_actor_loss, label='Smoothed Actor Loss', color='tab:blue')  # 指定颜色

            # 可以选择性绘制原始 Actor Loss
            plt.plot(x_axis_steps, self.actor_loss_history, label='Raw Actor Loss', alpha=0.3, color='tab:blue')

            plt.xlabel('Total Steps')
            plt.ylabel('Actor Loss')
            plt.title(f'{self.algorithm_name} Actor Loss During Training (Train Num: {self.train_num})')
            plt.legend()
            plt.grid(True)
            # 如果 Actor Loss 值非常小或跨度大，可以考虑对数坐标轴
            # plt.yscale('symlog', linthresh=1e-5) # 对称对数轴，处理可能的小于等于0的值
            plt.savefig(f"./results/{self.algorithm_name}/{self.train_num}/actor_loss_curves.png")
            plt.close()  # 关闭图形

        # --- 绘制 Critic Loss ---
        if self.critic_loss_history:  # 检查列表是否非空
            plt.figure(figsize=(10, 7.5))

            # 【修改】使用记录的 total_steps 作为 X 轴
            x_axis_steps = self.loss_record_steps

            # 绘制平滑 Critic Loss
            smoothed_critic_loss = smooth(self.critic_loss_history)
            if smoothed_critic_loss:
                plt.plot(x_axis_steps, smoothed_critic_loss, label='Smoothed Critic Loss',
                         color='tab:orange')

            # 绘制原始 Critic Loss
            plt.plot(x_axis_steps, self.critic_loss_history, label='Raw Critic Loss', alpha=0.3, color='tab:orange')

            #更新 X 轴标签
            plt.xlabel('Total Steps')
            plt.ylabel('Critic Loss')
            plt.title(f'{self.algorithm_name} Critic Loss During Training (Train Num: {self.train_num})')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"./results/{self.algorithm_name}/{self.train_num}/critic_loss_curves.png")
            plt.close()

        # 使用 self.evaluation_record_steps 作为X轴数据
        x_axis_eval_steps = self.evaluation_record_steps

        # 绘制 成功率 和 平均捕获步数 指标曲线
        plt.figure()
        plt.plot(x_axis_eval_steps, self.capture_rates, label='Capture Rate')
        plt.plot(x_axis_eval_steps, self.escape_rates, label='Escape Rate')
        plt.plot(x_axis_eval_steps, self.timeout_rates, label='Timeout Rate')
        plt.xlabel('Total Steps')
        plt.ylabel('Rate')
        plt.title('Task Success Rates over Training')
        plt.legend()
        plt.savefig(f"./results/{self.algorithm_name}/{self.train_num}/task_success_rates.png")

        if self.avg_capture_steps:
            plt.figure()
            plt.plot(x_axis_eval_steps, self.avg_capture_steps, label='Average Capture Steps')
            plt.xlabel('Total Steps')  # 【修改】X轴标签
            plt.ylabel('Average Steps')
            plt.title('Average Steps to Capture over Training')
            plt.legend()
            plt.savefig(f"./results/{self.algorithm_name}/{self.train_num}/avg_capture_steps.png")
            plt.show()

        # --- 训练结束后的收尾工作 ---
        self.evaluate_policy()
        self.env.close()


    def evaluate_policy(self, ):
        # --- 初始化本次评估的统计变量 ---
        sum_of_rewards_per_agent_all_eval_episodes = np.zeros(self.args.N)
        num_eval_episodes = self.args.evaluate_times
        if num_eval_episodes <= 0:
            print("警告: evaluate_times 非正数，跳过评估。")
            return

        captures = 0
        escapes = 0
        timeouts = 0
        collision_episodes = 0  # 记录发生碰撞的回合数
        capture_positions = []  # 记录所有成功捕获的位置
        total_capture_steps_for_successful_captures = 0
        successful_capture_count = 0

        # --- 执行评估循环 ---
        for _ in range(num_eval_episodes):
            obs_n = self.env_evaluate.reset(mode="eval", vel=None)
            current_eval_steps = 0
            episode_done_eval = False
            current_episode_rewards_per_agent = np.zeros(self.args.N)

            while not episode_done_eval and current_eval_steps < self.args.episode_limit:
                a_n, _ = self.agent_n.choose_action(obs_n, evaluate=True)
                obs_next_n, r_n, _ = self.env_evaluate.step(a_n)
                current_episode_rewards_per_agent += r_n
                obs_n = obs_next_n
                current_eval_steps += 1

                # 【修正 #1】在回合终止判断中加入碰撞检查
                episode_done_eval = (self.env_evaluate.scene.check_capture_condition() or
                                     self.env_evaluate.scene.check_evader_escape_condition() or
                                     current_eval_steps >= self.args.episode_limit or
                                     self.env_evaluate.scene.pursuer_collision_occurred)  # 发生碰撞则立刻终止

            # --- 记录单回合评估结果 ---
            sum_of_rewards_per_agent_all_eval_episodes += current_episode_rewards_per_agent

            # 获取最终状态
            final_captured = self.env_evaluate.scene.capture_position is not None
            final_escaped = self.env_evaluate.scene.check_evader_escape_condition()
            collision_happened = self.env_evaluate.scene.pursuer_collision_occurred

            # 建立有优先级的统计逻辑：碰撞失败 > 捕获成功 > 逃逸成功 > 超时
            if collision_happened:
                # 只要发生碰撞，就算作“超时/失败”，并且不能算作成功捕获
                timeouts += 1  # 将碰撞失败归为超时类别，或可另立'collision_failures'
                collision_episodes += 1  # 碰撞率计数器依然累加
            elif final_captured:
                # 仅在没有碰撞的情况下，才算作成功捕获
                captures += 1
                total_capture_steps_for_successful_captures += current_eval_steps
                successful_capture_count += 1
                if self.env_evaluate.scene.capture_position is not None:
                    capture_positions.append(self.env_evaluate.scene.capture_position)
            elif final_escaped:
                escapes += 1
            else:  # 剩下的情况是正常超时
                timeouts += 1

        # --- 计算本次评估的各项平均指标 ---
        # 1. 平均奖励
        avg_reward_per_agent_this_evaluation = sum_of_rewards_per_agent_all_eval_episodes / num_eval_episodes
        overall_avg_agent_reward = np.mean(avg_reward_per_agent_this_evaluation)
        self.evaluate_rewards.append(overall_avg_agent_reward)

        # 2. 任务成功率
        current_capture_rate = captures / num_eval_episodes
        current_escape_rate = escapes / num_eval_episodes
        current_timeout_rate = timeouts / num_eval_episodes
        self.capture_rates.append(current_capture_rate)
        self.escape_rates.append(current_escape_rate)
        self.timeout_rates.append(current_timeout_rate)

        # 3. 平均捕获时间 (步数) 和位置
        current_avg_capture_steps = total_capture_steps_for_successful_captures / successful_capture_count if successful_capture_count > 0 else 0
        self.avg_capture_steps.append(current_avg_capture_steps)

        avg_capture_position_str = "无捕获"
        if capture_positions:
            avg_pos = np.mean(capture_positions, axis=0)
            avg_capture_position_str = f"({avg_pos[0]:.1f}, {avg_pos[1]:.1f})"

        # 4. 场景配置信息
        pursuer_num = self.env.scene.pursuer_num
        evader_num = self.env.scene.evader_num
        has_obstacles = '有' if self.env.scene.obstacle_num > 0 else '无'
        speed_ratio = self.env.scene.v_max / (self.env.scene.v_max * 0.8) if self.env.scene.v_max > 0 else float('inf')

        # 5. 碰撞率
        collision_rate = collision_episodes / num_eval_episodes

        # --- 格式化并打印单行紧凑评估结果 ---
        reward_details_str = ' '.join(
            [f'P{i}:{avg_reward_per_agent_this_evaluation[i]:.2f}' for i in range(self.args.N)])
        capture_time_str = f"{current_avg_capture_steps:.2f}步" if successful_capture_count > 0 else "N/A"
        output_line = (
            f"| 进度: {self.total_steps:<7} | "
            f"平均奖励: {overall_avg_agent_reward:<6.2f} | "
            f"各艇奖励: [{reward_details_str}] | "
            f"结果(捕/逃/超): {current_capture_rate:.1%}/{current_escape_rate:.1%}/{current_timeout_rate:.1%} | "
            f"捕获时间: {capture_time_str:<7} | "
            f"捕获位置: {avg_capture_position_str:<12} | "
            f"追捕者碰撞率: {collision_rate:<6.1%} |"
            f" 场景配置: {pursuer_num}v{evader_num}, 速比:{speed_ratio:.2f}, 障碍物:{has_obstacles} |"
        )
        print(output_line)

        # --- 保存评估数据到文件 ---
        capture_steps_to_save = current_avg_capture_steps if successful_capture_count > 0 else np.nan
        row_data = [
            self.total_steps,
            overall_avg_agent_reward,
            *avg_reward_per_agent_this_evaluation,
            current_capture_rate,
            current_escape_rate,
            current_timeout_rate,
            capture_steps_to_save,
            collision_rate
        ]
        try:
            with open(self.results_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except IOError as e:
            print(f"警告: 无法向文件 {self.results_filename} 写入评估数据. 原因: {e}")

        # --- 记录到 TensorBoard ---
        self.writer.add_scalar('evaluate_rewards/overall_avg_agent_reward', overall_avg_agent_reward,
                               global_step=self.total_steps)
        self.writer.add_scalar('evaluate_metrics/capture_rate', current_capture_rate, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_metrics/escape_rate', current_escape_rate, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_metrics/timeout_rate', current_timeout_rate, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_metrics/collision_rate', collision_rate, global_step=self.total_steps)
        if successful_capture_count > 0:
            self.writer.add_scalar('evaluate_metrics/avg_capture_steps', current_avg_capture_steps,
                                   global_step=self.total_steps)
        self.evaluation_record_steps.append(self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        # 初始化隐藏状态
        self.agent_n.hidden_state = None
        self.agent_n.critic_hidden_state = None

        # 【核心修改】在重置环境时，传入当前课程学习的场景ID
        # 如果是评估模式，我们不使用课程学习，而是可以采用默认场景或轮流测试
        if evaluate:
            obs_n = self.env_evaluate.reset(mode="eval", vel=None)  # 评估时使用默认场景0
        else:
            # 训练时，使用由run()方法动态决定的场景ID
            obs_n = self.env.reset(mode="train", vel=None, scenario_id=self.current_training_scenario)

        done = False
        rewards = np.zeros(self.args.N)
        episode_step = 0

        if self.args.use_reward_scaling and not evaluate:
            self.reward_scaling.reset()

        while not done and episode_step < self.args.episode_limit:
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            obs_next_n, r_n, done_n = self.env.step(a_n)
            rewards += r_n

            # 在训练模式下才存储经验
            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # 只有在回合未结束时才存储，避免存储最后一步的无效信息
                if episode_step < self.args.episode_limit:
                    self.replay_buffer.store_transition(
                        episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
                    )

            obs_n = obs_next_n
            episode_step += 1
            done = any(done_n)

        # 存储最后一步的状态值 (仅在训练模式下)
        if not evaluate:
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step, v_n)

        return rewards, episode_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    # 训练规模与时长参数
    parser.add_argument('--experiment_num', type=str, default="1", help='当前实验的编号/轮次')
    parser.add_argument("--max_train_steps", type=int, default=int(2e6),
                        help=" Maximum number of training steps(定义了整个训练过程的总长度, 智能体与环境的总交互步数（timesteps）达到这个值时，训练将停止。)")  # -4e6 -2e6
    parser.add_argument("--episode_limit", type=int, default=500,
                        help="Maximum number of steps per episode(设置了单个回合(episode)的最长步数)")  # 500    # 模型评估参数
    parser.add_argument("--evaluate_freq", type=float, default=2000,
                        help="Evaluate the policy every 'evaluate_freq' steps-评估当前策略的性能(评估频率).系统暂停常规训练，启动一次独立的性能评估")  # 2000
    parser.add_argument("--evaluate_times", type=int, default=20,
                        help="Evaluate times(“评估次数”。在每次启动性能评估时，模型会连续运行20个独立的回合)")    # 数据与训练批次参数
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size (On-Policy算法进行一次“策略更新”所需的数据量。程序会先运行并收集20个完整回合的经验数据并存入经验池Replay Buffer)")
    parser.add_argument("--mini_batch_size", type=int, default=5,
                        help="Minibatch size (在进行参数更新时，并不会一次性将上述20个回合的数据全部灌入网络，而是会将这个“大批次”切分成若干个“小批次”Mini-batch)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128,
                        help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate") # 8e-4
    # 更低的学习率会使每次更新的步伐更小、更稳定，通常能有效避免这种断崖式的性能下跌，代价是收敛速度可能会稍慢一些
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=50, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True,
                        help="Trick 4:reward scaling. Here, we do not use it.") # False
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.值函数剪切")  # False
    parser.add_argument('--device', default='cuda', help='Device to use (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_1", number=args.experiment_num, seed=0)
    runner.run()
