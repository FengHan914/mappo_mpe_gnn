import argparse

from env.environment import MutiAgentEnv
from mappo_mpe import *
from scenes.entrap import Entrap_Scene

import argparse


# ...其他 import...

def get_display_width(s):
    """计算字符串的显示宽度，中文计为2，英文计为1"""
    width = 0
    for char in s:
        if '\u4e00' <= char <= '\u9fff':  # 判断是否为中文字符
            width += 2
        else:
            width += 1
    return width


parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
parser.add_argument("--max_train_steps", type=int, default=int(5e4), help=" Maximum number of training steps")
parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
parser.add_argument("--evaluate_freq", type=float, default=2000, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--evaluate_times", type=float, default=10, help="Evaluate times")

parser.add_argument("--batch_size", type=int, default=10, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=5, help="Minibatch size (the number of episodes)")
parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the mlp")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=25, help="GAE parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False,
                    help="Trick 4:reward scaling. Here, we do not use it.")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
parser.add_argument("--add_agent_id", type=float, default=False,
                    help="Whether to add agent_id. Here, we do not use it.")
parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

parser.add_argument("--N", type=int, default=3, help="Batch size (the number of episodes)")
parser.add_argument("--action_dim", type=int, default=2, help="Batch size (the number of episodes)")
parser.add_argument("--obs_dim", type=int, default=7, help="Batch size (the number of episodes)")
parser.add_argument("--state_dim", type=int, default=21, help="Batch size (the number of episodes)")

args = parser.parse_args()

# 建立环境
scene = Entrap_Scene(pursuer_num=3, evader_num=1)  # 设置博弈智能体数量
env = MutiAgentEnv(scene, reset_callback=scene.reset_world, reward_callback=scene.reward,
                   observation_callback=scene.observation)

args.N = env.agents_num
args.obs_dim_n = env.obs_n
args.action_dim_n = env.act_n

# 指定训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建智能体
# agent_n = MAPPO_MPE(args)

# 加载模型参数
step = 5000000  # 加载model
algorithm_name = "MAPPO"
path = "./model/" + algorithm_name + "/16" + "/model_{}.pth".format(step)
agent_n = torch.load(path)
num_test_scenarios = 3  # 我们在entrap.py中定义了0, 1, 2共3个场景
rewards = np.zeros(args.N)  # 初始化奖励

# test.py (替换从 "--- 模型测试 ---" 注释开始的整个循环)

# --- 模型测试 ---
# 获取静态场景信息
pursuer_num = env.scene.pursuer_num
evader_num = env.scene.evader_num
has_obstacles = '有' if env.scene.obstacle_num > 0 else '无'
speed_ratio = env.scene.v_max / (env.scene.v_max * 0.8) if env.scene.v_max > 0 else float('inf')

# 【修改】设置总测试次数和可视化次数
test_num = 1000
visualize_episodes = 0
successful_episodes = 0
# 【新增】用于记录每次成功捕获到目标边界的距离
capture_distances_to_boundary = []

# Set random seed
# seed = 0

# 随机数种子2-84% 3-80% 4-84% 5-86% 6-79.6%
seed = 7
np.random.seed(seed)
torch.manual_seed(seed)

print("\n" + "=" * 120)  # 增加了总宽度
print(f"--- 开始模型测试 (共 {test_num} 轮, 前 {visualize_episodes} 轮进行可视化) ---")
print(f"--- 加载模型: {path} ---")

# 【新增】定义列的宽度和表头
column_widths = [8, 8, 8, 8, 16, 8, 12, 12]  # 调整了列宽
header_titles = ['测试轮次', '结果', '用时(步)', '总奖励', '成功位置', '是否碰撞', '距目标边界', '当前成功率']  # 增加了新列
header_line = "| "
for title, width in zip(header_titles, column_widths):
    padding = ' ' * (width - get_display_width(title))
    header_line += f"{title}{padding} | "
print(header_line)
print("-" * 120)  # 增加了总宽度

# 循环进行多次测试
for i in range(test_num):

    # 【新增】轮流选择一个场景ID进行测试
    # scenario_to_run = i % num_test_scenarios
    scenario_to_run = 0

    # 【新增】当可视化结束时，打印提示信息

    if i == visualize_episodes:
        print("-" * 120)
        print(
            f"--- 前 {visualize_episodes} 轮可视化测试完成，开始加速执行剩余 {test_num - visualize_episodes} 轮测试 ---")
        print("-" * 120)

    obs_n = env.reset(mode="eval", vel=None, scenario_id=scenario_to_run)
    rewards = np.zeros(args.N)
    done = False
    step_count = 0

    while not done:
        # 【修改】只有在前 visualize_episodes 轮才调用 render
        if i < visualize_episodes:
            env.render()

        with torch.no_grad():
            actions, _ = agent_n.choose_action(obs_n, evaluate=True)
        obs_next_n, r_n, done_n = env.step(np.array(actions.cpu()))
        rewards += r_n
        step_count += 1
        obs_n = obs_next_n
        done = (
                env.scene.check_capture_condition() or
                env.scene.check_evader_escape_condition() or
                step_count >= args.episode_limit or
                env.scene.pursuer_collision_occurred
        )

    # ... (回合结束后的信息收集与打印逻辑保持不变) ...
    termination_reason_str = ""
    # 【新增】初始化距离字符串
    dist_to_boundary_str = "N/A"

    if env.scene.capture_position is not None:
        termination_reason_str = "任务成功"
        successful_episodes += 1
        # 【新增】计算到目标区域边界的距离
        capture_pos = np.array(env.scene.capture_position)
        target_center = np.array(env.scene.target_area)
        target_radius = env.scene.target_radius

        distance_to_center = np.linalg.norm(capture_pos - target_center)
        distance_to_boundary = distance_to_center - target_radius
        capture_distances_to_boundary.append(distance_to_boundary)
        dist_to_boundary_str = f"{distance_to_boundary:.2f}"

    elif env.scene.pursuer_collision_occurred:
        termination_reason_str = "碰撞失败"
    elif env.scene.check_evader_escape_condition():
        termination_reason_str = "逃逸者逃脱"
    elif step_count >= args.episode_limit:
        termination_reason_str = "任务超时"
    else:
        termination_reason_str = "其他情况"

    total_reward = np.sum(rewards)
    capture_pos_str = f"({env.scene.capture_position[0]:.1f}, {env.scene.capture_position[1]:.1f})" if env.scene.capture_position is not None else "N/A"
    collision_str = "是" if env.scene.pursuer_collision_occurred else "否"
    current_success_rate = successful_episodes / (i + 1)

    # 【新增】更新数据列表以包含新列
    data_list = [f'{i + 1}/{test_num}', termination_reason_str, str(step_count), f'{total_reward:.2f}', capture_pos_str,
                 collision_str, dist_to_boundary_str, f'{current_success_rate:.1%}', ]
    output_line = "| "
    for item, width in zip(data_list, column_widths):
        padding = ' ' * (width - get_display_width(item))
        output_line += f"{item}{padding} | "
    # 此处原始代码可能有误，修正为在循环外添加场景ID，以避免重复打印
    output_line += f"场景ID: {scenario_to_run}"
    print(output_line)

print("-" * 120)
final_success_rate = successful_episodes / test_num if test_num > 0 else 0
print(f"最终训练总结: 成功率 = 成功次数/总训练次数 = {successful_episodes}/{test_num} = {final_success_rate:.2%}")

# 【新增】计算并打印平均距离
if capture_distances_to_boundary:
    avg_dist = np.mean(capture_distances_to_boundary)
    print(f"成功捕获位置距目标区域边界的平均距离: {avg_dist:.2f}")
else:
    print("没有成功的捕获，无法计算平均距离。")

print(
    f"场景配置: {pursuer_num}个追捕者 vs {evader_num}个逃逸者 | 速度比(追捕者/逃逸者): {speed_ratio:.2f} | 障碍物信息: {has_obstacles}")
print("--- 测试结束 ---")

# 【新增】在程序最后关闭env，释放资源
env.close()