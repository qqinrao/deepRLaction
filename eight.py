##初始化环境和神经网络


# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import time
from PIL import Image

# 确保环境正确创建
try:
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    
    # 创建环境
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    print("环境创建成功!")
except Exception as e:
    print(f"环境创建失败: {e}")
    raise

# 图像处理函数
def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    """使用PIL缩小观察图像"""
    if obs is None:
        # 创建空白图像
        if to_gray:
            return np.zeros(new_size, dtype=np.float32)
        else:
            return np.zeros((*new_size, 3), dtype=np.float32)
    
    # 确保obs是uint8类型
    if obs.dtype != np.uint8:
        obs = np.uint8(obs)
    
    # PIL图像处理
    img = Image.fromarray(obs)
    img = img.resize((new_size[1], new_size[0]), Image.LANCZOS)
    
    if to_gray:
        img = img.convert('L')
        return np.array(img) / 255.0  # 归一化
    else:
        return np.array(img) / 255.0  # 归一化

# 检查环境
state = env.reset()
print(f"环境初始状态类型: {type(state)}, 形状: {state.shape if hasattr(state, 'shape') else 'Unknown'}")

# 尝试获取屏幕图像
try:
    screen = env.render(mode='rgb_array')
    plt.figure(figsize=(8, 8))
    plt.imshow(screen)
    plt.title("Mario 游戏画面")
    plt.show()
    
    # 测试缩放函数
    downscaled = downscale_obs(screen)
    plt.figure(figsize=(5, 5))
    plt.imshow(downscaled, cmap='gray')
    plt.title("缩放后的灰度图像")
    plt.show()
    
    print("渲染和图像处理正常!")
except Exception as e:
    print(f"渲染失败: {e}")




##定义神经网络和辅助函数
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
params = {
    'batch_size': 32,
    'gamma': 0.99,
    'eps_start': 1.0,
    'eps_end': 0.1,
    'eps_decay': 200,
    'learning_rate': 0.0005,
    'frames_per_state': 4,
    'action_repeats': 4,
    'max_episode_len': 500,
    'min_progress': 5,
    'intrinsic_weight': 0.01,
    'forward_loss_weight': 0.2,
    'inverse_loss_weight': 0.1,
}

# Q网络
class QNetwork(nn.Module):
    def __init__(self, params):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(params['frames_per_state'], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 12)  # 动作数量是12

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 编码器
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(params['frames_per_state'], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 使用自适应池化来确保固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 无论输入大小如何，输出总是 64*4*4
        self.fc = nn.Linear(64 * 4 * 4, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 使用自适应池化将特征图调整为固定大小
        x = self.adaptive_pool(x)
        
        # 现在特征图始终是 [batch_size, 64, 4, 4]
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

# 前向模型
class ForwardModel(nn.Module):
    def __init__(self, params):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(512 + 1, 512)  # 512维特征 + 1维动作
        self.fc2 = nn.Linear(512, 512)

    def forward(self, state_features, action):
        x = torch.cat([state_features, action.float().unsqueeze(1)], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 逆向模型
class InverseModel(nn.Module):
    def __init__(self, params):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(512 * 2, 512)  # 两个状态的特征
        self.fc2 = nn.Linear(512, 12)  # 12个动作

    def forward(self, state1_features, state2_features):
        x = torch.cat([state1_features, state2_features], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add_memory(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

# 动作选择策略
def policy(q_values, epsilon=0.0):
    if random.random() < epsilon:
        return random.randint(0, 11)  # 随机选择动作
    else:
        return q_values.argmax(1).item()  # 贪婪选择最佳动作

# 状态处理函数
def prepare_state(state):
    """处理单个状态帧"""
    if state is None:
        return torch.zeros((1, 42, 42), dtype=torch.float32)
    return torch.from_numpy(downscale_obs(state)).float().unsqueeze(0)

def get_screen(env):
    """安全获取游戏画面"""
    try:
        screen = None
        methods = [
            lambda: env.render(mode='rgb_array'),
            lambda: env.render(),
            lambda: env.unwrapped.ale.getScreenRGB(),
        ]
        
        for method in methods:
            try:
                screen = method()
                if screen is not None and isinstance(screen, np.ndarray):
                    return screen
            except Exception as e:
                continue
                
        if screen is None:
            return np.zeros((240, 256, 3), dtype=np.uint8)
        return screen
    except Exception as e:
        print(f"获取画面错误: {e}")
        return np.zeros((240, 256, 3), dtype=np.uint8)

def reset_env():
    """重置环境并初始化状态队列"""
    global env, state_deque
    
    state = env.reset()
    
    # 清空状态队列
    state_deque.clear()
    
    # 填充状态队列
    for _ in range(params['frames_per_state']):
        screen = get_screen(env)
        state_deque.append(prepare_state(screen))
    
    # 堆叠为初始状态
    initial_state = torch.stack(list(state_deque), dim=1)
    return initial_state



##初始化模型
# 创建神经网络
Qmodel = QNetwork(params).to(device)
target_Qmodel = QNetwork(params).to(device)
target_Qmodel.load_state_dict(Qmodel.state_dict())

encoder = Encoder(params).to(device) 
forward_model = ForwardModel(params).to(device)
inverse_model = InverseModel(params).to(device)

# 优化器
all_params = list(Qmodel.parameters()) + list(encoder.parameters()) + \
             list(forward_model.parameters()) + list(inverse_model.parameters())
opt = torch.optim.Adam(all_params, lr=params['learning_rate'])

# 经验回放
replay = ReplayMemory(10000)

# 训练函数
def minibatch_train(use_extrinsic=True):
    """训练 Q 网络和内在奖励模型"""
    # 从经验回放中采样
    s1_batch, a_batch, r_batch, s2_batch = replay.sample(params['batch_size'])
    
    # 将数据移动到设备
    s1_batch = s1_batch.to(device)
    a_batch = a_batch.to(device)
    r_batch = r_batch.to(device)
    s2_batch = s2_batch.to(device)
    
    # 检查批处理大小
    print(f"s1_batch 形状: {s1_batch.shape}")
    print(f"s2_batch 形状: {s2_batch.shape}")
    
    # 确保批处理大小至少为1
    if s1_batch.shape[0] == 0 or s2_batch.shape[0] == 0:
        print("警告：空批处理，跳过训练")
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    # 计算下一个状态的目标Q值
    with torch.no_grad():
        q_vals_next = target_Qmodel(s2_batch)
        max_q_vals_next = q_vals_next.max(1)[0].unsqueeze(1)
    
    # 计算当前状态的Q值
    q_vals = Qmodel(s1_batch)
    q_vals_for_actions = q_vals.gather(1, a_batch.unsqueeze(1))
    
    # 计算内在奖励
    encoded_s1 = encoder(s1_batch)
    encoded_s2 = encoder(s2_batch)
    
    # 使用前向模型预测下一个状态的特征
    pred_s2_features = forward_model(encoded_s1, a_batch)
    forward_pred_err = F.mse_loss(pred_s2_features, encoded_s2.detach(), reduction='none')
    
    # 使用逆向模型预测动作
    pred_actions = inverse_model(encoded_s1, encoded_s2)
    inverse_pred_err = F.cross_entropy(pred_actions, a_batch, reduction='none').unsqueeze(1)
    
    # 计算内在奖励
    intrinsic_reward = forward_pred_err.mean(dim=1, keepdim=True)
    
    # 计算总奖励
    if use_extrinsic:
        total_reward = r_batch + params['intrinsic_weight'] * intrinsic_reward
    else:
        total_reward = params['intrinsic_weight'] * intrinsic_reward
    
    # 计算目标Q值
    target_q_vals = total_reward + params['gamma'] * max_q_vals_next
    
    # 计算Q损失
    q_loss = F.mse_loss(q_vals_for_actions, target_q_vals.detach(), reduction='none')
    
    return forward_pred_err, inverse_pred_err, q_loss

def loss_fn(q_loss, forward_pred_err, inverse_pred_err):
    """组合损失函数"""
    q_loss_mean = q_loss.mean()
    forward_loss = forward_pred_err.mean()
    inverse_loss = inverse_pred_err.mean()
    
    return q_loss_mean + params['forward_loss_weight'] * forward_loss + \
           params['inverse_loss_weight'] * inverse_loss

def update_target_network():
    """更新目标Q网络"""
    target_Qmodel.load_state_dict(Qmodel.state_dict())


##训练循环
# 初始化状态队列
state_deque = deque(maxlen=params['frames_per_state'])

# 重置环境并获取初始状态
state = env.reset()
for _ in range(params['frames_per_state']):
    screen = get_screen(env)
    state_deque.append(prepare_state(screen))
state1 = torch.stack(list(state_deque), dim=1)

# 训练参数
epochs = 5000
eps = 0.15
losses = []
target_update_freq = 1000
episode_length = 0
e_reward = 0.0
last_x_pos = 0
done = False

# 主训练循环
try:
    for i in range(epochs):
        opt.zero_grad()
        episode_length += 1
        
        # 选择动作
        if state1.dim() == 3:  # 如果缺少批处理维度
            state1_batch = state1.unsqueeze(0)  # 添加批处理维度
        else:
            state1_batch = state1
        q_val_pred = Qmodel(state1_batch.to(device))
        if i > 1000:  # 使用epsilon-greedy策略
            action = policy(q_val_pred, eps)
        else:
            action = policy(q_val_pred)
        
        # 执行动作多次
        for j in range(params['action_repeats']):
            # 兼容不同版本的gym
            result = env.step(action)
            if len(result) == 4:  # 旧版gym
                state2, reward, done, info = result
                trunc = False
            else:  # 新版gymnasium
                state2, reward, done, trunc, info = result
            
            # 记录位置
            last_x_pos = info.get('x_pos', 0)
            
            # 如果游戏结束，重置环境
            if done:
                # 重置环境
                state = env.reset()
                state_deque.clear()
                for _ in range(params['frames_per_state']):
                    screen = get_screen(env)
                    state_deque.append(prepare_state(screen))
                break
            
            # 累计奖励并更新状态队列
            e_reward += reward
            screen = get_screen(env)
            state_deque.append(prepare_state(screen))
        
        # 构建新状态
        state2 = torch.stack(list(state_deque), dim=1)
        
        # 将经验添加到回放缓冲区
        replay.add_memory(state1, torch.tensor(action), torch.tensor(e_reward), state2)
        e_reward = 0
        
        # 处理长时间没有进展的情况
        if episode_length > params['max_episode_len']:
            if (info.get('x_pos', 0) - last_x_pos) < params['min_progress']:
                done = True
            else:
                last_x_pos = info.get('x_pos', 0)
        
        # 如果游戏结束，重置状态
        if done or trunc:
            # 重置环境
            state = env.reset()
            state_deque.clear()
            for _ in range(params['frames_per_state']):
                screen = get_screen(env)
                state_deque.append(prepare_state(screen))
            state1 = torch.stack(list(state_deque), dim=1)
            last_x_pos = 0
            episode_length = 0
        else:
            state1 = state2
        
        # 如果回放缓冲区中有足够的样本，进行训练
        if len(replay.memory) >= params['batch_size']:
            forward_pred_err, inverse_pred_err, q_loss = minibatch_train(use_extrinsic=True)
            loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err)
            loss_list = (q_loss.mean().item(), 
                         forward_pred_err.mean().item(),
                         inverse_pred_err.mean().item())
            losses.append(loss_list)
            
            # 反向传播
            loss.backward()
            opt.step()
        
        # 定期更新目标网络
        if i % target_update_freq == 0:
            update_target_network()
            print(f"Step {i}: 更新目标网络")
        
        # 输出进度
        if i % 100 == 0:
            print(f"Step {i}/{epochs}, Position: {last_x_pos}")
    
    print("训练完成!")
    
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    
finally:
    # 关闭环境
    env.close()


##可视化结果
# 可视化损失
if losses:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot([x[0] for x in losses])
    plt.title('Q 损失')
    plt.xlabel('训练步数')
    
    plt.subplot(1, 3, 2)
    plt.plot([x[1] for x in losses])
    plt.title('前向模型损失')
    plt.xlabel('训练步数')
    
    plt.subplot(1, 3, 3)
    plt.plot([x[2] for x in losses])
    plt.title('逆向模型损失')
    plt.xlabel('训练步数')
    
    plt.tight_layout()
    plt.show()
else:
    print("没有损失数据可供可视化")