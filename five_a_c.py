"""
Q-learning 学习预测给定状态和动作下的贴现奖励
策略算法学习给定状态下动作的概率分布
优势演员-评论家算法学习通过比较动作的期望值与实际观察到的奖励来计算优势
多进程是指在多个不同的处理器上运行代码，这些处理器可以同时独立运行
多线程就像多个任务处理，通过让操作系统在多个任务之间快速切换来实现更快运行多个任务。当一个任务空闲是操作系统可以继续处理另一个任务
"""
"""
分布式训练通过同时运行环境的多个实例和一个共享的深度强化学习模型实例来工作。在每个时间步后，计算每个独立模型的损失，收集每个
模型副本的梯度，然后对他们求和或求平均来更新共享参数，从而可以在在没有经验回放缓冲器的情况下进行小批量训练
"""
"""
N-step 学习介于完全在线学习（每次训练一步）和完全的蒙特卡洛学习（只在一个轮次的末尾训练）之间。因此N-step 学习兼具这两者
的优势--1-step 学习的效率和蒙特卡洛学习的准确性
"""
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gymnasium as gym
import torch.multiprocessing as mp #pytorch封装了 Python内置的multiprocessing库，且API相同

class ActorCritic(nn.Module): #为演员和评论家定义一个组合模型
    def __init__(self):
        #调用父类 nn.Module 的构造函数，确保正确初始化
        super(ActorCritic, self).__init__()
        #网络架构：共享层（l1、l2）、策略网络（actor_lin1）、价值网络（l3、critic_lin1）
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)  #输出动作概率分布
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)   #输出状态价值估计
    def forward(self,x):
        #对输入状态 x 沿维度 0 进行归一化处理
        x = F.normalize(x,dim=0)
        #将归一化后的状态通过共享层l1和l2，使用relu激活函数u
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        #log_softmax逻辑上等同于log(softmax(..))，但组合后的函数数值上更稳定。如果单独计算函数，则在softmax之后可能会得到满溢或下溢概率
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #演员这一头返回两个动作的对数概率
        #使用 y.detach() 创建一个分离的副本，阻止梯度从Critic流向共享层，
        #所以评论家损失不会在第一层和第二层中反向传播和修改权重。
        #只有演员才会导致这些权重被修改，所以当演员和评论家试图对前面的网络层做出相反的更新时，这能够防止二者之间的冲突
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) #评论家返回一个以-1 和+1 位界限的数字
        return actor, critic #以元组形式返回演员和评论家结果


#运行一个轮次。负责在一个完整的游戏回合中收集训练所需的数据
#worker_env: 游戏环境实例（这里是 CartPole 游戏）
#worker_model: Actor-Critic 神经网络模型实例
def run_episode(worker_env, worker_model):
    #将环境状态从 numpy 数组转为 pytorch张量
    state = torch.from_numpy(worker_env.env.state).float() 
    #values: 保存每一步的状态价值估计
    #logprobs: 保存每一步所采取动作的对数概率
    #rewards: 保存每一步的奖励
    values, logprobs, rewards = [],[],[]  #创建列表存储计算的状态值(评论家)、对数概率(演员)和奖励
    done = False
    j=0  #步数计数器
    while (done == False):   #持续玩游戏，直到轮次结束
        j+=1
        #将当前状态输入 Actor-Critic 模型，获取动作策略和状态价值
        policy, value = worker_model(state) 
        values.append(value)
        #.view(-1)：是PyTorch中的一个方法，用于改变张量的形状而不改变其数据。
        #.view()方法接受一个或多个整数作为参数，这些整数指定了新张量的维度。
        logits = policy.view(-1)
        #torch.distributions 是PyTorch中用于概率分布的一个模块。包含了许多常用的概率分布，用于生成随机数、计算概率密度函数等。
        #Categorical 分布是一种离散概率分布，用于表示一个有限数量的可能结果的概率。
        #logits:PyTorch张量，通常表示未归一化的对数概率
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() 
        logprob_ = policy.view(-1)[action] #获取选中动作的对数概率
        logprobs.append(logprob_)  
        #step方法是环境对象的一个方法，用于执行一个动作并接收环境的反馈。
        #.detach()方法用于从当前计算图中分离张量，使其不再需要梯度（这在执行动作时通常是必要的，因为动作不需要进行反向传播）。
        state_, _, done, _, info = worker_env.step(action.detach().numpy())
        #from_numpy()：这是PyTorch中的一个函数，用于将NumPy数组转换成PyTorch张量。
        state = torch.from_numpy(state_).float()
        if done: 
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards
"""
run_episode函数只运行于CartPole的单个轮次，并从评论家那里收集计算的状态值，从演员那里收集动作的对数概率以及来自环境的奖励。
这是一个演员-评论家算法而不是Q-learning算法，因此采取动作是直接从策略中抽样，而不是在Q-learning中任意选择一个策略(eg:epsilon贪婪策略)
"""


"""
update_params 函数是所有动作所在之处，他将分布式优势演员-评论家算法与目前学到的其他算法区分开来。
首先，获取奖励、对数概率和状态-价值的列表，并将他们转换为 pytorch 张量。
由于我们希望首先考虑最近的动作，因此对其进行逆序处理，并通过调用 view(-1)方法确保他们是扁平的一维数组。
actor_loss 的计算使用的是优势（技术上是基线，因为不存在自举）而不是原始奖励。
如果使用actor_loss 就必须将values张量从计算图中分离出来；否则将同时通过演员和评论家进行反向传播，但其实我们只想跟新演员头。
评论家损失仅仅是状态价值与回报的平方差，此处要确保没有进行分离，因为我们想要更新评论家头
然后将演员和评论家损失求和得到整体损失。通过乘0.1 来缩小评论家的损失，因为我们希望演员比评论家学习的更快。
此外还需要返回每个损失和奖励张量的长度（他表明轮次持续了多久）以监视他们的训练进展。
"""
#计算和最小化损失
#我们对values,logprobs,rewards数组进行逆序处理并调用 view(-1)来确保他们是扁平的一维数组
def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
        #将rewards转换成一个PyTorch张量，沿着第0维（最外层维度）翻转它，然后将其展平成一维张量。
        #rewards：将奖励列表转换为张量
        #.flip()函数用于将张量沿着指定的维度翻转。dims=(0,)指定第0维进行翻转
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) 
        #logprobs 和 values：使用 torch.stack 堆叠张量
        #torch.stack 函数用于将输入的张量（tensor）列表沿着一个新的维度堆叠起来。
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0]) #初始化折扣回报 ret_ 为零张量
        #从后向前循环（因为已经翻转了）
        for r in range(rewards.shape[0]): 
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)  #这实现了贝尔曼方程的逆向展开
        Returns = torch.stack(Returns).view(-1)  #将计算出的回报堆叠并展平
        Returns = F.normalize(Returns,dim=0) #归一化回报，使其范围适合训练
        #带基线的策略梯度损失
        #-1*logprobs：取反使得梯度上升变为梯度下降（优化器默认是梯度下降）
        #(Returns - values.detach())：优势函数，表示实际回报相对于预期价值的差异
        #values.detach()：分离价值估计，防止梯度流入价值网络
        #这种损失函数鼓励提高带来高回报的动作的概率，降低带来低回报的动作的概率
        actor_loss = -1*logprobs * (Returns - values.detach())
        #使用 torch.pow 函数对误差进行平方操作 
        critic_loss = torch.pow(values - Returns,2) 
        loss = actor_loss.sum() + clc*critic_loss.sum() 
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)

"""
worker函数是每个独立进程都将单独运行的函数。每个woker(进程)都将创建自己的CartPole环境和优化器，但会共享
演员-评论家模型会作为一个参数传入函数。模型是共享的，因此当一个worker更新模型参数时，worker 中的参数都会更新
在每个进程中，使用共享的模型来运行游戏的一个轮次。每个进程中都会计算损失，但优化器会更新每个进程使用的共享演员-评论家模型
 由于每个 worker 都是拥有自己内存的新进程中生成的，因此他需要的所有数据都应该作为参数显示的传递给函数，这可以防止程序出现漏洞
"""
def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #每个进程运行自己独立的环境和优化器，但共享模型
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        #run_episode函数将运行游戏的一个轮次，并沿途收集数据
        values, logprobs, rewards = run_episode(worker_env,worker_model) 
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards) 
    #使用从run_episode收集的数据允许一个参数更新步骤    
    update_params(worker_opt,values,logprobs,rewards)
    #counter是所有运行的进程之间的全局共享计数器
    counter.value=counter.value+1
    run_episode


if __name__ == '__main__':
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []
    params = {'epochs': 1000, 'n_workers': 7, }
    counter = mp.Value('i', 0)
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)


    env = gym.make("CartPole-v1")
    env.reset()

    for i in range(100):  #运行100个时间步
        ## 获取当前环境状态并转换为PyTorch张量
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits,value = MasterNode(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, info, _ = env.step(action.detach().numpy())
        if done:
            print("Lost")
            env.reset()
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()

    """
    N-step 学习意味着只会在N 步之后计算损失和更新参数。如果N 为 1，为完全在线学习；如果N很大，则为蒙特卡洛学习；最佳点位于两者之间。
    N-step 学习比 1-step（在线）学习更好的原因是：评论家的目标值更准确，所以评论家的训练会更稳定，并能够产生更小偏差的状态-价值。
    通过自举，我们从预测中做出预测，所以如果能在预测之前收集更多的数据，那么预测会更好。
    另外，自举能够提高抽样效率，在正确方向上更新参数之前不需要查看太多数据
    """
    #这段代码包含两个主要部分：一个改进版的 run_episode 函数和一个演示折扣回报计算的示例
    def run_episode(worker_env, worker_model, N_steps=10):
        raw_state = np.array(worker_env.env.state)
        #将环境状态从 numpy 数组转为 pytorch张量
        state = torch.from_numpy(raw_state).float()
        values, logprobs, rewards = [],[],[]
        done = False
        j=0
        G=torch.Tensor([0]) #变量G 表示收益，初始化为 0
        while (j < N_steps and done == False): #玩游戏直到N 步或当轮次结束
            j+=1
            policy, value = worker_model(state)
            values.append(value)
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            logprob_ = policy.view(-1)[action]
            logprobs.append(logprob_)
            state_, _, done, info = worker_env.step(action.detach().numpy())
            state = torch.from_numpy(state_).float()
            if done:
                reward = -10
                worker_env.reset()
            else: #如果轮次尚未结束，则将收益设置为前一个状态-价值
                reward = 1.0
                G = value.detach()
            rewards.append(reward)
        return values, logprobs, rewards, G

r1 = [1,1,-1]  #两个不同的奖励序列
r2 = [1,1,1]
R1,R2 = 0.0,0.0

#不使用引导的回报计算：从零开始累积折扣回报
for i in range(len(r1)-1,0,-1): #训练不包括索引0
    R1 = r1[i] + 0.99*R1
for i in range(len(r2)-1,0,-1):
    R2 = r2[i] + 0.99*R2
print("No bootstrapping")
print(R1,R2)

#使用引导的回报计算：从一个非零值（这里是1.0）开始累积折扣回报
R1,R2 = 1.0,1.0  #初始化为1.0模拟引导值
for i in range(len(r1)-1,0,-1):
    R1 = r1[i] + 0.99*R1
for i in range(len(r2)-1,0,-1):
    R2 = r2[i] + 0.99*R2
print("With bootstrapping")
print(R1,R2)