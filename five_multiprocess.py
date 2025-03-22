import multiprocessing as mp
import numpy as np
def square(x): #这个函数接收一个数组并对其中的元素求平方
    return np.square(x)

x = np.arange(64) #设置一个包含数字序列的数组
print(x)

mp.cpu_count()

if __name__ == '__main__':
    pool = mp.Pool(8)  # 设置一个包含 8 个进程的多进程处理器池
    # 使用池的 map 函数将 square 函数应用于列表中的每个数组，并以列表形式返回结果
    squared = pool.map(square, [x[8 * i:8 * i + 8] for i in range(8)])
    print(squared)
    pool.close()
    pool.join()


