import multiprocessing as mp
import numpy as np

def square(i, x, queue):
    print("In process {}".format(i))
    #np.square(x)：使用NumPy库的square函数计算x的平方
    #queue.put(...)：将计算得到的平方值（以及原始的i值）作为一个元组(i, np.square(x))放入队列queue中
    queue.put((i, np.square(x)))  # 将进程编号和结果一起放入队列

if __name__ == "__main__":
    processes = []  # 存储进程对象
    #Queue是multiprocessing模块提供的一个进程间通信（IPC）机制，允许不同的进程在运行时安全地交换数据
    queue = mp.Queue()  # 用于进程间通信的队列
    x = np.arange(64)  # 输入数据

    # 创建并启动进程
    #循环8次，每次创建一个新的进程
    for i in range(8):
        #计算数据分块的起始索引，计算每个进程要处理的数据块的起始位置
        #第0个进程处理索引 0-7 的数据，第一个进程处理 8-15的数据
        start_index = 8 * i
        proc = mp.Process(target=square, args=(i, x[start_index:start_index + 8], queue))
        proc.start()
        processes.append(proc)

    # 等待所有进程完成
    for proc in processes:
        proc.join()

    # 从队列中获取结果
    results = []
    while not queue.empty():
        #queue.get() 方法从队列头部取出一个元素（先进先出 FIFO 原则）
        results.append(queue.get())

    # 按进程编号排序结果
    #使用 sort 方法对 results 列表进行排序，排序依据是子列表的第一个元素
    #key=lambda x: x[0]指定了排序的依据是每个子列表的第一个元素，会按照子列表的第一个元素的值从小到大对results列表进行排序
    results.sort(key=lambda x: x[0])  # 根据进程编号排序
    #遍历results列表里的每个元素result，把每个result的第二个元素提取出来组合成一个新的列表sorted_results
    sorted_results = [result[1] for result in results]  # 提取排序后的结果

    print("Final results:")
    print(sorted_results)