import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#生成多元随机数据
np.random.seed(0)
#生成 100 个样本，每个样本有 3 个特征
x_multiple = np.random.rand(100, 3)
#定义真实的线性关系生成目标值，并添加一些噪声
y_multiple = 2 * x_multiple[:, 0] + 3 * x_multiple[:, 1] - 1 * x_multiple[:, 2] + 2 + 0.5 * np.random.randn(100)
y_multiple = y_multiple.reshape(-1, 1)

#创建并拟合多元线性回归模型
model_multiple = LinearRegression()
model_multiple.fit(x_multiple, y_multiple)

#预测值
y_pred = model_multiple.predict(x_multiple)

#可视化
plt.scatter(x_multiple[:, 0], y_multiple, color='blue', label='Actual data')
plt.scatter(x_multiple[:, 0], y_pred, color='red', label='Predicted data')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Multiple-Regression-Feature 1)')
plt.legend()
plt.show()
    