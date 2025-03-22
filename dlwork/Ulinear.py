import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成一些随机数据
np.random.seed(0)
x_1d = np.random.rand(100, 1)
y_1d = 2 * x_1d + 1 + 0.5 * np.random.randn(100, 1)

# 创建并拟合一元线性回归模型
model_1d = LinearRegression()
model_1d.fit(x_1d, y_1d)

# 预测值
y_pred = model_1d.predict(x_1d)

# 绘制散点图和拟合直线
plt.scatter(x_1d, y_1d, color='blue', label='Actual data')
plt.plot(x_1d, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Unitary linear regression')
plt.legend()
plt.show()