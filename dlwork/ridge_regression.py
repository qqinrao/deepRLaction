import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

X = np.array([[15, 20, 25, 30, 35, 40], [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]]).T
y = np.array([136, 140, 155, 160, 157, 175])

print("温度特征与花朵数量相关系数：\n", np.corrcoef(X[:, 0], y))
print("湿度特征与花朵数量相关系数：\n", np.corrcoef(X[:, 1], y))
print("温度特征与湿度特征相关系数：\n", np.corrcoef(X[:, 0], X[:, 1]))


class Ridge:
    def __init__(self):
        self.W = None

    def fit(self, x, y):
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return self.W

    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))
        pred = np.dot(X, self.W)
        return pred


# 创建并拟合模型
Y = y.reshape((-1, 1))
model = Ridge()
model.fit(X, Y)
W = model.W
print("岭回归系数：\n", W)
# 预测
y_test = model.predict(X)
print("预测结果：\n", y_test)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, marker='o', label='Actual Values', color='blue')
plt.plot(range(len(y_test)), y_test, marker='s', label='Predicted Values', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values in Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()