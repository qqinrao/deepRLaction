import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#生成数据
np.random.seed(0)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 * x ** 2 + 3 * x + 1 + 0.5 * np.random.randn(100, 1)

#多项式特征转换
degree = 2
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)

#创建并拟合线性回归模型
model = LinearRegression()
model.fit(x_poly, y)

#预测值
y_pred = model.predict(x_poly)

#绘制图形
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', linewidth=2, label=f'P-regression (degree={degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('P-Regression')
plt.legend()
plt.show()
    