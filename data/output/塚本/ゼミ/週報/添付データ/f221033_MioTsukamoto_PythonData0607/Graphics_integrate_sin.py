import numpy as np
import matplotlib.pyplot as plt

#sin(x)のグラフを描画
x = np.linspace(0, np.pi, 100)#0からπまでの範囲を100分割した配列
y = np.sin(x)#sin(x)
plt.plot(x, y, label="sin(x)")
plt.xlabel("x")#x軸のラベル
plt.ylabel("y")#y軸のラベル
plt.legend()#凡例を表示
plt.show()