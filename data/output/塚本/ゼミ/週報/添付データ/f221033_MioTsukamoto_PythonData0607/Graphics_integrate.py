import numpy as np
import matplotlib.pyplot as plt

#log(x)^2のグラフを描画
x = np.linspace(0.01, 2, 100)#0.01から2までの範囲を100分割した配列
y = np.log(x)**2#log(x)^2
plt.plot(x, y, label="log(x)")
plt.xlabel("x")#x軸のラベル
plt.ylabel("y")#y軸のラベル
plt.legend()#凡例を表示
plt.show()


