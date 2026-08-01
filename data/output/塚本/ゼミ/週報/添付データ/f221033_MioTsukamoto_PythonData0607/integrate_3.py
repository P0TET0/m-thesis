import numpy as np
from scipy.integrate import quad

def f_1(x):
    return (1/(2*np.sqrt(x)))  # 1/(2*√x)を返す関数

# 積分範囲を0から1として積分を計算
result, error = quad(f_1, 1,2)  # 積分結果と推定誤差を返す,積分範囲は0から1
print("積分結果:", result)
print("推定誤差:", error)


#回答は√2-1
print("回答の数値と一致しているか確認：")
print(np.sqrt(2) - 1)


def f_2(x):
    return (1/(1+x**2))  # 1/(1+x^2)を返す関数

# 積分範囲を1から√3として積分を計算
result, error = quad(f_2, 1, np.sqrt(3))  # 積分結果と推定誤差を返す,積分範囲は1から√3
print("積分結果:", result)
print("推定誤差:", error)

#回答はπ/12
print("回答の数値と一致しているか確認：")
print(np.pi/12)


#x*e^xの積分を求める
def f_3(x):
    return x*np.exp(-x)  # x*e^-xを返す関数

# 積分範囲を0から1として積分を計算
result, error = quad(f_3, 0, 1)  # 積分結果と推定誤差を返す,積分範囲は0から1
print("積分結果:", result)
print("推定誤差:", error)

#回答は1-2/e
print("回答の数値と一致しているか確認：")
print(1-2/np.e)

