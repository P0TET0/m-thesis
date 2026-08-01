import numpy as np
from scipy.integrate import quad

a = 0
b = np.pi / 2


#sin(x)の積分を求める
def f_1(x):
    return np.sin(x)  # sin(x)を返す関数

# 積分範囲を0からπとして積分を計算
result, error = quad(f_1, a, b)  # 積分結果と推定誤差を返す,積分範囲は0からπ/2
print("sin(x)の積分(0からπ/2までの範囲で計算")
print("積分結果:", result)
print("推定誤差:", error)

#cos(x)の積分を求める
def f_2(x):
    return np.cos(x)  # cos(x)を返す関数

# 積分範囲を0からπとして積分を計算
result, error = quad(f_2, a, b)  # 積分結果と推定誤差を返す,積分範囲は0からπ/2
print("cos(x)の積分(0からπ/2までの範囲で計算")
print("積分結果:", result)
print("推定誤差:", error)


#tan(x)の積分を求める
def f_3(x):
    return np.tan(x)  # tan(x)を返す関数

b = np.pi/2 - 0.0001 #発散しないようにπ/2から0.0001引いた値を積分範囲の終了値とする

# 積分範囲を0からπとして積分を計算
result, error = quad(f_3, a, b)  # 積分結果と推定誤差を返す,積分範囲は0からπ/2
print("tan(x)の積分(0からπ/2までの範囲で計算")
print("積分結果:", result)
print("推定誤差:", error)


def f_4(x):
    return np.sin(x) * np.cos(x)  # sin(x) * cos(x)を返す関数

b = np.pi / 2

# 積分範囲を0からπとして積分を計算
result, error = quad(f_4, a, b)  # 積分結果と推定誤差を返す,積分範囲は0からπ/2
print("sin(x) * cos(x)の積分(0からπ/2までの範囲で計算")
print("積分結果:", result)
print("推定誤差:", error)