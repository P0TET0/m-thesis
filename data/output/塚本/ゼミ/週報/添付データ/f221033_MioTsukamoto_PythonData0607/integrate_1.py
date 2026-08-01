import numpy as np
from scipy.integrate import quad


#log(1+x)の積分を求める
def f_1(x):
    return np.log(1+x)  # log(1+x)を返す関数

# 積分範囲を0から1として積分を計算
result, error = quad(f_1, 0, 1)  # 積分結果と推定誤差を返す,積分範囲は0から1
print("積分結果:", result)
print("推定誤差:", error)


#1/log(x)の積分を求める
def f_2(x):
    return 1/np.log(x)  # 1/log(x)を返す関数

# 積分範囲を2から3として積分を計算
result, error = quad(f_2, 2, 3)  # 積分結果と推定誤差を返す,積分範囲は2から3
print("積分結果:", result)
print("推定誤差:", error)


#log(x)^2の積分を求める
def f_3(x):
    return np.log(x)**2  # log(x)^2を返す関数

# 積分範囲を1から2として積分を計算
result, error = quad(f_3, 1, 2)  # 積分結果と推定誤差を返す,積分範囲は1から2
print("積分結果:", result)
print("推定誤差:", error)


#log(x)*xの積分を求める
def f_4(x):
    return np.log(x)*x  # log(x)*xを返す関数

# 積分範囲を1から2として積分を計算
result, error = quad(f_4, 1, 2)  # 積分結果と推定誤差を返す,積分範囲は1から2
print("積分結果:", result)
print("推定誤差:", error)
