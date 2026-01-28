# 任意の半径の球体の体積を求めるプログラム

import math

def sphere_volume(radius):
    return (4/3) * math.pi * radius ** 3

if __name__ == "__main__":
    r = float(input("半径を入力してください: "))
    v = sphere_volume(r)
    print(f"半径 {r} の球体の体積は {v:.6f} です")
