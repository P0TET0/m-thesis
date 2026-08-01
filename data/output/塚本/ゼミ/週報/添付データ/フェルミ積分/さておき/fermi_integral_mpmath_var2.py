import mpmath
from scipy.constants import h, k, pi

# mpmathの精度を設定
mpmath.mp.dps = 50  # 50桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
n = mpmath.mpf(1.0e29)  # 電子密度 (電子/m^3)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
T = mpmath.mpf(300)  # 温度 (K)

# フェルミエネルギーの計算
# EF = (hbar**2 / (2 * m)) * (3 * mpmath.pi**2 * n)**(2/3)

# ξ_Fの計算
xi_F = 20

# フェルミ積分の関数定義
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# S = 3/2 および S = -1/2 のフェルミ積分の計算
fermi_integral_32 = fermi_integral(3/2, xi_F)
fermi_integral_m12 = fermi_integral(-1/2, xi_F)

print("ξ_F:", xi_F)

print("F_3/2(xi_F):", fermi_integral_32)
print("F_-1/2(xi_F):", fermi_integral_m12)




