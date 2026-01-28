import torch
from math import pi
from scipy.constants import h, k, e
import matplotlib.pyplot as plt

# ===== device/dtype =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # GPUが使えるならGPU、無理ならCPU
DTYPE  = torch.float64              # 精度重視

print("=== 実行環境チェック ===")
print("CUDA が利用可能か:", torch.cuda.is_available())
print("使用する device:", device)
if torch.cuda.is_available():
    try:
        print("GPU 名称:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU 名称の取得でエラー:", repr(e))
print()


# ===== 定数を Tensor 化（device/dtypeを揃える）=====
hbar = torch.tensor(h/(2*pi), dtype=DTYPE, device=device)
kb   = torch.tensor(k,          dtype=DTYPE, device=device)
q    = torch.tensor(e,          dtype=DTYPE, device=device)

m_star_e = torch.tensor(1.4*9.11e-31, dtype=DTYPE, device=device)
m_star_h = torch.tensor(1.4*9.11e-31, dtype=DTYPE, device=device)
E_g      = torch.tensor(0.910022, dtype=DTYPE, device=device) * q
E_D      = torch.tensor(0.026,    dtype=DTYPE, device=device) * q
g_c      = torch.tensor(2.0,      dtype=DTYPE, device=device)

SQRT_PI  = torch.sqrt(torch.tensor(pi, dtype=DTYPE, device=device))

# ===== 状態密度など =====
def N_C(T): return 2.0 * ((m_star_e*kb*T)/(2*pi*hbar*hbar))**1.5
def N_V(T): return 2.0 * ((m_star_h*kb*T)/(2*pi*hbar*hbar))**1.5
def xi_g(T): return E_g/(kb*T)
def xi_D(T): return E_D/(kb*T)

# ===== 安定な 1/(1+exp(x))（torch.whereで分岐）=====
def inv1p_exp(x):
    x_clamped = torch.clamp(x, min=-40.0, max=40.0)
    return 1.0 / (1.0 + torch.exp(x_clamped))

# ===== F_{1/2}(xi) を台形則で =====
# 1) F_{1/2} を広い上限＆高分解能に
def fermi_dirac_half_torch(xi, x_max=800.0, n=4096):
    x = torch.linspace(0.0, float(x_max), n, dtype=DTYPE, device=device)
    t = x - xi
    denom = 1.0 + torch.exp(torch.clamp(t, min=-40.0, max=40.0))
    integrand = torch.sqrt(x) / denom
    return torch.trapz(integrand, x)

# 2) f_D: Python ifをやめてclampで
def f_D(xi_F, T):
    x = xi_D(T) - xi_F
    z = (1.0/g_c) * torch.exp(torch.clamp(x, min=-40.0, max=40.0))
    return 1.0 / (1.0 + z)

# 3) n,p: sqrt(pi) はテンソルで
SQRT_PI = torch.sqrt(torch.tensor(pi, dtype=torch.float64))
def n(xi_F, T):
    return (2.0/SQRT_PI) * N_C(T) * fermi_dirac_half_torch(xi_F)
def p(xi_F, T):
    return (2.0/SQRT_PI) * N_V(T) * fermi_dirac_half_torch(-xi_F - xi_g(T))

# 3) 目的関数
def Phi(xi_F, N_D, T):
    return n(xi_F, T) - N_D*(1.0 - f_D(xi_F, T)) - p(xi_F, T)

# ===== 誤差逆伝播法 =====
def solve_xiF_backprop(N_D, T, x0=0.0, lr=0.05, tol=1e-10, maxiter=2000):

    xi = torch.tensor(x0, dtype=DTYPE, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([xi], lr=lr)

    for _ in range(maxiter):
        optimizer.zero_grad()
        phi = Phi(xi, N_D, T)
        loss = phi**2
        if not torch.isfinite(loss):
            return torch.tensor(float('nan'), dtype=DTYPE, device=device)
        if loss < tol:
            break
        loss.backward()
        optimizer.step()

    return xi.detach()


# ===== 実行=====
N_D_values = [2.15e25, 4.64e25, 1e26, 2.15e26, 4.64e26]
T_range = torch.linspace(1300.0, 10.0, 100, dtype=DTYPE, device=device)

xi_F_vals = []
for N_D in N_D_values:
    Nd = torch.tensor(N_D, dtype=DTYPE, device=device)
    curve = []
    # 初回推定：真性近似 xi0 ≈ -xi_g(T)/2
    xi0 = float((-xi_g(T_range[0]) / 2.0).item())
    for T in T_range:
        xi = solve_xiF_backprop(Nd, T, x0=xi0)
        xi0 = float(xi) if torch.isfinite(xi) else xi0  # ウォームスタート
        curve.append(float(xi))
    xi_F_vals.append(curve)

# ===== 可視化 =====
plt.figure(figsize=(10,6))
for i, N_D in enumerate(N_D_values):
    plt.plot(T_range.cpu().numpy(), xi_F_vals[i], label=f'N_D={N_D:.2e}')
plt.xlabel('Temperature T [K]')
plt.ylabel('Fermi level ξ_F')
plt.grid(True); plt.legend(); plt.title('ξ_F vs T ')
plt.show()
