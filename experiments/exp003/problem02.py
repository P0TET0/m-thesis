import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み（x,y をTensorへ）
df = pd.read_csv("shukudai2.csv")

x_data = torch.tensor(df["x"].values, dtype=torch.float32)  # csvのx列をtorchのテンソルに変換
y_data = torch.tensor(df["y"].values, dtype=torch.float32)  # csvのy列をtorchのテンソルに変換
# print("x_data:", x_data)
# print("y_data:", y_data)


# train/val 分割
N = len(x_data)  # データの数
# print("データの数 N:", N)

train_size = int(0.8 * N)  # 訓練データのサイズ
# print("訓練データのサイズ:", train_size)


# データをシャッフルしてから訓練データとテストデータに分割
indices = torch.randperm(N)  # データのインデックスをランダムにシャッフル
train_indices = indices[:train_size]  # 訓練データのインデックス（先頭～train_size）
test_indices = indices[train_size:]  # テストデータのインデックス（train_size～末尾）
# print("訓練データのインデックス:", train_indices)
# print("テストデータのインデックス:", test_indices)


# シャッフルされたインデックスを使って訓練データとテストデータを作成
x_train = x_data[train_indices]  # 訓練データのx
y_train = y_data[train_indices]  # 訓練データのy
x_test = x_data[test_indices]  # テストデータのx
y_test = y_data[test_indices]  # テストデータのy
# print("x_train:", x_train)
# print("y_train:", y_train)
# print("x_test:", x_test)
# print("y_test:", y_test)


# パラメータ定義（a,b,c を学習する変数にする）
a = torch.randn(1, requires_grad=True)  # a をランダムに初期化して学習可能にする
b = torch.randn(1, requires_grad=True)  # b をランダムに初期化して学習可能にする
c = torch.randn(1, requires_grad=True)  # c をランダムに初期化して学習可能にする
print("初期パラメータ a:", a.item())
print("初期パラメータ b:", b.item())
print("初期パラメータ c:", c.item())


# 予測式と損失
y_hat = a * x_train**2 + b * x_train + c # 予測値
r = ((y_hat - y_train) ** 2) # 二乗誤差
MSE = torch.mean(r) # MSE（平均二乗誤差）
# print("予測値 y_hat:", y_hat)
# print("二乗誤差 r:", r)
# print("MSE:", MSE)


# 学習ループ
optimizer = torch.optim.Adam([a, b, c], lr=0.01)  # Adamオプティマイザを定義
num_epochs = 10000  # エポック数
train_losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 勾配をゼロにリセット
    y_hat = a * x_train**2 + b * x_train + c  # 予測値を計算
    MSE = torch.mean((y_hat - y_train) ** 2)  # MSEを計算
    MSE.backward()  # 勾配を計算
    RMSE = torch.sqrt(MSE)  # RMSEを計算
    optimizer.step()  # パラメータを更新
    train_losses.append(MSE.item())

    if (epoch + 1) % 100 == 0:  # 100エポックごとに進捗を表示
        print(f"Epoch [{epoch + 1}/{num_epochs}], MSE: {MSE.item():.10f}, RMSE: {RMSE.item():.10f}, a: {a.item():.4f}, b: {b.item():.4f}, c: {c.item():.4f}")


# テストデータで最終評価
with torch.no_grad():  # 勾配計算を無効にしてテストデータで予測
    y_test_hat = a * x_test**2 + b * x_test + c  # テストデータの予測値を計算
    test_mse = torch.mean((y_test_hat - y_test) ** 2)  # テストデータのMSEを計算
# print(f"Final train MSE: {MSE.item():.10f}")  # 最終的な訓練データのMSEを表示
# print(f"Final test  MSE: {test_mse.item():.10f}")  # 最終的なテストデータのMSEを表示
print(f"Learned params: a={a.item():.6f}, b={b.item():.6f}, c={c.item():.6f}")  # 学習されたパラメータを表示

# 誤差曲線（縦軸: 誤差, 横軸: エポック）を描画
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train MSE")
plt.xlabel("Epoch")
plt.ylabel("Error (MSE)")
plt.title("Training Error vs Epoch")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("problem02_loss_curve.png", dpi=150)
plt.show()
