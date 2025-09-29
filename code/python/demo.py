import numpy as np
import matplotlib.pyplot as plt

# 常数
mu0 = 4e-7 * np.pi   # H/m
eps0 = 8.854e-12     # F/m
c = 1 / np.sqrt(mu0 * eps0)  # 光速

# 参数
r = 0.1              # 板间距，单位 m
U0 = 10.0            # 最终电压，V
t0 = 1e-9            # 电压上升时间，s

# 计算电场变化率
def dE_dt(t):
    if 0 <= t <= t0:
        return U0 / (r * t0)  # 因为 E = U(t)/r, U(t) = U0 * t/t0
    else:
        return 0.0

# 磁场 B 与 h 的关系（平行板近似，h 为到中心线的垂直距离）
def B_field(h, t):
    # 从安培-麦克斯韦定律：2B L = mu0 * eps0 * (L * h) * dE/dt
    # 所以 B = (mu0 * eps0 * h / 2) * dE_dt(t)
    return (mu0 * eps0 * h / 2) * dE_dt(t)

# 矢量势 A：在对称情况下，B = dA/dh? 不对，应是 B_z = (1/h) d/dh (h A_phi) 在柱坐标
# 简化：假设均匀场变化，对称性得 A_phi 与 h 成正比？这里我们用 B = curl A，在平行板情形下，A 方向平行于电流方向（x方向），且随 z 变化？
# 实际上平行板电场变化产生的磁场是绕z轴的，A 是轴向的（柱坐标phi分量）。我们简化：假设 h 为到中心距离，且 h << 板尺寸，则 B 近似均匀？不对，B 随 h 线性增加。
# 更简单：直接用 B = dA/dh 的一维近似（如果 A 只有 z 分量且随 h 变化）——这不准确，但演示数值用。

# 我们改用：在对称轴上 h 处，A_phi(h) 满足 (1/h) d/dh (h A_phi) = B，解出 A_phi(h) = (B * h) / 2 （若 B 与 h 无关，但这里B与h有关）
# 实际上 B 与 h 成正比：B = k h，那么 (1/h) d/dh (h A_phi) = k h
# 积分：d/dh (h A_phi) = k h^2 => h A_phi = k h^3 / 3 => A_phi = k h^2 / 3
# 这里 k = (mu0 * eps0 / 2) * dE_dt(t)

def A_potential(h, t):
    k = (mu0 * eps0 / 2) * dE_dt(t)
    return k * h**2 / 3

# 示例计算
t_val = 0.5e-9  # 选择在上升中途
h_vals = np.linspace(0, 0.05, 100)  # h 从 0 到 5 cm
B_vals = [B_field(h, t_val) for h in h_vals]
A_vals = [A_potential(h, t_val) for h in h_vals]

# 绘图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(h_vals * 100, B_vals)
plt.xlabel('h (cm)')
plt.ylabel('B (T)')
plt.title('Magnetic field vs distance at t=0.5 ns')

plt.subplot(1, 2, 2)
plt.plot(h_vals * 100, A_vals)
plt.xlabel('h (cm)')
plt.ylabel('A (T·m)')
plt.title('Vector potential vs distance at t=0.5 ns')

plt.tight_layout()
plt.show()

# 输出某点的值
h_test = 0.02  # 2 cm
print(f"在 t={t_val:.2e} s, h={h_test*100:.1f} cm 处：")
print(f"电场变化率 dE/dt = {dE_dt(t_val):.2e} V/m/s")
print(f"磁场 B = {B_field(h_test, t_val):.2e} T")
print(f"矢量势 A = {A_potential(h_test, t_val):.2e} T·m")
