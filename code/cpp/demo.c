#include <stdio.h>
#include <math.h>

// 物理常数
#define MU0 (4e-7 * M_PI)     // 真空磁导率 H/m
#define EPS0 8.854e-12        // 真空介电常数 F/m
#define C 299792458.0         // 光速 m/s

// 计算电场变化率
double dE_dt(double t, double U0, double r, double t0) {
    if (t >= 0 && t <= t0) {
        return U0 / (r * t0);  // E = U(t)/r, U(t) = U0 * t/t0
    } else {
        return 0.0;
    }
}

// 计算磁场B
double B_field(double h, double t, double U0, double r, double t0) {
    double dedt = dE_dt(t, U0, r, t0);
    return (MU0 * EPS0 * h / 2.0) * dedt;
}

// 计算矢量势A
double A_potential(double h, double t, double U0, double r, double t0) {
    double k = (MU0 * EPS0 / 2.0) * dE_dt(t, U0, r, t0);
    return k * h * h / 3.0;
}

int main() {
    // 参数设置
    double r = 0.1;           // 板间距 m
    double U0 = 10.0;         // 最终电压 V
    double t0 = 1e-9;         // 电压上升时间 s
    double t_val = 0.5e-9;    // 计算时刻 s
    
    printf("电磁场参数计算（平行板电容器模型）\n");
    printf("=====================================\n");
    printf("板间距 r = %.2f m\n", r);
    printf("最终电压 U0 = %.1f V\n", U0);
    printf("电压上升时间 t0 = %.1e s\n", t0);
    printf("计算时刻 t = %.1e s\n", t_val);
    printf("-------------------------------------\n");
    
    // 计算不同距离的A值
    printf("距离h(cm)\t磁场B(T)\t矢量势A(T·m)\n");
    printf("-------------------------------------\n");
    
    for (int i = 0; i <= 10; i++) {
        double h = i * 0.005;  // 从0到5cm，步长0.5cm
        double B = B_field(h, t_val, U0, r, t0);
        double A = A_potential(h, t_val, U0, r, t0);
        
        printf("%.1f\t\t%.2e\t%.2e\n", h * 100, B, A);
    }
    
    // 详细计算示例点
    double h_test = 0.02;  // 2cm
    double dedt = dE_dt(t_val, U0, r, t0);
    double B_test = B_field(h_test, t_val, U0, r, t0);
    double A_test = A_potential(h_test, t_val, U0, r, t0);
    
    printf("\n详细计算结果（h=2cm处）：\n");
    printf("电场变化率 dE/dt = %.2e V/m/s\n", dedt);
    printf("磁场强度 B = %.2e T\n", B_test);
    printf("矢量势 A = %.2e T·m\n", A_test);
    
    return 0;
}
