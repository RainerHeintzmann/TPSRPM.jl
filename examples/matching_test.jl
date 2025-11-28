# make a simple example to test and visualize the point matching
using TPSRPM

X = [
    0.0 0.0
    0.0 1.0
    2 0
    2 1
]
α = 10.5 * pi/180;
rotM = [cos(α) sin(α)
-sin(α) cos(α)];

Y = (rotM * X')' .+ [3.3, 4.5]';

params = tps_rpm(X, X; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.1, iters_per_beta=3, verbose=true);
X_warped = TPSRPM.tps_rpm_apply(X, params, X)

params = tps_rpm(Y, X; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.1, iters_per_beta=3, verbose=true);

Y_warped = TPSRPM.tps_rpm_apply(Y, params, Y)

params = tps_rpm(Y, X; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.1, iters_per_beta=3, verbose=true, center=true);
