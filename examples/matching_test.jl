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

## Now a larger example 

sz = (256,256)
X = 50 .+ 150 .*rand(100,2)

α = 1.5 * pi/180;

rotM = [cos(α) sin(α)
-sin(α) cos(α)];

E = 0 # 4 .*(rand(100,2) .- 0.5);

Y = (rotM * X')' .+ [3.3, 4.5]' .+ E;

params = tps_rpm(X, Y; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.1, iters_per_beta=3, verbose=true);
X_warped = TPSRPM.tps_rpm_apply(X, params, X)

res = visualize_result(sz, X,Y, params);
@vv res
colmap = 11 # random colors
set_colormap_no(colmap, 2)
set_colormap_no(colmap, 3)
set_colormap_no(colmap, 4)


@vt show_pos(sz, X)
@vt show_pos(sz, Y)
@vt show_pos(sz, X_warped)
