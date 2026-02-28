# make a simple example to test and visualize the point matching
using TPSRPM
using Statistics: mean

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

params = tps_rpm(Y, X; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.01, iters_per_beta=3, verbose=true);
Y_warped = TPSRPM.tps_rpm_apply(Y, params, Y)
ids = assigment_indices(X, Y, params; min_weight=0.1, max_outlier=1.0, mutual=false)

params = tps_rpm(Y, X; beta_sched=collect(0.5:0.5:6.0), cout=1e-2, λ=0.1, iters_per_beta=3, verbose=true, center=true);

ids = assigment_indices(X, Y, params; min_weight=0.1, max_outlier=1.0, mutual=false)

## Now a larger example 

sz = (256,256)
X = 50 .+ 150 .*rand(100,2)

α = 2.0 * pi/180; # a small rotation

rotM = [cos(α) sin(α)
-sin(α) cos(α)];
transl = [3.3, 4.5]

E = 0 # 1 .*(rand(100,2) .- 0.5);

Y = (rotM * (X' .+ transl))' .+ E;

# Test with NEW refine_affine option
params = tps_rpm(Y, X; beta_sched=collect(0.5:0.5:100.0), cout=1.0, λ=100000.0, iters_per_beta=3, center=true, verbose=false, refine_affine=true);

println("=== With refine_affine=true ===")
found_mat, found_trans = get_affine_part(params; verbose=true)

# NEW: Refined affine from hard assignment
println("\n=== Refined affine from hard assignment ===")
refined_mat, refined_trans, assign, n_matched = get_affine_part_refined(params, Y, X)
println("Refined matrix:")
println(refined_mat)
println("Refined translation: ", refined_trans)
println("Number of matched pairs: ", n_matched, " / ", size(Y, 1))

# Direct least-squares affine from known correspondences (for comparison)
# We know Y[i] corresponds to X[i], so we want to find A, t such that X ≈ A*Y + t
# Using centered coordinates: (X - μX) ≈ A * (Y - μY)
μX_direct = mean(X, dims=1)
μY_direct = mean(Y, dims=1)
Xc_direct = X .- μX_direct
Yc_direct = Y .- μY_direct
# Solve Xc = A * Yc in least-squares sense: A = Xc * Yc' * inv(Yc * Yc')
A_direct = (Xc_direct' * Yc_direct) * inv(Yc_direct' * Yc_direct)
# Translation: t = μX - A * μY
t_direct = vec(μX_direct) - A_direct * vec(μY_direct)
println("\n=== Direct LS from known correspondences ===")
println("A_direct:")
println(A_direct)
println("t_direct:")
println(t_direct)

# Expected values
# Y = rotM * X + t, so X = rotM' * (Y - t) since rotM is already R^T
# Expected transform: f(y) = rotM' * y - rotM' * t = R * y - R * t
R_expected = rotM'  # This is R (since rotM was R^T)
t_expected = -R_expected * transl
println("\n=== Expected vs Actual ===")
println("Expected matrix (R):")
println(R_expected)
println("\nActual matrix:")
println(found_mat)
println("\nExpected translation (-R*t):")
println(t_expected)
println("\nActual translation:")
println(found_trans)
println("\nMatrix error:")
println(found_mat - R_expected)
println("\nTranslation error:")
println([found_trans...] - t_expected)

println("\n=== Refined vs Expected ===")
println("Refined matrix error:")
println(refined_mat - R_expected)
println("Refined translation error:")
println([refined_trans...] - t_expected)

X_warped = TPSRPM.tps_rpm_apply(X, params, X)
ids = assigment_indices(X, Y, params; min_weight=0.1, max_outlier=1.0, mutual=false)

# res = visualize_result(sz, X,Y, params);
# @vv res
# colmap = 11 # random colors
# set_colormap_no(colmap, 2)
# set_colormap_no(colmap, 3)
# set_colormap_no(colmap, 4)
# 
# 
# @vt show_pos(sz, X)
# @vt show_pos(sz, Y)
# @vt show_pos(sz, X_warped)
