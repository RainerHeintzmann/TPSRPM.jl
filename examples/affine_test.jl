# Test affine transform recovery
using TPSRPM
using Statistics: mean
using LinearAlgebra

sz = (256,256)
X = 50 .+ 150 .*rand(100,2)

α = 2.0 * pi/180; # a small rotation

# Note: rotM is R^T (transpose of standard rotation)
rotM = [cos(α) sin(α)
       -sin(α) cos(α)];
transl = [3.3, 4.5]

Y = (rotM * (X'))' .+ transl';

println("Testing TPS-RPM affine recovery")
println("================================")
println("Applied transform: Y = R^T * X + t")
println("  Rotation angle: $(rad2deg(α))°")
println("  Translation: $transl")
println()

# Expected inverse: f(y) = R * y - R * t (since R^T^{-1} = R)
R_expected = rotM'  # This is R
t_expected = -R_expected * transl
println("Expected inverse transform: X = R * Y - R * t")
println("  Expected matrix (R):")
display(R_expected)
println("  Expected translation (-R*t): $t_expected")
println()

println("=== Comparing three methods ===")
println()

# Method 1: Standard soft-assignment based (no refinement)
println("1. Standard TPS-RPM (refine_affine=false):")
params1 = tps_rpm(Y, X; 
                  beta_sched=collect(0.5:0.5:50.0), 
                  cout=1.0, λ=100000.0, 
                  iters_per_beta=3, 
                  center=true, 
                  verbose=false,
                  refine_affine=false);
mat1, trans1 = get_affine_part(params1)
println("   Matrix error:      $(norm(mat1 - R_expected))")
println("   Translation error: $(norm([trans1...] - t_expected))")
println()

# Method 2: With refine_affine option (integrated)
println("2. TPS-RPM with refine_affine=true (integrated refinement):")
params2 = tps_rpm(Y, X; 
                  beta_sched=collect(0.5:0.5:50.0), 
                  cout=1.0, λ=100000.0, 
                  iters_per_beta=3, 
                  center=true, 
                  verbose=false,
                  refine_affine=true);
mat2, trans2 = get_affine_part(params2)
println("   Matrix error:      $(norm(mat2 - R_expected))")
println("   Translation error: $(norm([trans2...] - t_expected))")
println()

# Method 3: Post-hoc refinement using get_affine_part_refined
println("3. Post-hoc refinement (get_affine_part_refined on unrefined params):")
mat3, trans3, _, n_matched = get_affine_part_refined(params1, Y, X)
println("   Matrix error:      $(norm(mat3 - R_expected))")
println("   Translation error: $(norm([trans3...] - t_expected))")
println("   Matched pairs:     $n_matched / $(size(Y,1))")
println()

println("=== Summary ===")
println("Method 1 (soft only):        Matrix err = $(round(norm(mat1 - R_expected), sigdigits=3))")
println("Method 2 (refine_affine):    Matrix err = $(round(norm(mat2 - R_expected), sigdigits=3))")
println("Method 3 (post-hoc refined): Matrix err = $(round(norm(mat3 - R_expected), sigdigits=3))")
