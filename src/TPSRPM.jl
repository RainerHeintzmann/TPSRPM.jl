module TPSRPM # Thin Plate Splin Robust Point Matching

using LinearAlgebra, Statistics
export vc2mat, tps_rpm_apply, tps_rpm, show_pos, make_grid_points, assigment_indices, visualize_result
export get_affine_part, get_hard_assignment, affine_from_correspondences, get_affine_part_refined

"""
    vc2mat(vec::AbstractVector) -> Matrix{Float64}

Convert a vector of tuples or CartesianIndices to an N×2 matrix for use with TPSRPM.

# Arguments
- `vec`: Vector of tuples or CartesianIndices (as obtained from `findlocalmaxima` etc.)

# Returns
- `Matrix{Float64}`: N×2 matrix where each row is [x, y] coordinates

# Example
```julia
positions = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
X = vc2mat(positions)
# X is now a 3×2 matrix
```
"""
function vc2mat(vec)
    res = zeros(length(vec), 2)
    for (v,n) in zip(vec, eachindex(vec))
        res[n,:] .= Tuple(v);
    end
    return res
end


# --- TPS pieces (same U, pairwise K, weighted solve/apply) ---
@inline function _tps_U(r2::Float64)
    r2 == 0.0 && return 0.0
    return r2 * log(r2)
end

function _pairwise_U(P::AbstractMatrix{Float64})
    N = size(P,1)
    K = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N
        xi, yi = P[i,1], P[i,2]
        for j in 1:N
            dx = xi - P[j,1]; dy = yi - P[j,2]
            K[i,j] = _tps_U(dx*dx + dy*dy)
        end
    end
    K
end

function _tps_solve(P::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}; λ=1e-3)
    # P: N×2 control points, Y: N×2 expected targets
    N = size(P,1)
    K = _pairwise_U(P)           # N×N
    Pm = [ones(N) P]             # N×3
    A = [K + λ*I  Pm; Pm'  zeros(3,3)]   # (N+3)×(N+3)
    # Solve independently for x and y outputs
    cx = A \ vcat(Y[:,1], zeros(3))
    cy = A \ vcat(Y[:,2], zeros(3))
    wx = cx[1:N]; ax = cx[N+1:end]  # [a0, ax, ay]
    wy = cy[1:N]; ay = cy[N+1:end]
    return wx, wy, ax, ay
end

# Solve TPS with fixed affine part - only find warping coefficients wx, wy
# Given: f(p) = a0 + A*p + Σ w_j U(||p - P_j||)
# With affine [a0, a1, a2] fixed, solve (K + λI) w = Y - P_affine * a
function _tps_solve_fixed_affine(P::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}, 
                                  ax::AbstractVector{Float64}, ay::AbstractVector{Float64}; λ=1e-3)
    N = size(P,1)
    K = _pairwise_U(P)           # N×N
    Pm = [ones(N) P]             # N×3 - design matrix for affine
    
    # Compute affine contribution at control points
    affine_x = Pm * ax  # N vector
    affine_y = Pm * ay  # N vector
    
    # Residual targets after subtracting affine
    residual_x = Y[:,1] .- affine_x
    residual_y = Y[:,2] .- affine_y
    
    # Solve (K + λI) w = residual
    # With constraint: P' * w = 0 (for valid TPS)
    # Use regularized solve: minimize ||w||² + λ⁻¹||(K+λI)w - residual||²
    # Simpler: directly solve with regularization
    KλI = K + λ * I
    wx = KλI \ residual_x
    wy = KλI \ residual_y
    
    return wx, wy
end

function _tps_apply(P::AbstractMatrix{Float64}, wx::AbstractVector{Float64}, wy::AbstractVector{Float64}, ax::AbstractVector{Float64}, ay::AbstractVector{Float64}, Q::AbstractMatrix{Float64})
    N = size(P,1); M = size(Q,1)
    out = similar(P, (M,2)) #  Matrix{Float64}(undef, M, 2)
    @inbounds for i in 1:M
        x, y = Q[i,1], Q[i,2]
        sx = ax[1] + ax[2]*x + ax[3]*y
        sy = ay[1] + ay[2]*x + ay[3]*y
        for j in 1:N
            dx = x - P[j,1]; dy = y - P[j,2]
            u = _tps_U(dx*dx + dy*dy)
            sx += wx[j]*u
            sy += wy[j]*u
        end
        out[i,1] = sx
        out[i,2] = sy
    end
    out
end

# --- Soft assignment with outliers (Sinkhorn-like normalization) ---
"""
    soft_assign(X, Y; beta, cout)

Compute soft correspondences A (N×(M+1)) between X[N×2] and Y[M×2]
with one dummy column for outliers, cost of matching is squared distance,
outlier column has constant cost cout. beta is inverse temperature.
Rows and (non-outlier) columns are approximately stochastic (Sinkhorn 5 iters).
Returns A, where last column is outlier probability per X row.
"""
function soft_assign(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}; beta::Float64, cout::Float64)
    N, M = size(X,1), size(Y,1)
    # Costs: D_ij = ||X_i - Y_j||^2, D_i,out = cout
    D = similar(X, (N,M+1)) # Matrix{Float64}(undef, N, M+1)
    @inbounds for i in 1:N
        xi, yi = X[i,1], X[i,2]
        for j in 1:M
            dx = xi - Y[j,1]; dy = yi - Y[j,2]
            D[i,j] = dx*dx + dy*dy
        end
        D[i,M+1] = cout
    end
    # Gibbs kernel
    A = exp.(-beta .* D)
    # Sinkhorn row/column normalization (ignore outlier column in column-normalization)
    for _ in 1:5
        # Row normalize
        A .= A ./ sum(A, dims=2)
        # Column normalize (only 1..M)
        colsum = sum(view(A, :, 1:M), dims=1)
        @inbounds for j in 1:M
            cj = colsum[1,j]; cj == 0 && continue
            A[:,j] ./= cj
        end
    end
    return A
end

"""
    soft_assign_bidir(X, Y; beta, cout_src, cout_dst)

Compute soft correspondences A ∈ ℝ^{(N+1)×(M+1)} with:
- last column = source outlier (unmatched source i), penalty cout_src
- last row    = destination outlier (unmatched dest j), penalty cout_dst

We apply Sinkhorn-like normalization over the real rows/cols:
- normalize rows 1..N over all columns 1..M and the source-outlier column (M+1)
- normalize cols 1..M over all rows 1..N and the dest-outlier row (N+1)
The outlier row/column allow unmatched mass to flow.

Returns A and views:
- A_src = A[1:N, 1:M]         real source-to-dest soft matches
- p_src_out = A[1:N, M+1]     source outlier probabilities
- p_dst_out = A[N+1, 1:M]     destination outlier probabilities
"""
function soft_assign_bidir(X::AbstractMatrix, Y::AbstractMatrix;
                           beta::Float64, cout_src::Float64, cout_dst::Float64)
    N, M = size(X,1), size(Y,1)
    # Build cost matrix with dummy row/col
    D = Matrix{Float64}(undef, N+1, M+1)
    @inbounds for i in 1:N
        xi, yi = X[i,1], X[i,2]
        for j in 1:M
            dx = xi - Y[j,1]; dy = yi - Y[j,2]
            D[i,j] = dx*dx + dy*dy
        end
    end
    
    # Only scale distances if there's risk of numerical underflow
    # exp(-700) ≈ 0 (underflow threshold)
    Dreal = @view D[1:N, 1:M]
    max_dist = maximum(Dreal)
    scale_factor = 1.0
    if beta * max_dist > 500  # conservative threshold
        scale_factor = 500 / (beta * max_dist)
        Dreal .*= scale_factor
    end
    
    # Set outlier costs (scaled consistently if distances were scaled)
    for i in 1:N
        D[i,M+1] = cout_src * scale_factor
    end
    for j in 1:M
        D[N+1,j] = cout_dst * scale_factor
    end
    D[N+1,M+1] = 0.0                # outlier↔outlier (unused but finite)

    # Gibbs kernel
    A = exp.(-beta .* D)

    # Sinkhorn-like normalization with outlier row/column
    for _ in 1:5
        # Row normalize (all real source rows including src-outlier col)
        rowsum = sum(A, dims=2)
        @inbounds for i in 1:N
            s = rowsum[i]; s == 0 && continue
            A[i, :] ./= s
        end
        # Column normalize (all real dest cols including dst-outlier row)
        colsum = sum(A, dims=1)
        @inbounds for j in 1:M
            s = colsum[j]; s == 0 && continue
            A[:, j] ./= s
        end
        # Optional: normalize the outlier row/col lightly to keep scales sane
        # (leave A[N+1, :] and A[:, M+1] unnormalized or clamp if needed)
    end

    return A, @view(A[1:N, 1:M]), @view(A[1:N, M+1]), @view(A[N+1, 1:M])
end


"""
    get_affine_part(params; verbose=false) -> (matrix, trans)

Extract the affine transformation (2×2 matrix + translation) from TPS-RPM results.
The parameters are transformed from scaled/centered space back to original coordinates.

# Arguments
- `params`: NamedTuple result from `tps_rpm`
- `verbose::Bool=false`: if true, print debugging information

# Returns
- `matrix`: 2×2 linear transformation matrix
- `trans`: (tx, ty) translation tuple

# Example
```julia
params = tps_rpm(X, Y; λ=1.0)
matrix, trans = get_affine_part(params)
# Apply: Y_approx = matrix * X[i,:] + [trans[1], trans[2]]
```
"""
function get_affine_part(params; verbose=false)
    # Linear part (2x2 matrix) - same in both coordinate systems
    matrix = [params.ax[2] params.ax[3];
              params.ay[2] params.ay[3]]

    # Translation: t = σX * a_s - A * μX + μY
    σX = haskey(params, :σX) ? params.σX : 1.0
    μX = haskey(params, :μX) ? params.μX : zeros(2)
    μY = haskey(params, :μY) ? params.μY : zeros(2)
    center = haskey(params, :center) ? params.center : false
    
    a_s = [params.ax[1], params.ay[1]]
    if verbose
        println("get_affine_part debug:")
        println("  a_s (scaled translation) = ", a_s)
        println("  matrix = ", matrix)
        println("  σX = ", σX)
        println("  μX = ", μX)
        println("  μY = ", μY)
        println("  center = ", center)
        println("  A*μX = ", matrix * μX)
        println("  σX*a_s = ", σX .* a_s)
        println("  σX*a_s - A*μX = ", σX .* a_s .- matrix * μX)
    end
    if center || haskey(params, :σX)
        trans_vec = σX .* a_s .- matrix * μX .+ μY
        trans = (trans_vec[1], trans_vec[2])
    else
        trans = (params.ax[1], params.ay[1])
    end

    return matrix, trans
end


"""
    get_hard_assignment(params; method=:greedy, min_weight=0.0, max_outlier=1.0, mutual=false) -> Vector{Int}

Extract 1-to-1 hard assignment from the soft assignment matrix.

# Arguments
- `params`: NamedTuple result from `tps_rpm` (must contain `A`)
- `method::Symbol=:greedy`: assignment algorithm
  - `:greedy` - Iteratively pick best remaining match (fast, not globally optimal)
  - `:hungarian` - True Hungarian algorithm (not yet implemented)
- `min_weight::Float64=0.0`: minimum assignment weight to accept
- `max_outlier::Float64=1.0`: maximum outlier probability (1.0 = no filtering)
- `mutual::Bool=false`: if true, only keep mutual best matches

# Returns
- `assign::Vector{Int}`: length N, where `assign[i] = j` means source i matches dest j,
  `assign[i] = 0` means unmatched

# Example
```julia
params = tps_rpm(X, Y; λ=1.0)
assign = get_hard_assignment(params; min_weight=0.1, mutual=true)
matched = findall(>(0), assign)  # indices of matched sources
```
"""
function get_hard_assignment(params; method::Symbol=:greedy,
                             min_weight::Float64=0.0,
                             max_outlier::Float64=1.0,
                             mutual::Bool=false)
    A = params.A
    N = size(A, 1) - 1  # Exclude outlier row
    M = size(A, 2) - 1  # Exclude outlier column
    
    Areal = @view A[1:N, 1:M]
    p_src_out = @view A[1:N, M+1]
    p_dst_out = @view A[N+1, 1:M]
    
    if method == :greedy
        if min_weight > 0.0 || max_outlier < 1.0 || mutual
            return _filtered_greedy_assignment(Areal, p_src_out, p_dst_out;
                                               min_weight=min_weight,
                                               max_outlier=max_outlier,
                                               mutual=mutual)
        else
            return _greedy_assignment(Areal)
        end
    elseif method == :hungarian
        error("Hungarian method not yet implemented. Use method=:greedy or install Hungarian.jl")
    else
        error("Unknown method: $method. Use :greedy or :hungarian")
    end
end

# Greedy 1-to-1 assignment: iteratively pick the highest weight unassigned pair
function _greedy_assignment(Areal::AbstractMatrix)
    N, M = size(Areal)
    assign = zeros(Int, N)
    used_dst = falses(M)
    
    # Create list of all (i, j, weight) and sort by weight descending
    pairs = [(i, j, Areal[i, j]) for i in 1:N for j in 1:M]
    sort!(pairs, by=x -> -x[3])  # Sort by weight descending
    
    matched_sources = 0
    for (i, j, w) in pairs
        if assign[i] == 0 && !used_dst[j] && w > 0
            assign[i] = j
            used_dst[j] = true
            matched_sources += 1
            if matched_sources == min(N, M)
                break  # All possible matches made
            end
        end
    end
    
    return assign
end

# Filtered greedy assignment with min_weight, max_outlier, and mutual constraints
function _filtered_greedy_assignment(Areal::AbstractMatrix, 
                                      p_src_out::AbstractVector,
                                      p_dst_out::AbstractVector;
                                      min_weight::Float64=0.0,
                                      max_outlier::Float64=1.0,
                                      mutual::Bool=false)
    N, M = size(Areal)
    assign = zeros(Int, N)
    used_dst = falses(M)
    
    # Precompute column-best indices for mutual check
    col_best = Vector{Int}(undef, M)
    @inbounds for j in 1:M
        _, ib = findmax(@view Areal[:, j])
        col_best[j] = ib
    end
    
    # Create list of all (i, j, weight) and sort by weight descending
    pairs = [(i, j, Areal[i, j]) for i in 1:N for j in 1:M]
    sort!(pairs, by=x -> -x[3])  # Sort by weight descending
    
    matched_sources = 0
    for (i, j, w) in pairs
        if assign[i] == 0 && !used_dst[j]
            # Apply filtering criteria
            passes_weight = w >= min_weight
            passes_src_outlier = p_src_out[i] <= max_outlier
            passes_dst_outlier = p_dst_out[j] <= max_outlier
            passes_mutual = !mutual || col_best[j] == i
            
            if passes_weight && passes_src_outlier && passes_dst_outlier && passes_mutual
                assign[i] = j
                used_dst[j] = true
                matched_sources += 1
                if matched_sources == min(N, M)
                    break  # All possible matches made
                end
            end
        end
    end
    
    return assign
end


"""
    affine_from_correspondences(src, dst, assign) -> (matrix, trans, n_matched)

Compute optimal affine transformation from point correspondences using least-squares.

Finds A, t such that `dst[assign[i]] ≈ A * src[i] + t` for matched pairs.

# Arguments
- `src::AbstractMatrix`: N×2 source point set
- `dst::AbstractMatrix`: M×2 destination point set  
- `assign::Vector{Int}`: assignment vector where `assign[i] = j` means src[i] ↔ dst[j]

# Returns
- `matrix`: 2×2 linear transformation matrix (maps src → dst)
- `trans`: (tx, ty) translation tuple
- `n_matched`: number of matched pairs used (must be ≥ 3)

# Example
```julia
assign = [2, 0, 1, 3]  # src[1]↔dst[2], src[3]↔dst[1], src[4]↔dst[3]
matrix, trans, n = affine_from_correspondences(src, dst, assign)
# n = 3 (three matched pairs)
```
"""
function affine_from_correspondences(src::AbstractMatrix, dst::AbstractMatrix, assign::Vector{Int})
    # Extract matched pairs
    matched_idx = findall(a -> a > 0, assign)
    n_matched = length(matched_idx)
    
    if n_matched < 3
        error("Need at least 3 matched pairs for affine estimation, got $n_matched")
    end
    
    # Build matched source and destination point sets
    src_m = src[matched_idx, :]           # N_matched × 2 (source points)
    dst_m = dst[assign[matched_idx], :]   # N_matched × 2 (corresponding destination points)
    
    # Compute centroids
    μ_src = vec(mean(src_m, dims=1))
    μ_dst = vec(mean(dst_m, dims=1))
    
    # Center the points
    src_c = src_m .- μ_src'
    dst_c = dst_m .- μ_dst'
    
    # Solve for A in: dst_c ≈ src_c * A'  (least squares, row form)
    # This is equivalent to: dst = A * src (column form)
    # Normal equations give: A' = (src_c' * src_c) \ (src_c' * dst_c)
    StS = src_c' * src_c
    StD = src_c' * dst_c
    A = (StS \ StD)'
    
    # Translation: t = μ_dst - A * μ_src
    t = μ_dst - A * μ_src
    
    return A, (t[1], t[2]), n_matched
end


"""
    get_affine_part_refined(params, src, dst; method=:greedy)

Extract hard 1-to-1 correspondences from the soft assignment matrix and 
recompute the affine transformation using direct least-squares.

This gives more accurate affine parameters than `get_affine_part` when the
soft assignment is slightly diffuse (which happens with small rotations or
nearby points).

Arguments:
- `params`: TPS-RPM result from `tps_rpm`
- `src`: Source point set (the first argument to `tps_rpm`)  
- `dst`: Destination point set (the second argument to `tps_rpm`)
- `method`: Assignment method (:greedy or :hungarian)
- `min_weight`: minimum assignment weight to accept (default 0.0)
- `max_outlier`: maximum outlier probability (default 1.0 = no filtering)
- `mutual`: if true, only keep mutual best matches

Returns:
- `matrix`: 2×2 affine matrix (maps src → dst)
- `trans`: (tx, ty) translation (maps src → dst)
- `assign`: hard assignment vector (assign[i] = j means src[i] matches dst[j])
- `n_matched`: number of matched pairs

Example:
```julia
params = tps_rpm(Y, X; ...)  # Y=source, X=destination
matrix, trans, assign, n = get_affine_part_refined(params, Y, X)
# Now: X[assign[i]] ≈ matrix * Y[i] + trans for matched pairs
```
"""
function get_affine_part_refined(params, src::AbstractMatrix, dst::AbstractMatrix; 
                                  method::Symbol=:greedy,
                                  min_weight::Float64=0.0,
                                  max_outlier::Float64=1.0,
                                  mutual::Bool=false)
    # Get hard assignment from soft matrix
    assign = get_hard_assignment(params; method=method, 
                                 min_weight=min_weight,
                                 max_outlier=max_outlier,
                                 mutual=mutual)
    
    # Compute affine from matched correspondences
    matrix, trans, n_matched = affine_from_correspondences(src, dst, assign)
    
    return matrix, trans, assign, n_matched
end


# --- diagnostics helpers ---

# Pairwise squared distances between two point sets (N×2, M×2)
function _distsq(X::AbstractMatrix, Y::AbstractMatrix)
    N, M = size(X,1), size(Y,1)
    D = Matrix{Float64}(undef, N, M)
    @inbounds for i in 1:N
        xi, yi = X[i,1], X[i,2]
        for j in 1:M
            dx = xi - Y[j,1]; dy = yi - Y[j,2]
            D[i,j] = dx*dx + dy*dy
        end
    end
    D
end

# Compute energy components for reporting
function _energy_components(; X::AbstractMatrix,
                             Xw::AbstractMatrix,
                             Y::AbstractMatrix,
                             Afull::AbstractMatrix,
                             Areal::AbstractMatrix,
                             p_src_out::AbstractVector,
                             p_dst_out::AbstractVector,
                             wx::AbstractVector,
                             wy::AbstractVector,
                             λ,
                             cout_src::Float64,
                             cout_dst::Float64)
    # Data term
    D2 = _distsq(Xw, Y)
    E_data = sum(Areal .* D2)

    # Outlier penalties
    E_src_out = cout_src * sum(p_src_out)
    E_dst_out = cout_dst * sum(p_dst_out)

    # TPS regularization terms
    KX = _pairwise_U(X)
    E_bend_K = dot(wx, KX * wx) + dot(wy, KX * wy)
    E_bend_I = λ * (dot(wx, wx) + dot(wy, wy))

    # Entropy term (informative only)
    ϵ = eps(Float64)
    S = 0.0
    @inbounds for a in Afull
        a > 0 && (S += a * log(a + ϵ))
    end
    # β used outside (reported separately); keep raw S here
    return (; E_data, E_src_out, E_dst_out, E_bend_K, E_bend_I, S)
end

# --- TPS-RPM main loop ---
"""
    tps_rpm(X, Y; beta_sched, cout, λ, iters_per_beta, verbose, center, init, 
            refine_affine, refine_min_weight, refine_max_outlier, refine_mutual) -> NamedTuple

Compute thin-plate-spline (TPS) robust point matching (RPM) between source points X
and destination points Y, allowing for outliers on both sides.

# Arguments
- `X::AbstractMatrix`: N×2 source (non-moving) point set
- `Y::AbstractMatrix`: M×2 destination (moving) point set
- `beta_sched=collect(0.5:0.5:6.0)`: annealing schedule of inverse temperatures (increasing)
- `cout::Float64=1.0`: outlier cost (used for both source and dest unless overridden)
- `cout_src::Float64=nothing`: outlier cost for unmatched sources (defaults to `cout`)
- `cout_dst::Float64=nothing`: outlier cost for unmatched destinations (defaults to `cout`)
- `λ::Float64=20.0`: TPS bending regularization. Large (1e4) for affine, small (0.1-100) for warping
- `iters_per_beta::Int=3`: EM iterations per β value
- `verbose::Bool=false`: print energy decomposition per iteration
- `center::Bool=false`: subtract centroids before optimization (translation-invariant)
- `init::Symbol=:nothing`: initialization (`:centroid`, `:auto`, `:nothing`)
- `refine_affine::Bool=false`: recompute affine from hard assignment after EM
- `refine_min_weight::Float64=0.0`: min assignment weight for refinement (0-1)
- `refine_max_outlier::Float64=1.0`: max outlier probability for refinement (0-1)
- `refine_mutual::Bool=false`: require mutual best match for refinement

# Returns
NamedTuple with fields:
- `wx, wy`: N-vectors of TPS warp coefficients
- `ax, ay`: 3-vectors [a0, a1, a2] for affine: out = a0 + a1*x + a2*y
- `A`: (N+1)×(M+1) soft assignment matrix (last row/col are outliers)
- `Areal`: N×M view of real assignments (excludes outlier row/col)
- `p_src_out, p_dst_out`: outlier probabilities
- `diag`: vector of per-iteration diagnostics
- `Xctl, μX, μY, σX, center`: coordinate transform info for `tps_rpm_apply`

# Example
```julia
using TPSRPM

# Create test data: Y is rotated/translated version of X
X = rand(20, 2) .* 100
θ = deg2rad(5)  # 5° rotation
R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
t = [10.0, -5.0]
Y = (R * X')' .+ t'

# Run TPS-RPM
params = tps_rpm(X, Y; λ=1e4, refine_affine=true, verbose=true)

# Get assignment and affine transform
assign = assigment_indices(X, Y, params)
matrix, trans = get_affine_part(params)

# Apply warp to new points
X_warped = tps_rpm_apply(X, params, X)
```

Note: Algorithm is asymmetric. Sources (X) define the TPS; typically use smaller set as source.
"""
function tps_rpm(X::AbstractMatrix, Y::AbstractMatrix; beta_sched=collect(0.5:0.5:6.0),
                 cout::Float64=1.0, λ::Float64=20.0, iters_per_beta::Int=3, verbose=false,
                 cout_src=nothing, cout_dst=nothing,
                 center::Bool=false, init::Symbol=:nothing,
                 refine_affine::Bool=false,
                 refine_min_weight::Float64=0.0,
                 refine_max_outlier::Float64=1.0,
                 refine_mutual::Bool=false)

    # Separate penalties (defaults to cout)
    cout_src_val = cout_src === nothing ? cout : cout_src
    cout_dst_val = cout_dst === nothing ? cout : cout_dst

    # Optionally center (zero-mean) to make the path translation-invariant
    μX = vec(mean(X, dims=1))
    μY = vec(mean(Y, dims=1))
    Xc = center ? (X .- (μX')) : X
    Yc = center ? (Y .- (μY')) : Y

    # Scale coordinates to make λ interpretable (K values become O(1))
    # Scale by characteristic spread of control points
    σX = max(std(Xc[:,1]), std(Xc[:,2]), 1.0)
    Xs = Xc ./ σX
    Ys = Yc ./ σX  # Use same scale for both

    N, M = size(Xs,1), size(Ys,1)

    # Check for near-identity case (X ≈ Y with same size)
    # Return identity transform directly to avoid numerical drift
    if N == M && maximum(abs.(Xs .- Ys)) < 1e-10
        wx = zeros(N); wy = zeros(N)
        ax = [0.0, 1.0, 0.0]
        ay = [0.0, 0.0, 1.0]
        # Create identity matching matrix
        A = zeros(N+1, M+1)
        for i in 1:N
            A[i, i] = 1.0
        end
        Areal = @view A[1:N, 1:M]
        p_src_out = @view A[1:N, M+1]
        p_dst_out = @view A[N+1, 1:M]
        diag = Vector{NamedTuple}()
        return (; wx, wy, ax, ay, A, Areal, p_src_out, p_dst_out, diag, 
                Xctl=Xs, μX, μY, σX, center)
    end

    # Initialize warp (affine can encode pure translation)
    wx = zeros(N); wy = zeros(N)
    ax = [0.0, 1.0, 0.0]  # [a0x, axx, axy] in scaled coords
    ay = [0.0, 0.0, 1.0]  # [a0y, ayx, ayy] in scaled coords
    if !center && (init == :centroid || init == :auto)
        Δ = vec(mean(Y, dims=1) .- mean(X, dims=1)) ./ σX
        ax[1] = Δ[1]; ay[1] = Δ[2]
    end

    # diagnostics log
    diag = Vector{NamedTuple}()

    for (kβ, β) in pairs(beta_sched)
        for n in 1:iters_per_beta
            # E-step: warp X, then bidirectional soft assignment
            Xw = _tps_apply(Xs, wx, wy, ax, ay, Xs)
            Afull, Areal, p_src_out, p_dst_out = soft_assign_bidir(Xw, Ys; beta=β,
                                                                   cout_src=cout_src_val,
                                                                   cout_dst=cout_dst_val)

            # Expected targets for each source (ignore source-outlier mass)
            rowmass = sum(Areal, dims=2)
            EY = similar(Xs, (N,2))
            @inbounds for i in 1:N
                m = rowmass[i]
                if m <= eps()
                    EY[i,1] = Xs[i,1]; EY[i,2] = Xs[i,2]
                else
                    s1 = 0.0; s2 = 0.0
                    @inbounds for j in 1:M
                        w = Areal[i,j]
                        s1 += w * Ys[j,1]; s2 += w * Ys[j,2]
                    end
                    EY[i,1] = s1 / m; EY[i,2] = s2 / m
                end
            end

            # M-step: fit TPS Xs -> EY (in scaled coordinates)
            wx, wy, ax, ay = _tps_solve(Xs, EY; λ=λ)

            # Diagnostics (energy decomposition)
            Xw_new = _tps_apply(Xs, wx, wy, ax, ay, Xs)
            comps = _energy_components(; X=Xs, Xw=Xw_new, Y=Ys, Afull=Afull, Areal=Areal,
                                        p_src_out=p_src_out, p_dst_out=p_dst_out,
                                        wx=wx, wy=wy, λ=λ, cout_src=cout_src_val, cout_dst=cout_dst_val)
            E_total = comps.E_data + comps.E_src_out + comps.E_dst_out + comps.E_bend_K + comps.E_bend_I
            push!(diag, (beta=β, step=(kβ, n), E_total=E_total, comps...))

            if verbose
                safe = E_total == 0 ? 1.0 : E_total
                pd = 100 * comps.E_data / safe
                ps = 100 * comps.E_src_out / safe
                pt = 100 * comps.E_dst_out / safe
                pk = 100 * comps.E_bend_K / safe
                pi = 100 * comps.E_bend_I / safe
                println("β=$(β) iter $(n)/$(iters_per_beta): E_total=$(round(E_total, sigdigits=5)) | ",
                        "data=$(round(comps.E_data, sigdigits=4)) ($(round(pd, digits=1))%), ",
                        "src_out=$(round(comps.E_src_out, sigdigits=4)) ($(round(ps, digits=1))%), ",
                        "dst_out=$(round(comps.E_dst_out, sigdigits=4)) ($(round(pt, digits=1))%), ",
                        "bendK=$(round(comps.E_bend_K, sigdigits=4)) ($(round(pk, digits=1))%), ",
                        "bendI=$(round(comps.E_bend_I, sigdigits=4)) ($(round(pi, digits=1))%)")
            end
        end
    end

    # Final A at last β (scaled frame)
    Xw_last = _tps_apply(Xs, wx, wy, ax, ay, Xs)
    A, Areal, p_src_out, p_dst_out = soft_assign_bidir(Xw_last, Ys;
                                                       beta=last(beta_sched),
                                                       cout_src=cout_src_val, cout_dst=cout_dst_val)

    # Optional: refine affine using hard assignment and recompute TPS
    if refine_affine
        # Get filtered hard 1-to-1 assignment from soft matrix
        hard_assign = _filtered_greedy_assignment(Areal, p_src_out, p_dst_out;
                                                   min_weight=refine_min_weight,
                                                   max_outlier=refine_max_outlier,
                                                   mutual=refine_mutual)
        n_matched = count(>(0), hard_assign)
        
        if n_matched >= 3
            if verbose
                println("Refining affine with $n_matched hard-matched pairs (filtered from $(N) sources)...")
            end
            
            # Extract matched pairs in scaled coordinates
            matched_idx = findall(>(0), hard_assign)
            src_matched = Xs[matched_idx, :]           # source points
            dst_matched = Ys[hard_assign[matched_idx], :]  # corresponding destinations
            
            # Compute optimal affine in scaled coordinates: dst ≈ A * src + t
            μ_src = vec(mean(src_matched, dims=1))
            μ_dst = vec(mean(dst_matched, dims=1))
            src_c = src_matched .- μ_src'
            dst_c = dst_matched .- μ_dst'
            
            # Solve: dst_c ≈ src_c * A'  =>  A' = (src_c' * src_c) \ (src_c' * dst_c)
            StS = src_c' * src_c
            StD = src_c' * dst_c
            A_refined = (StS \ StD)'  # 2×2 matrix
            t_refined = μ_dst - A_refined * μ_src  # 2-vector
            
            # Update affine coefficients: [a0, a1, a2] for each dimension
            ax = [t_refined[1], A_refined[1,1], A_refined[1,2]]
            ay = [t_refined[2], A_refined[2,1], A_refined[2,2]]
            
            # Recompute TPS warping coefficients with fixed refined affine
            # Use weighted targets from soft assignment (as before)
            rowmass = sum(Areal, dims=2)
            EY = similar(Xs, (N,2))
            @inbounds for i in 1:N
                m = rowmass[i]
                if m <= eps()
                    EY[i,1] = Xs[i,1]; EY[i,2] = Xs[i,2]
                else
                    s1 = 0.0; s2 = 0.0
                    @inbounds for j in 1:M
                        w = Areal[i,j]
                        s1 += w * Ys[j,1]; s2 += w * Ys[j,2]
                    end
                    EY[i,1] = s1 / m; EY[i,2] = s2 / m
                end
            end
            
            # Solve for warping coefficients with fixed affine
            wx, wy = _tps_solve_fixed_affine(Xs, EY, ax, ay; λ=λ)
            
            if verbose
                println("Affine refinement complete.")
            end
        else
            if verbose
                println("Warning: Only $n_matched matched pairs, need ≥3 for affine refinement. Skipping.")
            end
        end
    end
    
    # Map affine constants back to original coordinates if centered:
    # f_orig(p) = a0' + A p + Σ w U(||p - X_j||), with a0' = a0 - A μX + μY
    # Store scaled control points and coordinate transform info for tps_rpm_apply
    # Parameters (wx, wy, ax, ay) stay in scaled space; tps_rpm_apply handles conversion
    return (; wx, wy, ax, ay, A, Areal, p_src_out, p_dst_out, diag, 
            Xctl=Xs, μX, μY, σX, center)
end


"""
    show_pos(sz, posmat, idxpos=1) -> Matrix{Float64}

Create an image with positions marked as dots, labeled by index values.

# Arguments
- `sz::Tuple{Int,Int}`: output image size (height, width)
- `posmat::AbstractMatrix`: N×2 matrix of (x, y) positions
- `idxpos=1`: either a scalar (all dots same value) or Vector{Int} of per-point labels.
  Values are stored as `idxpos[i] + 1`, so unassigned (0) becomes 1.

# Returns
- `Matrix{Float64}`: image of size `sz` with dots at positions

# Example
```julia
positions = [50.0 100.0; 150.0 200.0]  # 2 points
img = show_pos((256, 256), positions)  # all dots = 2
img = show_pos((256, 256), positions, [0, 5])  # dots = 1, 6
```
"""
function show_pos(sz, posmat, idxpos=1)
    res = zeros(sz)
    n = 1
    idxval = 0;
    for p in eachslice(posmat, dims=1)
        pos = round.(Int, p)
        if isa(idxpos, AbstractArray)
            idxval = idxpos[n];
        else
            idxval = idxpos;
        end
        if (all(pos.>0) && all(pos .<= sz))
            res[pos...] = idxval+1;
        end
        n += 1;
    end
    return res
end


"""
    tps_rpm_apply(X::AbstractMatrix, params::NamedTuple, Q::AbstractMatrix) -> Matrix{Float64}

Apply the TPS warp to arbitrary query points.

# Arguments
- `X::AbstractMatrix`: source control points (ignored if `params` contains `Xctl`)
- `params::NamedTuple`: TPS parameters from `tps_rpm` (contains `wx`, `wy`, `ax`, `ay`, etc.)
- `Q::AbstractMatrix`: K×2 matrix of query points to transform

# Returns
- `Matrix{Float64}`: K×2 matrix of warped positions

# Example
```julia
params = tps_rpm(X, Y; λ=1.0)
X_warped = tps_rpm_apply(X, params, X)  # warp source points
grid = make_grid_points(X, (512, 512); gridstep=20)
grid_warped = tps_rpm_apply(X, params, grid)  # warp a grid
```
"""
function tps_rpm_apply(X::AbstractMatrix, params::NamedTuple, Q::AbstractMatrix)
    # Use stored control points (in scaled space)
    Xctl = haskey(params, :Xctl) ? params.Xctl : X
    
    # Transform query points to scaled-centered space
    μX = haskey(params, :μX) ? params.μX : zeros(2)
    μY = haskey(params, :μY) ? params.μY : zeros(2)
    σX = haskey(params, :σX) ? params.σX : 1.0
    center = haskey(params, :center) ? params.center : false
    
    if center || haskey(params, :σX)
        Qs = (Q .- μX') ./ σX
    else
        Qs = Q
    end
    
    # Apply TPS in scaled space
    result_s = _tps_apply(Xctl, params.wx, params.wy, params.ax, params.ay, Qs)
    
    # Transform result back to original coordinates
    if center || haskey(params, :σX)
        result = σX .* result_s .+ μY'
    else
        result = result_s
    end
    
    return result
end


"""
    make_grid_points(inpts, sz=nothing; gridstep=1) -> Matrix{Float64}

Create a grid of points for visualizing TPS warping.

# Arguments
- `inpts::AbstractMatrix`: N×2 point set (used for bounding box if `sz` is nothing)
- `sz=nothing`: (height, width) tuple for grid bounds, or `nothing` to use bounding box of `inpts`
- `gridstep::Int=1`: spacing between grid lines

# Returns
- `Matrix{Float64}`: K×2 matrix of grid points (horizontal + vertical lines)

# Example
```julia
grid = make_grid_points(X, (512, 512); gridstep=20)
grid_warped = tps_rpm_apply(X, params, grid)
img_grid = show_pos((512, 512), grid_warped, 2)
```
"""
function make_grid_points(inpts, sz=nothing; gridstep::Int=1)
    xmin = 1; xmax = 100; ymin=1; ymax=100;
    if isnothing(sz)
        xmin, ymin =  round.(Int,minimum(inpts,dims=1))
        xmax, ymax =  round.(Int,maximum(inpts,dims=1))
    else
        xmax, ymax = sz
    end
    stepx = xmin : gridstep : xmax
    eachy = ymin:ymax
    stepy = ymin : gridstep : ymax
    eachx = xmin:xmax

    pts = Matrix{Float64}(undef, length(stepx)*length(eachy)+length(stepy)*length(eachx), 2)
    k = 1
    @inbounds for y in eachy, x in stepx
            pts[k,1] = x; pts[k,2] = y
        k+=1;
    end
    @inbounds for y in stepy, x in eachx
            pts[k,1] = x; pts[k,2] = y
        k+=1;
    end
    pts
end

"""
    assigment_indices(X, Y, params; min_weight=0.2, max_outlier=0.5, mutual=true) -> Vector{Int}

Return a source-to-destination assignment vector with filtering.

# Arguments
- `X::AbstractMatrix`: N×2 source point set
- `Y::AbstractMatrix`: M×2 destination point set
- `params`: NamedTuple from `tps_rpm` (must contain `A`, `p_src_out`, `p_dst_out`)
- `min_weight::Float64=0.2`: minimum weight on best destination to accept match
- `max_outlier::Float64=0.5`: maximum outlier probability for source and destination
- `mutual::Bool=true`: if true, require mutual best match (1-1 correspondence)

# Returns
- `Vector{Int}`: length N, where `assign[i] = j` (1..M) if matched, `0` if unmatched

# Example
```julia
params = tps_rpm(X, Y; λ=1.0)
assign = assigment_indices(X, Y, params; min_weight=0.1, max_outlier=0.8)
n_matched = count(>(0), assign)
println("Matched \$n_matched of \$(size(X,1)) sources")
```
"""
function assigment_indices(X::AbstractMatrix, Y::AbstractMatrix, params;
                             min_weight::Float64=0.2,
                             max_outlier::Float64=0.5,
                             mutual::Bool=true)
    A = params.A
    N, M = size(X, 1), size(Y, 1)
    @assert size(A,1) == N+1 && size(A,2) == M+1 "A must be (N+1)×(M+1)"

    Areal      = @view A[1:N, 1:M]
    p_src_out  = params.p_src_out  # length N
    p_dst_out  = params.p_dst_out  # length M

    # Column-best indices (for mutual check)
    col_best = Vector{Int}(undef, M)
    @inbounds for j in 1:M
        _, ib = findmax(@view Areal[:, j])
        col_best[j] = ib
    end

    assign = fill(0, N)  # 0 = dummy (no assignment)

    # Row-wise best with gating
    @inbounds for i in 1:N
        # If source mostly outlier → unmatched
        if p_src_out[i] > max_outlier
            assign[i] = 0
            continue
        end
        w, j = findmax(@view Areal[i, :])  # best destination for source i
        good = (w >= min_weight) &&
               (p_dst_out[j] <= max_outlier) &&
               (!mutual || col_best[j] == i)
        assign[i] = good ? j : 0
    end

    return assign
end

"""
    visualize_result(sz, X, Y, params; min_weight=0.0, max_outlier=10000.0, mutual=true, gridstep=15) -> Array{Float64,4}

Create a visualization of the TPS-RPM matching result.

# Arguments
- `sz::Tuple{Int,Int}`: output image size (height, width)
- `X::AbstractMatrix`: N×2 source point set (first argument to `tps_rpm`)
- `Y::AbstractMatrix`: M×2 destination point set (second argument to `tps_rpm`)
- `params`: NamedTuple from `tps_rpm`
- `min_weight::Float64=0.0`: minimum weight for assignment filtering
- `max_outlier::Float64=10000.0`: maximum outlier probability for filtering
- `mutual::Bool=true`: require mutual best match
- `gridstep::Int=15`: spacing for visualization grid lines

# Returns
- `Array{Float64,4}`: H×W×1×5 array with 5 visualization slices:
  1. Source grid
  2. Warped grid
  3. Source points (colored by match index)
  4. Warped source points
  5. Destination reference points

# Example
```julia
params = tps_rpm(X, Y; λ=1.0)
res = visualize_result((512, 512), X, Y, params; gridstep=20)
# View with: @vt res  (if using View5D)
```
"""
function visualize_result(sz, X,Y, params; min_weight=0.0, max_outlier=10000.0, mutual=true, gridstep=15)
    grid = make_grid_points(X, sz; gridstep=gridstep);
    grid_warped = TPSRPM.tps_rpm_apply(X, params, grid)
    
    src_grid = show_pos(sz, grid)
    dst_grid = show_pos(sz, grid_warped, 2)

    idxY = assigment_indices(X, Y, params; min_weight=min_weight, max_outlier=max_outlier, mutual=mutual)
    X_warped = tps_rpm_apply(X, params, X)

    toassign = show_pos(sz, X, idxY)
    assigned = show_pos(sz, X_warped, idxY)
    dst_idx = zeros(Int64, size(Y,1)) .- 1
    dst_idx[idxY[idxY.>0]] = idxY[idxY.>0];
    reference = show_pos(sz, Y, dst_idx)

    res = cat(src_grid, dst_grid, toassign, assigned, reference, dims=4)

    res
end

end # module
