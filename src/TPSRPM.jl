module TPSRPM # Thin Plate Splin Robust Point Matching

using LinearAlgebra, Statistics
export vc2mat, tps_rpm_apply, tps_rpm, show_pos, make_grid_points, infer_order, assigment_indices, visualize_result

"""
    vc2mat(vec)

converts a vector of cartesian indices (as obtained by findmaxima) into a matrix (as using in inputs of the TPSRPM algorithm).
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
    # P: N×2, Y: N×2 expected targets (possibly fractional from soft assignments)
    N = size(P,1)
    K = _pairwise_U(P)           # N×N
    Pm = [ones(N) P]             # N×3
    A = [K + λ*I  Pm; Pm'  zeros(3,3)]   # (N+3)×(N+3)
    # Solve independently for x and y outputs
    cx = A \ vcat(Y[:,1], zeros(3))
    cy = A \ vcat(Y[:,2], zeros(3))
    wx = view(cx, 1:N); ax = @view cx[N+1:end]  # [a0, ax, ay]
    wy = view(cy, 1:N); ay = @view cy[N+1:end]
    return wx, wy, ax, ay
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
        D[i,M+1] = cout_src         # source outlier
    end
    for j in 1:M
        D[N+1,j] = cout_dst         # destination outlier
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
    tps_rpm(X::AbstractMatrix, Y::AbstractMatrix;
            beta_sched=collect(0.5:0.5:6.0),
            cout=1.0, λ=1e-3, iters_per_beta::Int=3, verbose=false,
            cout_src::Union{Nothing,Float64}=nothing,
            cout_dst::Union{Nothing,Float64}=nothing,
            center::Bool=true,
            init::Symbol=:auto)

calculates an optimized thin-plate-spline (TPS) robust point matching (RPM) between the (non-moving) sources X[N×2] and the
(moving) destination postions Y[M×2]. 

# Parameters
- beta_sched: vector of β values (increasing, signifying the "inverse temperature"), e.g. 0.5:0.5:4.0
The schedule of beta values to follow. The final beta determines how mucht the result corresponds to a permutation matrix.
Anneal to very large β (i.e. a low temperature) to obtain a unique assiment.

- cout: outlier cost for dummy column (used for source and destination if provided)
- cout_src: outlier cost for dummy column sources
- cout_dst: outlier cost for dummy column destination
- λ: TPS regularization
- iters_per_beta: EM steps per β
- center=true: subtract centroids of X and Y before optimization (translation-invariant path). The returned affine constants are mapped back to original coordinates.
- init when center=false: 
    if `init == :centroid`, initialize the affine translation to mean(Y)−mean(X).
    `:auto` behaves like `:centroid` when center=false, otherwise identity.
    `:nothing` means to not apply centering.

Note that this algorithm is asymmetric. The sources should normally be the smaller of the two points to match.
The sources define the thin plate spline (TPS)
Returns warp parameters (wx, wy, ax, ay), final soft matrix A, and diagnostics.

"""
function tps_rpm(X::AbstractMatrix, Y::AbstractMatrix; beta_sched=collect(0.5:0.5:6.0),
                 cout=1.0, λ=1e-3, iters_per_beta::Int=3, verbose=false,
                 cout_src=nothing, cout_dst=nothing,
                 center::Bool=false, init::Symbol=:nothing)

    # Separate penalties (defaults to cout)
    cout_src_val = cout_src === nothing ? cout : cout_src
    cout_dst_val = cout_dst === nothing ? cout : cout_dst

    # Optionally center (zero-mean) to make the path translation-invariant
    μX = vec(mean(X, dims=1))
    μY = vec(mean(Y, dims=1))
    Xc = center ? (X .- (μX')) : X
    Yc = center ? (Y .- (μY')) : Y

    N, M = size(Xc,1), size(Yc,1)

    # Initialize warp (affine can encode pure translation)
    wx = zeros(N); wy = zeros(N)
    ax = [0.0, 1.0, 0.0]  # [a0x, axx, axy]
    ay = [0.0, 0.0, 1.0]  # [a0y, ayx, ayy]
    if !center && (init == :centroid || init == :auto)
        Δ = vec(mean(Y, dims=1) .- mean(X, dims=1))
        ax[1] = Δ[1]; ay[1] = Δ[2]
    end

    # diagnostics log
    diag = Vector{NamedTuple}()

    for (kβ, β) in pairs(beta_sched)
        for n in 1:iters_per_beta
            # E-step: warp X, then bidirectional soft assignment
            Xw = _tps_apply(Xc, wx, wy, ax, ay, Xc)
            Afull, Areal, p_src_out, p_dst_out = soft_assign_bidir(Xw, Yc; beta=β,
                                                                   cout_src=cout_src_val,
                                                                   cout_dst=cout_dst_val)

            # Expected targets for each source (ignore source-outlier mass)
            rowmass = sum(Areal, dims=2)
            EY = similar(Xc, (N,2))
            @inbounds for i in 1:N
                m = rowmass[i]
                if m <= eps()
                    EY[i,1] = Xc[i,1]; EY[i,2] = Xc[i,2]
                else
                    s1 = 0.0; s2 = 0.0
                    @inbounds for j in 1:M
                        w = Areal[i,j]
                        s1 += w * Yc[j,1]; s2 += w * Yc[j,2]
                    end
                    EY[i,1] = s1 / m; EY[i,2] = s2 / m
                end
            end

            # M-step: fit TPS Xc -> EY
            wx, wy, ax, ay = _tps_solve(Xc, EY; λ=λ)

            # Diagnostics (energy decomposition)
            Xw_new = _tps_apply(Xc, wx, wy, ax, ay, Xc)
            comps = _energy_components(; X=Xc, Xw=Xw_new, Y=Yc, Afull=Afull, Areal=Areal,
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

    # Final A at last β (centered frame)
    Xw_last = _tps_apply(Xc, wx, wy, ax, ay, Xc)
    A, Areal, p_src_out, p_dst_out = soft_assign_bidir(Xw_last, Yc;
                                                       beta=last(beta_sched),
                                                       cout_src=cout_src_val, cout_dst=cout_dst_val)

    # Map affine constants back to original coordinates if centered:
    # f_orig(p) = a0' + A p + Σ w U(||p - X_j||), with a0' = a0 - A μX + μY
    if center
        a0x = ax[1] + μY[1] - (ax[2]*μX[1] + ax[3]*μX[2])
        a0y = ay[1] + μY[2] - (ay[2]*μX[1] + ay[3]*μX[2])
        ax = [a0x, ax[2], ax[3]]
        ay = [a0y, ay[2], ay[3]]
        # Note: weights wx/wy and centers X can stay in original coords (U uses distances, translation-invariant)
    end

    return (; wx, wy, ax, ay, A, Areal, p_src_out, p_dst_out, diag)
end


"""
    show_pos(sz, posmat, idxpos)

creates an image with the positions as dots and filled by the index (if present) plus one.
    This yields unassigned sources having the number one.
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


"Apply final warp to arbitrary points Q"
function tps_rpm_apply(X::AbstractMatrix, params::NamedTuple, Q::AbstractMatrix)
    _tps_apply(X, params.wx, params.wy, params.ax, params.ay, Q)
end


"""
    make_grid_points(inpts, sz; mystep::Int=1)
Create grid points over an image of size (H,W). order=:xy or :yx.

# Parameters
- inpts: input vector
- sz: optional size to define the bounding box. If not given, the ranges are estimated by the bounding box of inpts.
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
    assigment_indices(X, Y, params; min_weight=0.2, max_outlier=0.5, mutual=true)

Return a source-to-destination assignment vector of length N:
- assign[i] = j (1..M) if source i is matched to destination j
- assign[i] = 0 if unmatched (assigned to dummy column)

Arguments:
- X, Y: point sets (N×2, M×2)
- params: result from `tps_rpm` (must contain `A`, `p_src_out`, `p_dst_out`)
- min_weight: minimum row mass on best destination to accept a match
        It require the best column for a source to carry enough mass.
- max_outlier: max allowed outlier probability for source i and destination j
        It rejects sources/destinations whose probability goes mostly to the outlier.
- mutual: if true, require mutual best (i is the argmax in column j)
        It enforces a simple 1–1 (mutual best) filter.
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
