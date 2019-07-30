using LinearAlgebra

@inline function forward(::typeof(*), A::Matrix, B::Matrix)
    A * B, function (Δ::Matrix)
        Base.@_inline_meta
        (nothing, Δ * B', A' * Δ)
    end
end

@inline function forward(::typeof(tr), A::Matrix)
    tr(A), function (Δ::Real)
        Base.@_inline_meta
        (nothing, Δ * Matrix(I, size(A)))
    end
end
