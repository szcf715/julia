# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    ConjArray(array)

A lazy-view wrapper of an `AbstractArray`, taking the elementwise complex conjugate. This
type is usually constructed (and unwrapped) via the [`conj`](@ref) function (or related
[`adjoint`](@ref)), but currently this is the default behavior for `RowVector` only. For
other arrays, the `ConjArray` constructor can be used directly.

# Examples
```jldoctest
julia> [1+im, 1-im]'
1×2 RowVector{Complex{Int64},ConjArray{Complex{Int64},1,Array{Complex{Int64},1}}}:
 1-1im  1+1im

julia> ConjArray([1+im 0; 0 1-im])
2×2 ConjArray{Complex{Int64},2,Array{Complex{Int64},2}}:
 1-1im  0+0im
 0+0im  1+1im
```
"""
struct ConjArray{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    parent::A
end

@inline ConjArray(a::AbstractArray{T,N}) where {T,N} = ConjArray{conj_type(T),N,typeof(a)}(a)

const ConjVector{T,V<:AbstractVector} = ConjArray{T,1,V}
@inline ConjVector(v::AbstractVector{T}) where {T} = ConjArray{conj_type(T),1,typeof(v)}(v)

const ConjMatrix{T,M<:AbstractMatrix} = ConjArray{T,2,M}
@inline ConjMatrix(m::AbstractMatrix{T}) where {T} = ConjArray{conj_type(T),2,typeof(m)}(m)

# This type can cause the element type to change under conjugation - e.g. an array of complex arrays.
@inline conj_type(x) = conj_type(typeof(x))
@inline conj_type(::Type{T}) where {T} = promote_op(conj, T)

@inline parent(c::ConjArray) = c.parent
@inline parent_type(c::ConjArray) = parent_type(typeof(c))
@inline parent_type(::Type{ConjArray{T,N,A}}) where {T,N,A} = A

@inline size(a::ConjArray) = size(a.parent)
IndexStyle(::CA) where {CA<:ConjArray} = IndexStyle(parent_type(CA))
IndexStyle(::Type{CA}) where {CA<:ConjArray} = IndexStyle(parent_type(CA))

@propagate_inbounds getindex(a::ConjArray{T,N}, i::Int) where {T,N} = conj(getindex(a.parent, i))::T
@propagate_inbounds getindex(a::ConjArray{T,N}, i::Vararg{Int,N}) where {T,N} = conj(getindex(a.parent, i...))::T
@propagate_inbounds setindex!(a::ConjArray{T,N}, v, i::Int) where {T,N} = setindex!(a.parent, conj(v), i)
@propagate_inbounds setindex!(a::ConjArray{T,N}, v, i::Vararg{Int,N}) where {T,N} = setindex!(a.parent, conj(v), i...)

@inline similar(a::ConjArray, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(a), T, dims)

# Currently, this is default behavior for RowVector only
@inline conj(a::ConjArray) = parent(a)
@inline adjoint(a::ConjArray) = ConjAdjointArray(parent(a))

"""
AdjointArray(array)

A lazy-view wrapper of an `AbstractArray`, taking the elementwise adjoint. This
type is usually constructed (and unwrapped) via the [`adjoint`](@ref) function, but 
currently this is the default behavior for `RowVector` only. For other arrays, the 
`AdjointArray` constructor can be used directly.

# Examples
```jldoctest
julia> [1+im, 1-im]'
1×2 RowVector{Complex{Int64},AdjointArray{Complex{Int64},1,Array{Complex{Int64},1}}}:
1-1im  1+1im

julia> AdjointArray([1+im 0; 0 1-im])
2×2 AdjointArray{Complex{Int64},2,Array{Complex{Int64},2}}:
1-1im  0+0im
0+0im  1+1im
```
"""
struct AdjointArray{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    parent::A
end

@inline AdjointArray(a::AbstractArray{T,N}) where {T,N} = AdjointArray{adjoint_type(T),N,typeof(a)}(a)

const AdjointVector{T,V<:AbstractVector} = AdjointArray{T,1,V}
@inline AdjointVector(v::AbstractVector{T}) where {T} = AdjointArray{adjoint_type(T),1,typeof(v)}(v)

const AdjointMatrix{T,M<:AbstractMatrix} = AdjointArray{T,2,M}
@inline AdjointMatrix(m::AbstractMatrix{T}) where {T} = AdjointArray{adjoint_type(T),2,typeof(m)}(m)

# This type can cause the element type to change under conjugation - e.g. an array of complex arrays.
@inline adjoint_type(x) = conj_type(typeof(x))
@inline adjoint_type(::Type{T}) where {T} = promote_op(adjoint, T)

@inline parent(c::AdjointArray) = c.parent
@inline parent_type(c::AdjointArray) = parent_type(typeof(c))
@inline parent_type(::Type{AdjointArray{T,N,A}}) where {T,N,A} = A

@inline size(a::AdjointArray) = size(a.parent)
IndexStyle(::AA) where {AA<:AdjointArray} = IndexStyle(parent_type(AA))
IndexStyle(::Type{AA}) where {AA<:AdjointArray} = IndexStyle(parent_type(AA))

@propagate_inbounds getindex(a::AdjointArray{T,N}, i::Int) where {T,N} = adjoint(getindex(a.parent, i))::T
@propagate_inbounds getindex(a::AdjointArray{T,N}, i::Vararg{Int,N}) where {T,N} = adjoint(getindex(a.parent, i...))::T
@propagate_inbounds setindex!(a::AdjointArray{T,N}, v, i::Int) where {T,N} = setindex!(a.parent, adjoint(v), i)
@propagate_inbounds setindex!(a::AdjointArray{T,N}, v, i::Vararg{Int,N}) where {T,N} = setindex!(a.parent, adjoint(v), i...)

@inline similar(a::AdjointArray, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(a), T, dims)

# Currently, this is default behavior for RowVector only
@inline adjoint(a::AdjointArray) = parent(a)
@inline conj(a::AdjointArray) = ConjAdjointArray(parent(a))

"""
ConjAdjointArray(array)

A lazy-view wrapper of an `AbstractArray`, mapping each element `i` to `conj(adjoint(i))`.
This type is usually constructed (and unwrapped) via consecutive [`adjoint`](@ref) and
[`conj`](@ref) functions, but currently this is the default behavior for `RowVector` only.
For other arrays, the `ConjAdjointArray` constructor can be used directly.

# Examples
```jldoctest
julia> conj([1+im, 1-im]')
1×2 RowVector{Complex{Int64},ConjAdjointArray{Complex{Int64},1,Array{Complex{Int64},1}}}:
1+1im  1-1im

julia> ConjAdjointArray([1+im 0; 0 1-im])
2×2 ConjAdjointArray{Complex{Int64},2,Array{Complex{Int64},2}}:
1+1im  0+0im
0+0im  1-1im
```
"""
struct ConjAdjointArray{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    parent::A
end

@inline ConjAdjointArray(a::AbstractArray{T,N}) where {T,N} = ConjAdjointArray{conjadjoint_type(T),N,typeof(a)}(a)

const ConjAdjointVector{T,V<:AbstractVector} = ConjAdjointArray{T,1,V}
@inline ConjAdjointVector(v::AbstractVector{T}) where {T} = ConjAdjointArray{conjadjoint_type(T),1,typeof(v)}(v)

const ConjAdjointMatrix{T,M<:AbstractMatrix} = ConjAdjointArray{T,2,M}
@inline ConjAdjointMatrix(m::AbstractMatrix{T}) where {T} = ConjAdjointArray{conjadjoint_type(T),2,typeof(m)}(m)

# This type can cause the element type to change under conjugation - e.g. an array of complex arrays.
@inline conjadjoint_type(x) = conjadjoint_type(typeof(x))
@inline conjadjoint_type(::Type{T}) where {T} = promote_op(x -> conj(adjoint(x)), T)

@inline parent(c::ConjAdjointArray) = c.parent
@inline parent_type(c::ConjAdjointArray) = parent_type(typeof(c))
@inline parent_type(::Type{ConjAdjointArray{T,N,A}}) where {T,N,A} = A

@inline size(a::ConjAdjointArray) = size(a.parent)
IndexStyle(::A) where {A<:ConjAdjointArray} = IndexStyle(parent_type(A))
IndexStyle(::Type{A}) where {A<:ConjAdjointArray} = IndexStyle(parent_type(A))

@propagate_inbounds getindex(a::ConjAdjointArray{T,N}, i::Int) where {T,N} = conj(adjoint(getindex(a.parent, i)))::T
@propagate_inbounds getindex(a::ConjAdjointArray{T,N}, i::Vararg{Int,N}) where {T,N} = conj(adjoint(getindex(a.parent, i...)))::T
@propagate_inbounds setindex!(a::ConjAdjointArray{T,N}, v, i::Int) where {T,N} = setindex!(a.parent, conj(adjoint(v)), i)
@propagate_inbounds setindex!(a::ConjAdjointArray{T,N}, v, i::Vararg{Int,N}) where {T,N} = setindex!(a.parent, conj(adjoint(v)), i...)

@inline similar(a::ConjAdjointArray, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(a), T, dims)

# Currently, this is default behavior for RowVector only
@inline adjoint(a::ConjAdjointArray) = ConjArray(parent(a))
@inline conj(a::ConjAdjointArray) = AdjointArray(parent(a))

# Helper functions, currently used by RowVector
# (some of these are simplify types that would not typically occur)
@inline _conj(a::AbstractArray) = ConjArray(a)
@inline _conj(a::AbstractArray{<:Real}) = a
@inline _conj(a::ConjArray) = parent(a)
@inline _conj(a::AdjointArray) = ConjAdjointArray(parent(a))
@inline _conj(a::AdjointArray{<:Number}) = parent(a)
@inline _conj(a::ConjAdjointArray) = AdjointArray(parent(a))
@inline _conj(a::ConjAdjointArray{<:Real}) = parent(a)
@inline _conj(a::ConjAdjointArray{<:Number}) = ConjArray(parent(a))

@inline _adjoint(a::AbstractArray) = AdjointArray(a)
@inline _adjoint(a::AbstractArray{<:Real}) = a
@inline _adjoint(a::AbstractArray{<:Number}) = ConjArray(a)
@inline _adjoint(a::AdjointArray) = parent(a)
@inline _adjoint(a::ConjArray) = ConjAdjointArray(parent(a))
@inline _adjoint(a::ConjArray{<:Number}) = parent(a)
@inline _adjoint(a::ConjAdjointArray) = ConjArray(parent(a))
@inline _adjoint(a::ConjAdjointArray{<:Real}) = parent(a)
@inline _adjoint(a::ConjAdjointArray{<:Number}) = ConjArray(parent(a))
