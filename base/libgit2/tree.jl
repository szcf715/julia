# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    treewalk(f::Function, tree::GitTree, payload=Any[], post::Bool = false)

Traverse the entries in `tree` and its subtrees in post or pre order.

Function parameter should have following signature:

    (Cstring, Ptr{Void}, Ptr{Void}) -> Cint

A negative value returned from `f` stops the tree walk. A positive value means
that the entry will be skipped if `post` is `false`.
"""
function treewalk(f::Function, tree::GitTree, payload=Any[], post::Bool = false)
    cbf = cfunction(f, Cint, Tuple{Cstring, Ptr{Void}, Ptr{Void}})
    cbf_payload = Ref{typeof(payload)}(payload)
    @check ccall((:git_tree_walk, :libgit2), Cint,
                  (Ptr{Void}, Cint, Ptr{Void}, Ptr{Void}),
                   tree.ptr, post, cbf, cbf_payload)
    return cbf_payload
end

repository(tree::GitTree) = tree.owner
repository(te::GitTreeEntry) = repository(te.owner)

"""
    filename(te::GitTreeEntry)

Return the filename of the object on disk to which `te` refers.
"""
function filename(te::GitTreeEntry)
    str = ccall((:git_tree_entry_name, :libgit2), Cstring, (Ptr{Void},), te.ptr)
    str != C_NULL && return unsafe_string(str)
    return nothing
end

"""
    filemode(te::GitTreeEntry) -> Cint

Return the UNIX filemode of the object on disk to which `te` refers as an integer.
"""
function filemode(te::GitTreeEntry)
    return ccall((:git_tree_entry_filemode, :libgit2), Cint, (Ptr{Void},), te.ptr)
end

"""
    entrytype(te::GitTreeEntry)

Return the type of the object to which `te` refers. The result will be
one of the types which [`objtype`](@ref) returns, e.g. a `GitTree` or `GitBlob`.
"""
function entrytype(te::GitTreeEntry)
    otype = ccall((:git_tree_entry_type, :libgit2), Cint, (Ptr{Void},), te.ptr)
    return objtype(Consts.OBJECT(otype))
end

"""
    entryid(te::GitTreeEntry)

Return the [`GitHash`](@ref) of the object to which `te` refers.
"""
function entryid(te::GitTreeEntry)
    oid_ptr = ccall((:git_tree_entry_id, :libgit2), Ptr{UInt8}, (Ptr{Void},), te.ptr)
    return GitHash(oid_ptr)
end

function Base.count(tree::GitTree)
    return ccall((:git_tree_entrycount, :libgit2), Csize_t, (Ptr{Void},), tree.ptr)
end

function Base.getindex(tree::GitTree, i::Integer)
    if i < 1 || i > count(tree)
        throw(BoundsError(tree, i))
    end
    te_ptr = ccall((:git_tree_entry_byindex, :libgit2),
                   Ptr{Void},
                   (Ptr{Void}, Csize_t), tree.ptr, i-1)
    return GitTreeEntry(tree, te_ptr, false)
end

"""
    (::Type{T})(te::GitTreeEntry) where T<:GitObject

Get the git object to which `te` refers and return it as its actual type (the type
[`entrytype`](@ref) would show), for instance a `GitBlob` or `GitTag`.

# Examples
```julia
tree = LibGit2.GitTree(repo, "HEAD^{tree}")
tree_entry = tree[1]
blob = LibGit2.GitBlob(tree_entry)
```
"""
function GitObject(e::GitTreeEntry) end
function (::Type{T})(te::GitTreeEntry) where T<:GitObject
    repo = repository(te)
    obj_ptr_ptr = Ref{Ptr{Void}}(C_NULL)
    @check ccall((:git_tree_entry_to_object, :libgit2), Cint,
                  (Ptr{Ptr{Void}}, Ptr{Void}, Ref{Void}),
                   obj_ptr_ptr, repo.ptr, te.ptr)
    return T(repo, obj_ptr_ptr[])
end

function Base.show(io::IO, te::GitTreeEntry)
    println(io, "GitTreeEntry:")
    println(io, "Entry name: ", filename(te))
    println(io, "Entry type: ", entrytype(te))
    println(io, "Entry OID: ", entryid(te))
end

function Base.show(io::IO, tree::GitTree)
    println(io, "GitTree:")
    println(io, "Owner: ", repository(tree))
    println(io, "Number of entries: ", count(tree))
end
