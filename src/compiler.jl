export forward

using IRTools
using IRTools: var, IR, block,
    argument!, arguments, Variable,
    slots!, pis!, inlineable!,
    argnames!, varargs!, Pipe, xcall,
    insertafter!, stmt, substitute, finish,
    return!, block, arguments, blocks


"""
    Pullback{S, T}

A struct used for simulating closures. `S` is the signature of function,
`T` is a `Tuple` type to store pullbacks.
"""
struct Pullback{S, T}
    data::T
end

Pullback{S}(data::T) where {S, T} = Pullback{S, T}(data)

"""
    track(ir, F)

This function will transform the original function call in the original IR `ir` to
`forward` function calls. Besides the forward evaluation results, `forward` function
will also return a pullback function (closure) for backward evaluation use.
"""
function track(ir, F)
    pr = Pipe(ir)
    pbs = Dict{Variable, Variable}()
    argument!(pr, at = 1)
    for (v, st) in pr
        ex = st.expr
        if Meta.isexpr(ex, :call)
            yJ = insert!(pr, v, stmt(xcall(YASSAD, :forward, ex.args...), line = ir[v].line))
            pr[v] = xgetindex(yJ, 1)
            J = insertafter!(pr, v, stmt(xgetindex(yJ, 2), line = ir[v].line))
            pbs[v] = substitute(pr, J)
        end
    end
    pr = finish(pr)
    v = push!(pr, xtuple(values(pbs)...))
    pbv = push!(pr, Expr(:call, Pullback{F}, v))
    ret = pr.blocks[end].branches[end].args[1]
    ret = push!(pr, xtuple(ret, pbv))
    pr.blocks[end].branches[end].args[1] = ret
    return pr, pbs
end

"""
    forward(f, xs...)

This is the entry of IR transformation.
"""
@generated function forward(f, xs...)
    T = Tuple{f, xs...}
    m = IRTools.meta(T)
    m === nothing && return
    frw, _ = track(IR(m), T)
    argnames!(m, Symbol("#self#"), :f, :xs)
    frw = varargs!(m, frw, 2)
    # frw = slots!(pis!(inlineable!(frw)))
    return IRTools.update!(m, frw)
end

"""
    adjoint(ir, pbs)

This generates the adjoint function of forward pass. (the closure).
"""
function adjoint(ir, pbs)
    adj = empty(ir)
    self = argument!(adj)
    delta = argument!(adj)
    pullbacks = pushfirst!(adj, xcall(:getfield, self, QuoteNode(:data)))

    grads = Dict()
    grad(x, x̄) = push!(get!(grads, x, []), x̄)
    grad(x) = xaccum(adj, get(grads, x, [])...)
    grad(last(keys(ir)), delta)

    vars = keys(ir)
    for k in length(vars):-1:1
        v = vars[k]
        ex = ir[v].expr
        if haskey(pbs, v)
            pbv = insertafter!(adj, pullbacks, xcall(:getindex, pullbacks, k))
            g = push!(adj, Expr(:call, pbv, grad(v)))

            for (i, x) in enumerate(ex.args)
                x isa Variable || continue
                grad(x, push!(adj, xgetindex(g, i)))
            end
        end
    end
    gs = [grad(x) for x in arguments(ir)]
    Δ = push!(adj, xtuple(gs...))
    return!(adj, Δ)
    return adj
end

"""
our fake closure.
"""
@generated function (::Pullback{S})(delta) where S
    m = IRTools.meta(S)
    m === nothing && return
    ir = IR(m)
    _, pbs = track(ir, S)
    back = adjoint(ir, pbs)
    argnames!(m, Symbol("#self#"), :delta)
    return IRTools.update!(m, back)
end
