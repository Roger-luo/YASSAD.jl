xgetindex(x, i...) = xcall(Base, :getindex, x, i...)
xtuple(xs...) = xcall(Core, :tuple, xs...)
xaccum(ir) = nothing
xaccum(ir, x) = x
xaccum(ir, xs...) = push!(ir, xcall(YASSAD, :accum, xs...))
accum() = nothing
accum(x) = x
accum(x, y) =
  x == nothing ? y :
  y == nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, y::Tuple) = accum.(x, y)
accum(x::AbstractArray, y::AbstractArray) = accum.(x, y)
