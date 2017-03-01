module utils

#Import existing push!() and pop!() method definitions to qualify our push!() and pop()! methods for export.
import Base.push!,
        Base.pop!;

export if_, FIFOQueue, Stack, push!, pop!, extend!;

function if_(boolean_expression, ans1, ans2)
    if (boolean_expression)
        return ans1;
    else
        return ans2;
    end
end

function distance2(p1::Tuple{Any, Any}, p2::Tuple{Any, Any})
    return (Float64(p1[1]) - Float64(p2[1]))^2 + (Float64(p1[2]) - Float64(p2[2]))^2;
end

function index{T <: Any}(v::Array{T, 1}, item::T)
    local i = 0;
    for element in v
        i = i + 1;
        if (element == item)
            return i;
        end
    end
    return -1;          #couldn't find the item in the array
end

function turn_heading(heading::Tuple{Any, Any}, inc::Int64)
    local o = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    return o[(index(o, heading) + inc) % length(o)];
end

function vector_add_tuples(a::Tuple, b::Tuple)
    return map(+, a, b);
end

#=

    Define a Queue as an abstract DataType.

    FIFOQueue, PriorityQueue, Stack are implementations of the Queue DataType.

=#

abstract Queue;

#=

    Stack is a Last In First Out (LIFO) Queue implementation.

=#
type Stack <: Queue
    array::Array{Any, 1}

    function Stack()
        return new(Array{Any, 1}());
    end
end

#=

    FIFOQueue is a First In First Out (FIFO) Queue implementation.

=#
type FIFOQueue <: Queue
    array::Array{Any, 1}

    function FIFOQueue()
        return new(Array{Any, 1}());
    end
end

#=

    PQueue is a Priority Queue implementation.

    The array must consist of Tuple{Any, Any} such that,

        -the first element is the priority of the item.

        -the second element is the item.

=#
type PQueue <: Queue
    array::Array{Tuple{Any, Any}, 1}

    function PQueue()
        return new(Array{Tuple{Any, Any}}());
    end
end

#=

    Define method definitions of push!(), pop()!, and extend()! for Queue implementations.

=#

"""
    push!(s::Stack, i::Any)

Push the given item 'i' to the end of the collection.
"""
function push!(s::Stack, i::Any)
    push!(s.array, i);
    nothing;
end

"""
    push!(s::FIFOQueue, i::Any)

Push the given item 'i' to the end of the collection.
"""
function push!(fq::FIFOQueue, i::Any)
    push!(fq.array, i);
    nothing;
end

"""
    pop!(s::Stack)

Delete the last item of the collection and return the deleted item.
"""
function pop!(s::Stack)
    return pop!(s.array);
end
"""
    pop!(s::FIFOQueue)

Delete the first item of the collection and return the deleted item.
"""
function pop!(fq::FIFOQueue)
    return shift!(fq.array);
end

"""
    extend!(s1::Stack, s2::Stack)

Push item(s) of s2 to the end of s1.
"""
function extend!(s1::Stack, s2::Stack)
    for e in s2.array
        push!(s1.array, e);
    end
    nothing;
end


"""
    extend!(fq1::FIFOQueue, fq2::FIFOQueue)

Push item(s) of fq2 to the end of fq1.
"""
function extend!(fq1::FIFOQueue, fq2::FIFOQueue)
    for e in fq2.array
        push!(fq1.array, e);
    end
    nothing;
end

# Modified sorted binary search for array of tuples
#   https://github.com/JuliaLang/julia/blob/master/base/sort.jl
#       searchsortedfirst(), searchsortedlast(), and searchsorted()
#
# Base.Order.Forward will make the PQueue ordered by minimums.
# Base.Order.Reverse will make the PQueue ordered by maximums.

end