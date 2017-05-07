module utils

#Import existing push!() and pop!() method definitions to qualify our push!() and pop()! methods for export.
import Base.push!,
        Base.pop!,
        Base.start,
        Base.next,
        Base.done,
        Base.length,
        Base.delete!;

export if_, Queue, FIFOQueue, Stack, PQueue, push!, pop!, extend!, delete!,
        start, next, done, length,
        MemoizedFunction, eval_memoized_function,
        AbstractProblem,
        argmin, argmax, argmin_random_tie, argmax_random_tie,
        weighted_sampler, weighted_sample_with_replacement,
        distance, distance2,
        RandomDeviceInstance,
        isfunction, removeall;

function if_(boolean_expression::Bool, ans1::Any, ans2::Any)
    if (boolean_expression)
        return ans1;
    else
        return ans2;
    end
end

function distance(p1::Tuple{Number, Number}, p2::Tuple{Number, Number})
    return sqrt(((Float64(p1[1]) - Float64(p2[1]))^2) + ((Float64(p1[2]) - Float64(p2[2]))^2));
end

function distance2(p1::Tuple{Number, Number}, p2::Tuple{Number, Number})
    return (Float64(p1[1]) - Float64(p2[1]))^2 + (Float64(p1[2]) - Float64(p2[2]))^2;
end

function null_index(v::AbstractVector)
    local i::Int64 = 0;
    for element in v
        i = i + 1;
        if (isnull(element))
            return i;
        end
    end
    return -1;          #couldn't find the item in the array
end

function index{T <: Any}(v::Array{T, 1}, item::T)
    local i::Int64 = 0;
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

abstract AbstractProblem;

RandomDeviceInstance = RandomDevice();

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

# Map length function calls to underlying array in Stack
length(s::Stack) = length(s.array);
isempty(s::Stack) = isempty(s.array);

# Map iterator function calls to underlying array in Stack
start(s::Stack) = start(s.array);
next(s::Stack, i) = next(s.array, i);
done(s::Stack, i) = done(s.array, i);

#=

    FIFOQueue is a First In First Out (FIFO) Queue implementation.

=#
type FIFOQueue <: Queue
    array::Array{Any, 1}

    function FIFOQueue()
        return new(Array{Any, 1}());
    end
end

# Map length function calls to underlying array in FIFOQueue
length(fq::FIFOQueue) = length(fq.array);
isempty(fq::FIFOQueue) = isempty(fq.array);

# Map iterator function calls to underlying array in FIFOQueue
start(fq::FIFOQueue) = start(fq.array);
next(fq::FIFOQueue, i) = next(fq.array, i);
done(fq::FIFOQueue, i) = done(fq.array, i);

#=

    PQueue is a Priority Queue implementation.

    The array must consist of Tuple{Any, Any} such that,

        -the first element is the priority of the item.

        -the second element is the item.

=#
type PQueue <: Queue
    array::Array{Tuple{Any, Any}, 1}
    order::Base.Order.Ordering

    function PQueue(;order::Base.Order.Ordering=Base.Order.Forward)
        return new(Array{Tuple{Any, Any}, 1}(), order);
    end
end

# Map length function calls to underlying array in PQueue
length(pq::PQueue) = length(pq.array);
isempty(pq::PQueue) = isempty(pq.array);

# Map iterator function calls to underlying array in PQueue
start(pq::PQueue) = start(pq.array);
next(pq::PQueue, i) = next(pq.array, i);
done(pq::PQueue, i) = done(pq.array, i);

#=

    MemoizedFunction is a DataType that wraps the original function (.f) with a dictionary

    (.values) containing the previous given arguments as keys to their computed values.

=#

type MemoizedFunction
    f::Function     #original function
    values::Dict{Tuple{Vararg}, Any}

    function MemoizedFunction(f::Function)
        return new(f, Dict{Tuple{Vararg}, Any}());
    end
end

function eval_memoized_function(mf::MemoizedFunction, args::Vararg{Any})
    if (haskey(mf.values, args))
        return mf.values[args];
    else
        mf.values[args] = mf.f(args...);
        return mf.values[args];
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
    push!(fq::FIFOQueue, i::Any)

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
    pop!(fq::FIFOQueue)

Delete the first item of the collection and return the deleted item.
"""
function pop!(fq::FIFOQueue)
    return shift!(fq.array);
end

"""
    extend!(s1::Stack, s2::Queue)
    extend!(s1::Stack, s2::AbstractVector)

Push item(s) of s2 to the end of s1.
"""
function extend!{T <: Queue}(s1::Stack, s2::T)
    if (!(typeof(s2) <: PQueue))
        for e in s2.array
            push!(s1, e);
        end
    else
        for e in s2.array
            push!(s1, getindex(e, 2));
        end
    end
    nothing;
end

function extend!(s1::Stack, s2::AbstractVector)
    for e in s2
        push!(s1, e);
    end
    nothing;
end

"""
    extend!(fq1::FIFOQueue, fq2::Queue)
    extend!(fq1::FIFOQueue, fq2::AbstractVector)

Push item(s) of fq2 to the end of fq1.
"""
function extend!{T <: Queue}(fq1::FIFOQueue, fq2::T)
    if (!(typeof(fq2) <: PQueue))
        for e in fq2.array
            push!(fq1, e);
        end
    else
        for e in fq2.array
            push!(fq1, getindex(e, 2));
        end
    end
    nothing;
end

function extend!(fq1::FIFOQueue, fq2::AbstractVector)
    for e in fq2
        push!(fq1, e);
    end
    nothing;
end

# Modified sorted binary search for array of tuples
#   https://github.com/JuliaLang/julia/blob/master/base/sort.jl
#       searchsortedfirst(), searchsortedlast(), and searchsorted()
#
#       These 3 functions were renamed to avoid confusion.
#
#
# Base.Order.Forward will make the PQueue ordered by minimums.
# Base.Order.Reverse will make the PQueue ordered by maximums.
function bisearchfirst{T <: Tuple{Any, Any}}(v::Array{T, 1}, x::T, lo::Int, hi::Int, o::Base.Sort.Ordering)
    lo = lo-1;
    hi = hi+1;
    @inbounds while (lo < hi-1)
        m = (lo+hi) >>> 1
        if (Base.Order.lt(o, getindex(v[m], 1), getindex(x, 1)))
            lo = m;
        else
            hi = m;
        end
    end
    return hi;
end

function bisearchlast{T <: Tuple{Any, Any}}(v::Array{T, 1}, x::T, lo::Int, hi::Int, o::Base.Sort.Ordering)
    lo = lo-1;
    hi = hi+1;
    @inbounds while (lo < hi-1)
        m = (lo+hi) >>> 1;
        if (Base.Order.lt(o, getindex(x, 1), getindex(v[m], 1)))
            hi = m;
        else
            lo = m;
        end
    end
    return lo;
end

function bisearch{T <: Tuple{Any, Any}}(v::Array{T, 1}, x::T, ilo::Int, ihi::Int, o::Base.Sort.Ordering)
    lo = ilo-1;
    hi = ihi+1;
    @inbounds while (lo < hi-1)
        m = (lo+hi) >>> 1;
        if (Base.Order.lt(o, getindex(v[m], 1), getindex(x, 1)))
            lo = m;
        elseif (Base.Order.lt(o, getindex(x, 1), getindex(v[m], 1)))
            hi = m;
        else
            a = bisearchfirst(v, x, max(lo, ilo), m, o)
            b = bisearchlast(v, x, m, min(hi, ihi), o)
            return a : b;
        end
    end
    return (lo + 1) : (hi - 1);
end

"""
    push!(pq::PQueue, i::Tuple{Any, Tuple})

Push the given item 'i' to the index after existing entries with the same priority as getitem(i, 1).
"""
function push!(pq::PQueue, item::Tuple{Any, Any})
    bsi = bisearch(pq.array, item, 1, length(pq), pq.order);

    if (pq.order == Base.Order.Forward)
        insert!(pq.array, bsi.stop + 1, item);
    else
        insert!(pq.array, bsi.start, item);
    end
    nothing;
end

function push!(pq::PQueue, item::Any, mf::MemoizedFunction)
    local item_tuple = (eval_memoized_function(mf, item), item);
    bsi = bisearch(pq.array, item_tuple, 1, length(pq), pq.order);

    if (pq.order == Base.Order.Forward)
        insert!(pq.array, bsi.stop + 1, item_tuple);
    else
        insert!(pq.array, bsi.start, item_tuple);
    end
    nothing;
end

function push!(pq::PQueue, item::Any, mf::Function)
    local item_tuple = (mf(item), item);
    bsi = bisearch(pq.array, item_tuple, 1, length(pq), pq.order);

    if (pq.order == Base.Order.Forward)
        insert!(pq.array, bsi.stop + 1, item_tuple);
    else
        insert!(pq.array, bsi.start, item_tuple);
    end
    nothing;
end

function push!{T <: AbstractProblem}(pq::PQueue, item::Any, mf::MemoizedFunction, problem::T)
    local item_tuple = (eval_memoized_function(mf, problem, item), item);
    bsi = bisearch(pq.array, item_tuple, 1, length(pq), pq.order);

    if (pq.order == Base.Order.Forward)
        insert!(pq.array, bsi.stop + 1, item_tuple);
    else
        if (bsi.start != 0)
            insert!(pq.array, bsi.start, item_tuple);
        else
            insert!(pq.array, 1, item_tuple);
        end
    end
    nothing;
end

"""
    pop!(pq::PQueue)

Delete the lowest/highest item based on ordering of the collection and return the deleted item.
"""
function pop!(pq::PQueue)
    return getindex(shift!(pq.array), 2);   #return lowest/highest priority tuple by pq.order
end

"""
    extend!(pq1::PQueue, pq2::Queue, pv::Function)
    extend!(pq1::PQueue, pq2::AbstractVector, pv::Function)

Push item(s) of pq2 to pq1 by the priority of the item(s) returned by pv().
"""
function extend!{T <: Queue}(pq1::PQueue, pq2::T, pv::Function)
    if (!(typeof(pq2) <: PQueue))
        for e in pq2.array
            push!(pq1, (pv(e), e));
        end
    else
        for e in pq2.array
            push!(pq1, (pv(getindex(e, 2)), getindex(e, 2)));
        end
    end
    nothing;
end

#pv - function that returns priority value
function extend!(pq1::PQueue, pq2::AbstractVector, pv::Function)
    for e in pq2
        push!(pq1, (pv(e), e));
    end
    nothing;
end

#mpv - function that returns memoized priority value
function extend!(pq1::PQueue, pq2::AbstractVector, mpv::MemoizedFunction)
    for e in pq2
        push!(pq1, (eval_memoized_function(mpv, e), e));
    end
    nothing;
end

function extend!{T <: AbstractProblem}(pq1::PQueue, pq2::AbstractVector, mpv::MemoizedFunction, problem::T)
    for e in pq2
        push!(pq1, (eval_memoized_function(mpv, problem, e), e));
    end
    nothing;
end

"""
    delete!(pq::PQueue, item::Any)

Remove the item if it already exists in pq.array.
"""
function delete!(pq::PQueue, item::Any)
    for (i, entry) in enumerate(pq.array)
        if (item == getindex(entry, 2))
            deleteat!(pq.array, i);
            return nothing;
        end
    end
    return nothing;
end

function removeall(v::String, item)
    return replace(v, item, "");
end

function removeall(v::AbstractVector, item)
    return collect(x for x in v if (x != item));
end

"""
    weighted_sample_with_replacement(seq, weights, n)

Return an array of 'n' elements that are chosen from 'seq' at random with replacement, with
the probability of picking each element based on its corresponding weight in 'weights'.
"""
function weighted_sample_with_replacement{T1 <: AbstractVector, T2 <: AbstractVector}(seq::T1, weights::T2, n::Int64)
    local sample = weighted_sampler(seq, weights);
    return collect(sample() for i in 1:n);
end

function weighted_sample_with_replacement{T <: AbstractVector}(seq::String, weights::T, n::Int64)
    local sample = weighted_sampler(seq, weights);
    return collect(sample() for i in 1:n);
end

"""
    weighted_sampler(seq, weights)

Return a random sample function that chooses an element from 'seq' based on its corresponding
weight in 'weight'.
"""
function weighted_sampler{T1 <: AbstractVector, T2 <: AbstractVector}(seq::T1, weights::T2)
    local totals = Array{Any, 1}();
    for w in weights
        if (length(totals) != 0)
            push!(totals, (w + totals[length(totals)]));
        else
            push!(totals, w);
        end
    end
    return (function(;sequence=seq, totals_array=totals)
                bsi = bisearch(totals_array,
                                (rand(RandomDeviceInstance)*totals_array[length(totals_array)]),
                                1,
                                length(totals_array),
                                Base.Order.Forward);
                return seq[bsi.stop + 1];
            end);
end

function weighted_sampler{T <: AbstractVector}(seq::String, weights::T)
    local totals = Array{Any, 1}();
    for w in weights
        if (length(totals) != 0)
            push!(totals, (w + totals[length(totals)]));
        else
            push!(totals, w);
        end
    end
    return (function(;sequence=seq, totals_array=totals)
                bsi = searchsorted(totals_array,
                                (rand(RandomDeviceInstance)*totals_array[length(totals_array)]),
                                1,
                                length(totals_array),
                                Base.Order.Forward);
                return seq[bsi.stop + 1];
            end);
end

"""
    argmin(seq, fn)

Applies fn() to each element in seq and returns the element that has the lowest fn() value. argmin()
is similar to mapreduce(fn, min, seq) in computing the best score, but returns the corresponding element.
"""
function argmin{T <: AbstractVector}(seq::T, fn::Function)
    local best_element = seq[1];
    local best_score = fn(best_element);
    for element in seq
        element_score = fn(element);
        if (element_score < best_score)
            best_element = element;
            best_score = element_score;
        end
    end
    return best_element;
end

function argmin_random_tie{T <: AbstractVector}(seq::T, fn::Function)
    local best_score = fn(seq[1]);
    local n::Int64 = 0;
    local best_element;
    for element in seq
        element_score = fn(element);
        if (element_score < best_score)
            best_element = element;
        elseif (element_score == best_score)
            n = n + 1;
            if (rand(RandomDeviceInstance, 1:n) == 1)
                best_element = element;
            end
        end
    end
    return best_element;
end

"""
    argmax(seq, fn)

Applies fn() to each element in seq and returns the element that has the highest fn() value. argmax()
is similar to mapreduce(fn, max, seq) in computing the best score, but returns the corresponding element.
"""
function argmax{T <: AbstractVector}(seq::T, fn::Function)
    local best_element = seq[1];
    local best_score = fn(best_element);
    for element in seq
        element_score = fn(element);
        if (element_score > best_score)
            best_element = element;
            best_score = element_score;
        end
    end
    return best_element;
end

function argmax_random_tie{T <: AbstractVector}(seq::T, fn::Function)
    local best_score = fn(seq[1]);
    local n::Int64 = 1;
    local best_element;
    for element in seq
        element_score = fn(element);
        if (element_score > best_score)
            best_element = element;
        elseif (element_score == best_score)
            n = n + 1;
            if (rand(RandomDeviceInstance, 1:n) == 1)
                best_element = element;
            end
        end
    end
    return best_element;
end

"""
    isfunction(var)

Check if 'var' is callable as a function.
"""
function isfunction(var)
    return (typeof(var) <: Function);
end

end