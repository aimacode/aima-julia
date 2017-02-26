module utils

export if_;

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

end