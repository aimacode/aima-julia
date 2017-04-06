
import Base: get, getindex, getkey,
            copy, haskey, in, display;

export ConstantFunctionDict, CSPDict, CSP,
    get, getkey, getindex, copy, haskey,
    in,
    assign, unassign, nconflicts, display;

#Constraint Satisfaction Problems (CSP)

type ConstantFunctionDict{V}
    value::V

    function ConstantFunctionDict{V}(val::V)
        return new(val);
    end
end

ConstantFunctionDict(val) = ConstantFunctionDict{typeof(val)}(val);

copy(cfd::ConstantFunctionDict) = ConstantFunctionDict{typeof(cfd.value)}(cfd.value);

type CSPDict
    dict::Nullable

    function CSPDict(dictionary::Union{Dict, ConstantFunctionDict})
        return new(Nullable(dictionary));
    end
end

function getindex(dict::CSPDict, key)
    if (eltype(dict.dict) <: ConstantFunctionDict)
        return get(dict.dict).value;
    else
        return getindex(get(dict.dict), key);
    end
end

function getkey(dict::CSPDict, key, default)
    if (eltype(dict.dict) <: ConstantFunctionDict)
        return get(dict.dict).value;
    else
        return getkey(get(dict.dict), key, default);
    end
end
 
function get(dict::CSPDict, key, default)
    if (eltype(dict.dict) <: ConstantFunctionDict)
        return get(dict.dict).value;
    else
        return get(get(dict.dict), key, default);
    end
end

function haskey(dict::CSPDict, key)
    if (eltype(dict.dict) <: ConstantFunctionDict)
        return true;
    else
        return haskey(get(dict.dict), key);
    end
end

function in(pair::Pair, dict::CSPDict)
    if (eltype(dict.dict) <: ConstantFunctionDict)
        if (getindex(pair, 2) == get(dict.dict).value)
            return true;
        else
            return false;
        end
    else
        #Call in() function from dict.jl(0.5)/associative.jl(0.6~nightly).
        return in(pair, get(dict.dict));
    end
end

#=

    CSP is a Constraint Satisfaction Problem implementation of AbstractProblem.

    This problem contains an unused initial state field to accommodate the requirements 

    of some search algorithms.

=#
type CSP <: AbstractProblem
	vars::AbstractVector
	domains::CSPDict
	neighbors::CSPDict
	constraints::Function
	initial::Tuple
	current_domains::Nullable{Dict}
	nassigns::Int64

    function CSP(vars::AbstractVector, domains::CSPDict, neighbors::CSPDict, constraints::Function;
                initial::Tuple=(), current_domains::Union{Void, Dict}=nothing, nassigns::Int64=0)
        return new(vars, domains, neightbors, initial, Nullable{Dict}(current_domains), nassigns)
    end
end

function assign(problem::CSP, key, val, assignment::Dict)
    println("key: ", typeof(key), " | val: ", typeof(val));
    assignment[key] = val;
    problem.nassigns = problem.nassigns + 1;
    nothing;
end

function unassign(problem::CSP, key, assignment::Dict)
    println("key: ", typeof(key));
    if (haskey(assignment, key))
        delete!(assignment, key);
    end
    nothing;
end

function nconflicts(problem::CSP, var, val, assignment::Dict)
    return count(
                (function(second_var, ; relevant_problem::CSP=problem, first_var=var, relevant_val=val, dict::Dict=assignment)
                    return (haskey(dict, second_var) &&
                        !(relevant_problem.constraints(first_var, relevant_val, second_var, dict[second_var])));
                end),
                problem.neighbors[var]);
end

function display(problem::CSP, assignment::Dict)
    println("CSP: ", problem, " with assignment: ", assignment);
    nothing;
end

