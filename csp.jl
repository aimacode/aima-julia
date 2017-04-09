
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

abstract AbstractCSP <: AbstractProblem;

#=

    CSP is a Constraint Satisfaction Problem implementation of AbstractProblem.

    This problem contains an unused initial state field to accommodate the requirements 

    of some search algorithms.

=#
type CSP <: AbstractCSP
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

function assign{T <: AbstractCSP}(problem::T, key, val, assignment::Dict)
    println("key: ", typeof(key), " | val: ", typeof(val));
    assignment[key] = val;
    problem.nassigns = problem.nassigns + 1;
    nothing;
end

function unassign{T <: AbstractCSP}(problem::T, key, assignment::Dict)
    println("key: ", typeof(key));
    if (haskey(assignment, key))
        delete!(assignment, key);
    end
    nothing;
end

function nconflicts{T <: AbstractCSP}(problem::T, var, val, assignment::Dict)
    return count(
                (function(second_var, ; relevant_problem::CSP=problem, first_var=var, relevant_val=val, dict::Dict=assignment)
                    return (haskey(dict, second_var) &&
                        !(relevant_problem.constraints(first_var, relevant_val, second_var, dict[second_var])));
                end),
                problem.neighbors[var]);
end

function display{T <: AbstractCSP}(problem::T, assignment::Dict)
    println("CSP: ", problem, " with assignment: ", assignment);
    nothing;
end

function actions{T <: AbstractCSP}(problem::T, state::Tuple)
    if (length(state) == length(problem.vars))
        return [];
    else
        let
            local assignment = Dict(state);
            local var = problem.vars[findfirst((function(e)
                                        return haskey(assignment, e);
                                    end), problem.vars)];
            return collect((var, val) for val in problem.domains[var]
                            if nconflicts(problem, var, val, assignment) == 0);
        end
    end
end

function result{T <: AbstractCSP}(problem::T, state::Tuple, action::Tuple)
    return (state..., action);
end

function goal_test{T <: AbstractCSP}(problem::T, state::Tuple)
    let
        local assignment = Dict(state);
        return (length(assignment) == length(problem.vars) &&
                every((function(element, ; prob::CSP=problem)
                            return nconflicts(prob, element, assignment[element], assignment) == 0;
                        end)
                        ,
                        problem.vars));
    end
end

function support_pruning{T <: AbstractCSP}(problem::T)
    if (isnull(problem.current_domains))
        problem.current_domains = Dict(collect(Pair(key, collect(problem.domains[key])) for key in problem.vars))
    end
    nothing;
end

function suppose{T <: AbstractCSP}(problem::T, var, val)
    support_pruning(problem);
    local removals::AbstractVector = collect(Pair(var, a) for a in problem.current_domains[var]
                                            if (a != val));
    problem.current_domains[var] = [val];
    return removals;
end

function prune{T <: AbstractCSP}(problem::T, var, value, removals)
    local list::AbstractVector = problem.current_domains[var];
    local index::Int64 = 0;
    for (i, element) in enumerate(list)
        if (element == value)
            index = i;
            break;
        end
    end
    if (index != 0)
        deleteat!(problem.current_domains[var], index)
    end
    if (!(typeof(removals) <: Void))
        push!(removals, Pair(var, value));
    end
    nothing;
end

function choices{T <: AbstractCSP}(problem::T, var)
    if (!isnull(problem.current_domains))
        return get(problem.current_domains)[var];
    else
        return problem.domains[vars];
    end
end

function infer_assignment{T <: AbstractCSP}(problem::T)
    support_pruning(problem);
    return Dict(collect(Pair(v, get(problem.current_domains)[v][1])
                        for v in problem.vars
                            if (1 == length(get(problem.current_domains)[v]))));
end

function restore{T <: AbstractCSP}(problem::T, removals::AbstractVector)
    for (key, val) in removals
        push!(get(problem.current_domains)[key], val);
    end
    nothing;
end

function conflicted_variables{T <: AbstractCSP}(problem::T, current_assignment::Dict)
    return collect(var for var in problem.vars
                    if (nconflicts(problem, var, current_assignment[var], current_assignment) > 0));
end

"""
    AC3(problem)

Apply the arc-consitency algorithm AC-3 (Fig 6.3) to the given contraint satisfaction problem.
Return a boolean indicating whether every arc in the problem is arc-consistent.
"""
function AC3{T <: AbstractCSP}(problem::T; queue::Union{Void, AbstractVector}=nothing, removals::Union{Void, AbstractVector}=nothing)
    if (typeof(queue) <: Void)
        queue = collect((X_i, X_k) for X_i in problem.vars for X_k in problem.neighbors[X_i]);
    end
    support_pruning(problem);
    while (length(queue) != 0)
        local X_i, X_j = shift!(queue); #Remove the first item from queue
        if (revise(problem, X_i, X_j, removals))
            if (!haskey(problem.current_domains, X_i))
                return false;
            end
            for X_k in problem.neighbors[X_i]
                if (X_k != X_i)
                    push!(queue, (X_k, X_i));
                end
            end
        end
    end
    return true;
end

function revise{T <: AbstractCSP}(problem::T, X_i, X_j, removals::Union{Void, AbstractVector})
    local revised::Bool = false;
    for x in deepcopy(problem.current_domains[X_i])
        if (all((function(y)
                    return !constraints(problem, X_i, x, X_j, y);
                end),
                problem.current_domains[X_j]))
            prune(problem, X_i, x, removals);
            revised = true;
        end
    end
    return revised;
end

