
import Base: get, getindex, getkey,
            deepcopy, copy, haskey, in, display;

export ConstantFunctionDict, CSPDict, CSP,
    get, getkey, getindex, deepcopy, copy, haskey, in,
    assign, unassign, nconflicts, display, infer_assignment,
    MapColoringCSP, backtracking_search, parse_neighbors,
    AC3, first_unassigned_variable, minimum_remaining_values,
    num_legal_values, unordered_domain_values, least_constraining_values,
    no_inference, forward_checking, maintain_arc_consistency,
    min_conflicts, tree_csp_solver, topological_sort;

#Constraint Satisfaction Problems (CSP)

type ConstantFunctionDict{V}
    value::V

    function ConstantFunctionDict{V}(val::V)
        return new(val);
    end
end

ConstantFunctionDict(val) = ConstantFunctionDict{typeof(val)}(val);

copy(cfd::ConstantFunctionDict) = ConstantFunctionDict{typeof(cfd.value)}(cfd.value);

deepcopy(cfd::ConstantFunctionDict) = ConstantFunctionDict{typeof(cfd.value)}(deepcopy(cfd.value));

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
        return new(vars, domains, neighbors, constraints, initial, Nullable{Dict}(current_domains), nassigns)
    end
end

function assign{T <: AbstractCSP}(problem::T, key::String, val::String, assignment::Dict)
    assignment[key] = val;
    problem.nassigns = problem.nassigns + 1;
    nothing;
end

function unassign{T <: AbstractCSP}(problem::T, key::String, assignment::Dict)
    if (haskey(assignment, key))
        delete!(assignment, key);
    end
    nothing;
end

function nconflicts{T <: AbstractCSP}(problem::T, var::String, val::String, assignment::Dict)
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
                                        return !haskey(assignment, e);
                                    end), problem.vars)];
            return collect((var, val) for val in problem.domains[var]
                            if nconflicts(problem, var, val, assignment) == 0);
        end
    end
end

function get_result{T <: AbstractCSP}(problem::T, state::Tuple, action::Tuple)
    return (state..., action);
end

function goal_test{T <: AbstractCSP}(problem::T, state::Tuple)
    let
        local assignment = Dict(state);
        return (length(assignment) == length(problem.vars) &&
                all((function(element, ; prob::CSP=problem)
                            return nconflicts(prob, element, assignment[element], assignment) == 0;
                        end)
                        ,
                        problem.vars));
    end
end

function goal_test{T <: AbstractCSP}(problem::T, state::Dict)
    let
        local assignment = deepcopy(state);
        return (length(assignment) == length(problem.vars) &&
                all((function(element, ; prob::CSP=problem)
                            return nconflicts(prob, element, assignment[element], assignment) == 0;
                        end)
                        ,
                        problem.vars));
    end
end

function path_cost{T <: AbstractCSP}(problem::T, cost::Float64, state1::Tuple, action::Tuple, state2::Tuple)
    return cost + 1;
end

function support_pruning{T <: AbstractCSP}(problem::T)
    if (isnull(problem.current_domains))
        problem.current_domains = Nullable(Dict(collect(Pair(key, collect(problem.domains[key])) for key in problem.vars)));
    end
    nothing;
end

function suppose{T <: AbstractCSP}(problem::T, var, val)
    support_pruning(problem);
    local removals::AbstractVector = collect(Pair(var, a) for a in get(problem.current_domains)[var]
                                            if (a != val));
    get(problem.current_domains)[var] = [val];
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
        deleteat!(get(problem.current_domains)[var], index)
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
        return problem.domains[var];
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
        local X = shift!(queue);    #Remove the first item from queue
        local X_i::String = getindex(X, 1);
        local X_j::String = getindex(X, 2);
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
    for x in deepcopy(get(problem.current_domains)[X_i])
        if (all((function(y)
                    return !problem.constraints(X_i, x, X_j, y);
                end),
                get(problem.current_domains)[X_j]))
            prune(problem, X_i, x, removals);
            revised = true;
        end
    end
    return revised;
end

function first_unassigned_variable{T <: AbstractCSP}(problem::T, assignment::Dict)
    return getindex(problem.vars, findfirst((function(var)
                        return !haskey(assignment, var);
                    end),
                    problem.vars));
end

function minimum_remaining_values{T <: AbstractCSP}(problem::T, assignment::Dict)
    return argmin_random_tie(collect(v for v in problem.vars if !haskey(assignment, v)),
                            (function(var)
                                return num_legal_values(problem, var, assignment);
                            end));
end

function num_legal_values{T <: AbstractCSP}(problem::T, var, assignment::Dict)
    if (!isnull(problem.current_domains))
        return length(get(problem.current_domains)[var]);
    else
        return count((function(val)
                        return nconflicts(problem, var, val, assignment);
                    end),
                    problem.domains[vars]);
    end
end

function unordered_domain_values{T <: AbstractCSP}(problem::T, var, assignment::Dict)
    return choices(problem, var);
end

function least_constraining_values{T <: AbstractCSP}(problem::T, var, assignment::Dict)
    return sort!(deepcopy(choices(problem, var)),
                lt=(function(val)
                       return nconflicts(problem, var, val, assignment);                                 
                    end));
end

function no_inference{T <: AbstractCSP}(problem::T, var, value, assignment::Dict, removals::Union{Void, AbstractVector})
    return true;
end

function forward_checking{T <: AbstractCSP}(problem::T, var, value, assignment::Dict, removals::Union{Void, AbstractVector})
    for B in problem.neighbors[var]
        if (!haskey(assignment, B))
            for b in deepcopy(get(problem.current_domains)[B])
                if (!problem.constraints(var, value, B, b))
                    prune(problem, B, b, removals);
                end
            end
            if (length(get(problem.current_domains)[B]) == 0)
                return false;
            end
        end
    end
    return true;
end

function maintain_arc_consistency{T <: AbstractCSP}(problem::T, var, value, assignment::Dict, removals::Union{Void, AbstractVector})
    return AC3(problem, queue=collect((X, var) for X in problem.neighbors[var]), removals=removals);
end

function backtrack{T <: AbstractCSP}(problem::T, assignment::Dict;
                                    select_unassigned_variable::Function=first_unassigned_variable,
                                    order_domain_values::Function=unordered_domain_values,
                                    inference::Function=no_inference)
    if (length(assignment) == length(problem.vars))
        return assignment;
    end
    local var = select_unassigned_variable(problem, assignment);
    for value in order_domain_values(problem, var, assignment)
        if (nconflicts(problem, var, value, assignment) == 0)
            assign(problem, var, value, assignment);
            removals = suppose(problem, var, value);
            if (inference(problem, var, value, assignment, removals))
                result = backtrack(problem, assignment,
                                    select_unassigned_variable=select_unassigned_variable,
                                    order_domain_values=order_domain_values,
                                    inference=inference);
                if (!(typeof(result) <: Void))
                    return result;
                end
            end
            restore(problem, removals);
        end
    end
    unassign(problem, var, assignment);
    return nothing;
end


function backtracking_search{T <: AbstractCSP}(problem::T;
                                                select_unassigned_variable::Function=first_unassigned_variable,
                                                order_domain_values::Function=unordered_domain_values,
                                                inference::Function=no_inference)
    local result = backtrack(problem, Dict(),
                            select_unassigned_variable=select_unassigned_variable,
                                    order_domain_values=order_domain_values,
                                    inference=inference);
    if (!(typeof(result) <: Void || goal_test(problem, result)))
        error("BacktrackingSearchError: Unexpected result!")
    end
    return result;
end

function parse_neighbors(neighbors::String; vars::AbstractVector=[])
    local new_dict = Dict();
    for var in vars
        new_dict[var] = [];
    end
    local specs::AbstractVector = collect(map(String, split(spec, [':'])) for spec in split(neighbors, [';']));
    for (A, A_n) in specs
        A = strip(A);
        if (!haskey(new_dict, A))
            new_dict[A] = [];
        end
        for B in map(String, split(A_n))
            push!(new_dict[A], B);
            if (!haskey(new_dict, B))
                new_dict[B] = [];
            end
            push!(new_dict[B], A);
        end
    end
    return new_dict;
end

function different_values_constraint(A::String, a::String, B::String, b::String)
    return (a != b);
end

function min_conflicts_value{T <: AbstractCSP}(problem::T, var::String, current_assignment::Dict)
    return argmin_random_tie(problem.domains[var],
                            (function(val)
                                return nconflicts(problem, var, val, current_assignment);
                            end));
end

function min_conflicts{T <: AbstractCSP}(problem::T; max_steps::Int64=100000)
    local current::Dict = Dict();
    for var in problem.vars
        val = min_conflicts_value(problem, var, current);
        assign(problem, var, val, current);
    end
    for i in 1:max_steps
        local conflicted::AbstractVector = conflicted_variables(problem, current);
        if (length(conflicted) == 0)
            return current;
        end
        local var = rand(RandomDeviceInstance, conflicted);
        local val = min_conflicts_value(problem, var, current);
        assign(problem, var, val, current);
    end
    return nothing;
end

function tree_csp_solver{T <: AbstractCSP}(problem::T)
    local num_of_vars = length(problem.vars);
    local assignment = Dict();
    local root = problem.vars[1];
    X::AbstractVector, parent_dict::Dict = topological_sort(problem, root);
    for X_j in reverse(X)
        if (!make_arc_consistent(problem, parent_dict, X_j))
            return nothing;
        end
    end
    for X_i in X
        if (length(get(problem.current_domains)[X_i]) == 0)
            return nothing;
        end
        assignment[X_i] = get(problem.current_domains)[X_i][1];
    end
    return assignment;
end

"""
    topological_sort(problem, root)

Return a topological sorted AbstractVector and a dictionary of vertices and their parents as keys and values.
The topological depth first search sort first visits the 'root' vertex, traversing the graph by recursively
calling topological_sort_postorder_dfs on vertices returned by problem.neighbors[vertex].

The dictionary returned does not have keys (vertices) that do not have a parent vertex.
"""
function topological_sort{T <: AbstractCSP}(problem::T, root::String)
    local sorted_nodes = [];
    local parents = Dict();
    local visited = Dict();
    for v in problem.vars
        visited[v] = false;
    end
    local neighbors = problem.neighbors;
    topological_sort_postorder_dfs(root, visited, neighbors, parents, sorted_nodes, nothing);
    return sorted_nodes, parents;
end

function topological_sort_postorder_dfs(vertex::String,
                                        visited::Dict,
                                        neighbors::CSPDict,
                                        parents::Dict,
                                        sorted_vertices::AbstractVector,
                                        parent::Union{Void, String})
    visited[vertex] = true;
    for w in neighbors[vertex]
        if (!visited[w])
            topological_sort_postorder_dfs(w, visited, neighbors, parents, sorted_vertices, vertex);
        end
    end
    insert!(sorted_vertices, 1, vertex);
    if (!(typeof(parent) <: Void))
        parents[vertex] = parent;
    end
    nothing;
end

function make_arc_consistent{T <: AbstractCSP}(problem::T, parent::Dict, X_j)
    println("make_arc_consistent() is not implemented yet for ", typeof(problem), "!");
    nothing;
end

function MapColoringCSP(colors::AbstractVector, neighbors::String)
    local parsed_neighbors = parse_neighbors(neighbors);
    return CSP(collect(keys(parsed_neighbors)), CSPDict(ConstantFunctionDict(colors)), CSPDict(parsed_neighbors), different_values_constraint);
end

function MapColoringCSP(colors::AbstractVector, neighbors::Dict)
    return CSP(collect(keys(neighbors)), CSPDict(ConstantFunctionDict(colors)), CSPDict(neighbors), different_values_constraint);
end

australia_csp = MapColoringCSP(["R", "G", "B"], "SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ");

usa_csp = MapColoringCSP(["R", "G", "B", "Y"],
                        "WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                        UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX;
                        ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                        TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                        LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                        MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                        PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                        NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                        HI: ; AK: ");

france_csp = MapColoringCSP(["R", "G", "B", "Y"],
                            "AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR");

