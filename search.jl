import Base: ==, expand, length, in;

export Problem, InstrumentedProblem,
        actions, get_result, goal_test, path_cost, value,
        format_instrumented_results,
        Node, expand, child_node, solution, path, ==,
        GAState, mate, mutate,
        tree_search, graph_search,
        breadth_first_tree_search, depth_first_tree_search, depth_first_graph_search,
        breadth_first_search, best_first_graph_search, uniform_cost_search,
        recursive_dls, depth_limited_search, iterative_deepening_search,
        greedy_best_first_graph_search,
        Graph, make_undirected, connect_nodes, get_linked_nodes, get_nodes,
        UndirectedGraph, RandomGraph,
        GraphProblem,
        astar_search, recursive_best_first_search,
        hill_climbing, exp_schedule, simulated_annealing,
        or_search, and_search, and_or_graph_search,
        genetic_search, genetic_algorithm,
        NQueensProblem, conflict, conflicted,
        random_boggle, print_boggle, boggle_neighbors, int_sqrt,
        WordList, lookup, length, in,
        BoggleFinder, set_board, find, words, score,
        boggle_hill_climbing, mutate_boggle,
        execute_searcher, compare_searchers, beautify_node;

#=

    Problem is a abstract problem that contains a initial state and goal state.

=#
type Problem <: AbstractProblem
    initial::String
    goal::Nullable{String}

    function Problem(initial_state::String; goal_state::Union{Void, String}=nothing)
        return new(initial_state, Nullable{String}(goal_state));
    end
end

function actions{T <: AbstractProblem}(ap::T, state::String)
    println("actions() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function get_result{T <: AbstractProblem}(ap::T, state::String, action::String)
    println("get_result() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function goal_test{T <: AbstractProblem}(ap::T, state::String)
    return ap.goal == state;
end

function path_cost{T <: AbstractProblem}(ap::T, cost::Float64, state1::String, action::String, state2::String)
    return cost + 1;
end

function path_cost{T <: AbstractProblem}(ap::T, cost::Float64, state1::AbstractVector, action::Int64, state2::AbstractVector)
    return cost + 1;
end

function value{T <: AbstractProblem}(ap::T, state::String)
    println("value() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

#=

    InstrumentedProblem is a AbstractProblem implementation that wraps another AbstractProblem

    implementation and tracks the number of function calls made. This problem is used in

    compare_searchers() and execute_searcher().

=#
type InstrumentedProblem <: AbstractProblem
    problem::AbstractProblem
    actions::Int64
    results::Int64
    goal_tests::Int64
    found::Nullable

    function InstrumentedProblem{T <: AbstractProblem}(ap::T)
        return new(ap, Int64(0), Int64(0), Int64(0), Nullable(nothing));
    end
end

function actions(ap::InstrumentedProblem, state::AbstractVector)
    ap.actions = ap.actions + 1;
    return actions(ap.problem, state);
end

function actions(ap::InstrumentedProblem, state::String)
    ap.actions = ap.actions + 1;
    return actions(ap.problem, state);
end

function get_result(ap::InstrumentedProblem, state::String, action::String)
    ap.results = ap.results + 1;
    return get_result(ap.problem, state, action);
end

function get_result(ap::InstrumentedProblem, state::AbstractVector, action::Int64)
    ap.results = ap.results + 1;
    return get_result(ap.problem, state, action);
end

function goal_test(ap::InstrumentedProblem, state::String)
    ap.goal_tests = ap.goal_tests + 1;
    local result::Bool = goal_test(ap.problem, state);
    if (result)
        ap.found = Nullable(state);
    end
    return result;
end

function goal_test(ap::InstrumentedProblem, state::AbstractVector)
    ap.goal_tests = ap.goal_tests + 1;
    local result::Bool = goal_test(ap.problem, state);
    if (result)
        ap.found = Nullable(state);
    end
    return result;
end

function path_cost(ap::InstrumentedProblem, cost::Float64, state1::String, action::String, state2::String)
    return path_cost(ap.problem, cost, state1, action, state2);
end

function path_cost(ap::InstrumentedProblem, cost::Float64, state1::AbstractVector, action::Int64, state2::AbstractVector)
    return path_cost(ap.problem, cost, state1, action, state2);
end

function value(ap::InstrumentedProblem, state::String)
    return value(ap.problem, state);
end

function value(ap::InstrumentedProblem, state::AbstractVector)
    return value(ap.problem, state);
end

function format_instrumented_results(ap::InstrumentedProblem)
    return @sprintf("<%4d/%4d/%4d/%s>", ap.actions, ap.goal_tests, ap.results, string(get(ap.found)));
end

#A node should not exist without a state.
type Node{T}
    state::T
    path_cost::Float64
    depth::UInt32
    action::Nullable
    parent::Nullable{Node}
    f::Float64

    function Node{T}(state::T; parent::Union{Void, Node}=nothing, action::Union{Void, String, Int64, Tuple}=nothing, path_cost::Float64=0.0, f::Union{Void, Float64}=nothing)
        nn = new(state, path_cost, UInt32(0), Nullable(action), Nullable{Node}(parent));
        if (typeof(parent) <: Node)
            nn.depth = UInt32(parent.depth + 1);
        end
        if (typeof(f) <: Float64)
            nn.f = f;
        end
        return nn;
    end
end

function expand{T <: AbstractProblem}(n::Node, ap::T)
    return collect(child_node(n, ap, act) for act in actions(ap, n.state));
end

function child_node{T <: AbstractProblem}(n::Node, ap::T, action::String)
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

function child_node{T <: AbstractProblem}(n::Node, ap::T, action::Int64)
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

function child_node{T <: AbstractProblem}(n::Node, ap::T, action::Tuple)
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

function solution(n::Node)
    local path_sequence = path(n);
    return [get(node.action) for node in path_sequence[2:length(path_sequence)]];
end

function path(n::Node)
    local node = n;
    local path_back = [];
    while true
        push!(path_back, node);
        if (!isnull(node.parent))
            node = get(node.parent);
        else     #the root node does not have a parent node
            break;
        end
    end
    path_back = reverse(path_back);
    return path_back;
end

function ==(n1::Node, n2::Node)
    if (typeof(n1.state) == typeof(n2.state))
        return (n1.state == n2.state);
    else
        if (typeof(n1.state) <: AbstractVector && typeof(n2.state) <: AbstractVector)
            local n1a::AbstractVector;
            local n2a::AbstractVector;
            if (eltype(typeof(n1.state)) <: Nullable)
                n1a = map(get, n1.state);
            else
                n1a = n1.state;
            end
            if (eltype(typeof(n2.state)) <: Nullable)
                n2a = map(get, n2.state);
            else
                n2a = n2.state;
            end
            return (n1a == n2a);
        else
            return (n1.state == n2.state);
        end
    end
end

#=

    SimpleProblemSolvingAgentProgram is a abstract problem solving agent (Fig. 3.1).

=#
type SimpleProblemSolvingAgentProgram <: AgentProgram
    state::Nullable{String}
    goal::Nullable{String}
    seq::Array{String, 1}
    problem::Nullable{Problem}

    function SimpleProblemSolvingAgentProgram(;initial_state::Union{Void, String}=nothing)
        return new(Nullable{String}(initial_state), Nullable{String}(), Array{String, 1}(), Nullable{Problem}());
    end
end

function execute(spsap::SimpleProblemSolvingAgentProgram, percept::Tuple{Any, Any})
    spsap.state = update_state(spsap, spsap.state, percept);
    if (length(spsap.seq) == 0)
        spsap.goal = formulate_problem(spsap, spsap.state);
        spsap.problem = forumate_problem(spsap, spsap.state, spsap.goal);
        spsap.seq = search(spsap, spsap.problem);
        if (length(spsap.seq) == 0)
            return Void;
        end
    end
    local action = shift!(spsap.seq);
    return action;
end

function update_state(spsap::SimpleProblemSolvingAgentProgram, state::String, percept::Tuple{Any, Any})
    println("update_state() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function formulate_goal(spsap::SimpleProblemSolvingAgentProgram, state::String)
    println("formulate_goal() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function formulate_problem(spsap::SimpleProblemSolvingAgentProgram, state::String, goal::String)
    println("formulate_problem() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function search{T <: AbstractProblem}(spsap::SimpleProblemSolvingAgentProgram, problem::T)
    println("search() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

type GAState
    genes::Array{Any, 1}

    function GAState(genes::Array{Any, 1})
        return new(Array{Any,1}(deepcopy(genes)));
    end
end

function mate{T <: GAState}(ga_state::T, other::T)
    local c = rand(RandomDeviceInstance, range(1, length(ga_state.genes)));
    local new_ga_state = deepcopy(ga_state[1:c]);
    for element in other.genes[(c + 1):length(other.genes)]
        push!(new_ga_state, element);
    end
    return new_ga_state;
end

function mutate{T <: GAState}(ga_state::T)
    println("mutate() is not implemented yet for ", typeof(ga_state), "!");
    nothing;
end

"""
    tree_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)

Search the given problem by using the general tree search algorithm (Fig. 3.7) and return the node solution.
"""
function tree_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)
    push!(frontier, Node{typeof(problem.initial)}(problem.initial));
    while (length(frontier) != 0)
        local node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        extend!(frontier, expand(node, problem));
    end
    return nothing;
end

function tree_search{T <: Queue}(problem::InstrumentedProblem, frontier::T)
    push!(frontier, Node{typeof(problem.problem.initial)}(problem.problem.initial));
    while (length(frontier) != 0)
        local node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        extend!(frontier, expand(node, problem));
    end
    return nothing;
end

"""
    graph_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)

Search the given problem by using the general graph search algorithm (Fig. 3.7) and return the node solution.

The uniform cost algorithm (Fig. 3.14) should be used when the frontier is a priority queue.
"""
function graph_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)
    local explored::Set;
    if (typeof(problem.initial) <: Tuple)
        explored = Set{NTuple}();
    else
        explored = Set{typeof(problem.initial)}();
    end
    push!(frontier, Node{typeof(problem.initial)}(problem.initial));
    while (length(frontier) != 0)
        local node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        push!(explored, node.state);
        extend!(frontier, collect(child_node for child_node in expand(node, problem)
                                if (!(child_node.state in explored) && !(child_node in frontier))));
    end
    return nothing;
end

function graph_search{T <: Queue}(problem::InstrumentedProblem, frontier::T)
    local explored::Set;
    if (typeof(problem.problem.initial) <: Tuple)
        explored = Set{NTuple}();
    else
        explored = Set{typeof(problem.problem.initial)}();
    end
    push!(frontier, Node{typeof(problem.problem.initial)}(problem.problem.initial));
    while (length(frontier) != 0)
        local node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        push!(explored, node.state);
        extend!(frontier, collect(child_node for child_node in expand(node, problem)
                                if (!(child_node.state in explored) && !(child_node in frontier))));
    end
    return nothing;
end

function breadth_first_tree_search{T <: AbstractProblem}(problem::T)
    return tree_search(problem, FIFOQueue());
end

function depth_first_tree_search{T <: AbstractProblem}(problem::T)
    return tree_search(problem, Stack());
end

function depth_first_graph_search{T <: AbstractProblem}(problem::T)
    return graph_search(problem, Stack());
end

function breadth_first_search{T <: AbstractProblem}(problem::T)
    local node = Node{typeof(problem.initial)}(problem.initial);
    if (goal_test(problem, node.state))
        return node;
    end
    local frontier = FIFOQueue();
    push!(frontier, node);
    local explored = Set{String}();
    while (length(frontier) != 0)
        node = pop!(frontier);
        push!(explored, node.state);
        for child_node in expand(node, problem)
            if (!(child_node.state in explored) && !(child_node in frontier))
                if (goal_test(problem, child_node.state))
                    return child_node;
                end
                push!(frontier, child_node);
            end
        end
    end
    return nothing;
end

function breadth_first_search(problem::InstrumentedProblem)
    local node = Node{typeof(problem.problem.initial)}(problem.problem.initial);
    if (goal_test(problem, node.state))
        return node;
    end
    local frontier = FIFOQueue();
    push!(frontier, node);
    local explored = Set{String}();
    while (length(frontier) != 0)
        node = pop!(frontier);
        push!(explored, node.state);
        for child_node in expand(node, problem)
            if (!(child_node.state in explored) && !(child_node in frontier))
                if (goal_test(problem, child_node.state))
                    return child_node;
                end
                push!(frontier, child_node);
            end
        end
    end
    return nothing;
end

function best_first_graph_search{T <: AbstractProblem}(problem::T, f::Function)
    #local mf = MemoizedFunction(f);
    local node = Node{typeof(problem.initial)}(problem.initial);
    if (goal_test(problem, node.state))
        return node;
    end
    local frontier = PQueue();
    #push!(frontier, node, mf);
    push!(frontier, node, f);
    local explored = Set{String}();
    while (length(frontier) != 0)
        node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        push!(explored, node.state);
        for child_node in expand(node, problem)
            if (!(child_node.state in explored) &&
                !(child_node in collect(getindex(x, 2) for x in frontier.array)))
                #push!(frontier, child_node, mf);
                push!(frontier, child_node, f);
            elseif (child_node in [getindex(x, 2) for x in frontier.array])
            #Recall that Nodes can share the same state and different values for other fields.
                local existing_node = pop!(collect(getindex(x, 2)
                                                    for x in frontier.array
                                                    if (getindex(x, 2) == child_node)));
                #if (eval_memoized_function(mf, child_node) < eval_memoized_function(mf, existing_node))
                if (f(child_node) < f(existing_node))
                    delete!(frontier, existing_node);
                    #push!(frontier, child_node, mf);
                    push!(frontier, child_node, f);
                end
            end
        end
    end
    return nothing;
end


"""
    uniform_cost_search{T <: AbstractProblem}(problem::T)

Search the given problem by using the uniform cost algorithm (Fig. 3.14) and return the node solution.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function uniform_cost_search{T <: AbstractProblem}(problem::T)
    return best_first_graph_search(problem, (function(n::Node)return n.path_cost;end));
end

function recursive_dls{T <: AbstractProblem}(node::Node, problem::T, limit::Int64)
    if (goal_test(problem, node.state))
        return node;
    elseif (node.depth == limit)
        return "cutoff";
    else
        local cutoff_occurred = false;
        for child_node in expand(node, problem)
            local result = recursive_dls(child_node, problem, limit);
            if (result == "cutoff")
                cutoff_occurred = true;
            elseif (!(typeof(result) <: Void))
                return result;
            end
        end
        return if_(cutoff_occurred, "cutoff", nothing);
    end
end;

"""
    depth_limited_search{T <: AbstractProblem}(problem::T; limit::Int64)

Search the given problem by using the depth limited tree search algorithm (Fig. 3.17)
and return the node solution if a solution was found. Otherwise, this function returns 'nothing'.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function depth_limited_search{T <: AbstractProblem}(problem::T; limit::Int64=50)
    return recursive_dls(Node{typeof(problem.initial)}(problem.initial), problem, limit);
end

function depth_limited_search(problem::InstrumentedProblem; limit::Int64=50)
    return recursive_dls(Node{typeof(problem.problem.initial)}(problem.problem.initial), problem, limit);
end

"""
    iterative_deepening_search{T <: AbstractProblem}(problem::T)

Search the given problem by using the iterative deepening search algorithm (Fig. 3.18)
and return the node solution if a solution was found. Otherwise, this function returns 'nothing'.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function iterative_deepening_search{T <: AbstractProblem}(problem::T)
    for depth in 1:typemax(Int64)
        local result = depth_limited_search(problem, limit=depth)
        if (result != "cutoff")
            return result;
        end
    end
    return nothing;
end

const greedy_best_first_graph_search = best_first_graph_search;

type Graph{N}
    dict::Dict{N, Any}
    locations::Dict{N, Tuple{Any, Any}}
    directed::Bool

    function Graph{N}(;dict::Union{Void, Dict{N, }}=nothing, locations::Union{Void, Dict{N, Tuple{Any, Any}}}=nothing, directed::Bool=true)
        local ng::Graph;
        if ((typeof(dict) <: Void) && (typeof(locations) <: Void))
            ng = new(Dict{Any, Any}(), Dict{Any, Tuple{Any, Any}}(), Bool(directed));
        else
            ng = new(Dict{eltype(dict.keys), Any}(dict), Dict{eltype(locations.keys), Tuple{Any, Any}}(locations), Bool(directed));
        end
        if (!ng.directed)
            make_undirected(ng);
        end
        return ng;
    end

    function Graph{N}(graph::Graph{N})
        return new(Dict{Any, Any}(graph.dict), Dict{String, Tuple{Any, Any}}(graph.locations), Bool(graph.directed));
    end
end

function make_undirected(graph::Graph)
    for location_A in keys(graph.dict)
        for (location_B, d) in graph.dict[location_A]
            connect_nodes(graph, location_B, location_A, distance=d);
        end
    end
end

function connect_nodes{N}(graph::Graph{N}, A::N, B::N; distance::Int64=Int64(1))
    get!(graph.dict, A, Dict{String, Int64}())[B]=distance;
    if (!graph.directed)
        get!(graph.dict, B, Dict{String, Int64}())[A]=distance;
    end
    nothing;
end

function get_linked_nodes{N}(graph::Graph{N}, a::N; b::Union{Void, N}=nothing)
    local linked = get!(graph.dict, a, Dict{Any, Any}());
    if (typeof(b) <: Void)
        return linked;
    else
        return get(linked, b, nothing);
    end
end

function get_nodes(graph::Graph)
    return collect(keys(graph.dict));
end

function UndirectedGraph{T <: Any}(dict::Dict{T, }, locations::Dict{T, Tuple{Any, Any}})
    return Graph{eltype(dict.keys)}(dict=dict, locations=locations, directed=false);
end

function UndirectedGraph()
    return Graph{Any}(directed=false);
end

function RandomGraph(;nodes::Range=1:10,
                    min_links::Int64=2,
                    width::Int64=400,
                    height::Int64=300,
                    curvature::Function=(function()
                                            return (0.4*rand(RandomDeviceInstance)) + 1.1;
                                        end))
    local g = UndirectedGraph();
    for node in nodes
        g.locations[node] = Tuple((rand(RandomDeviceInstance, 1:width), rand(RandomDeviceInstance, 1:height)));
    end
    for i in 1:min_link
        for node in nodes
            if (get_linked_nodes(g, node) < min_links)
                local here = g.locations[node];
                local neighbor = argmin(nodes, (function(n, ; graph::Graph=g, current_node::Node=node, current_location::Tuple=here)
                                                    if (n == current_node || get_linked_nodes(graph, current_node, n) != nothing)
                                                        return Inf;
                                                    end
                                                    return distance(g.locations[n], current_location);
                                                end));
                local d = distance(g.locations[neighbor], here) * curvature();
                connect(g, node, neighbor, Int64(floor(d)));
            end
        end
    end
    return g;
end


type GraphProblem <: AbstractProblem
    initial::String
    goal::String
    graph::Graph
    h::MemoizedFunction


    function GraphProblem(initial_state::String, goal_state::String, graph::Graph)
        return new(initial_state, goal_state, Graph(graph), MemoizedFunction(initial_to_goal_distance));
    end
end

function actions(gp::GraphProblem, loc::String)
    return collect(keys(get_linked_nodes(gp.graph,loc)));
end

function get_result(gp::GraphProblem, state::String, action::String)
    return action;
end

function path_cost(gp::GraphProblem, current_cost::Float64, location_A::String, action::String, location_B::String)
    local AB_distance::Float64;
    if (haskey(gp.graph.dict, location_A) && haskey(gp.graph.dict[location_A], location_B))
        AB_distance= Float64(get_linked_nodes(gp.graph,location_A, b=location_B));
    else
        AB_distance = Float64(Inf);
    end
    return current_cost + AB_distance;
end

"""
    initial_to_goal_distance(gp::GraphProblem, n::Node)

Compute the straight line distance between the initial state and goal state.
"""
function initial_to_goal_distance(gp::GraphProblem, n::Node)
    local locations = gp.graph.locations;
    if (isempty(locations))
        return Inf;
    else
        return Float64(floor(distance(locations[n.state], locations[gp.goal])));
    end
end

function initial_to_goal_distance(gp::InstrumentedProblem, n::Node)
    local locations = gp.problem.graph.locations;
    if (isempty(locations))
        return Inf;
    else
        return Float64(floor(distance(locations[n.state], locations[gp.problem.goal])));
    end
end

function astar_search(problem::GraphProblem; h::Union{Void, Function}=nothing)
    #local mh::MemoizedFunction; #memoized h(n) function
    local mh::Function;
    if (!(typeof(h) <: Void))
        mh = h;
    else
        mh = problem.h.f;
    end
    return best_first_graph_search(problem,
                                    (function(node::Node; h::Function=mh, prob::GraphProblem=problem)
                                        #return node.path_cost + eval_memoized_function(h, prob, node);end));
                                        return node.path_cost + h(prob, node);end));
end

"""
    RBFS{T1 <: AbstractProblem, T2 <: Node}(problem::T1, node::T2, flmt::Float64, h::MemoizedFunction)

Recursively calls RBFS() with a new 'flmt' value and returns its solution to recursive_best_first_search().
"""
function RBFS{T1 <: AbstractProblem, T2 <: Node}(problem::T1, node::T2, flmt::Float64, h::MemoizedFunction)
    if (goal_test(problem, node.state))
        return Nullable{Node}(node), 0.0;
    end
    local successors = expand(node, problem);
    if (length(successors) == 0);
        return Nullable{Node}(node), Inf;
    end
    for successor in successors
        successor.f = max(successor.path_cost + eval_memoized_function(h, problem, successor), node.f);
    end
    while (true)
        sort!(successors, lt=(function(n1::Node, n2::Node)return isless(n1.f, n2.f);end));
        local best::Node = successors[1];
        if (best.f > flmt)
            return Nullable{Node}(), best.f;
        end
        local alternative::Float64;
        if (length(successors) > 1)
            alternative = successors[1].f;
        else
            alternative = Inf;
        end
        result, best.f = RBFS(problem, best, min(flmt, alternative), h);
        if (!isnull(result))
            return result, best.f;
        end
    end
end

"""
    recursive_best_first_search{T <: AbstractProblem}(problem::T; h::Union{Void, MemoizedFunction})

Search the given problem by using the recursive best first search algorithm (Fig. 3.26)
and return the node solution.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function recursive_best_first_search{T <: AbstractProblem}(problem::T; h::Union{Void, MemoizedFunction}=nothing)
    local mh::MemoizedFunction; #memoized h(n) function
    if (!(typeof(h) <: Void))
        mh = MemoizedFunction(h);
    else
        mh = problem.h;
    end

    local node = Node{typeof(problem.initial)}(problem.initial);
    node.f = eval_memoized_function(mh, problem, node);
    result, bestf = RBFS(problem, node, Inf, mh);
    return get(result);
end

function recursive_best_first_search(problem::InstrumentedProblem; h::Union{Void, MemoizedFunction}=nothing)
    local mh::MemoizedFunction; #memoized h(n) function
    if (!(typeof(h) <: Void))
        mh = MemoizedFunction(h);
    else
        mh = problem.problem.h;
    end

    local node = Node{typeof(problem.problem.initial)}(problem.problem.initial);
    node.f = eval_memoized_function(mh, problem, node);
    result, bestf = RBFS(problem, node, Inf, mh);
    return get(result);
end

function hill_climbing{T <: AbstractProblem}(problem::T)
    local current_node = Node{typeof(problem.initial)}(problem.initial);
    while (true)
        local neighbors = expand(current_node, problem);
        if (length(neighbors) == 0)
            break;
        end
        local neighbor = argmax_random_tie(neighbors,
                                            (function(n::Node,; p::AbstractProblem=problem)
                                                return value(p, n.state);
                                            end));
        if (value(problem, neighbor.state) <= value(problem, current_node.state))
            break;
        end
        current_node = neighbor;
    end
    return current_node.state;
end

function exp_schedule(;kvar::Int64=20, delta::Float64=0.005, lmt::Int64=100)
    return (function(t::Real; k=kvar, d=delta, limit=lmt)
                return if_((t < limit), (k * exp(-d * t)), 0);
            end);
end

function simulated_annealing{T <: AbstractProblem}(problem::T; schedule=exp_schedule)
    local current_node = Node{typeof(problem.initial)}(problem.initial);
    for t in 0:(typemax(Int64) - 1)
        local temperature = schedule(t);
        if (temperature == 0)
            return current_node;
        end
        local neighbors = expand(current_node, problem);
        if (length(neighbors) == 0)
            return current_node;
        end
        local next_node = rand(RandomDeviceInstance, neighbors);
        delta_e = value(problem, next_node.state) - value(problem, current_node.state);
        if ((delta_e > 0) || (exp(delta_e/temperature) > rand(RandomDeviceInstance)))
            current_node = next_node;
        end
    end
    return nothing;
end

#=

    and_search() and or_search() are used by and_or_graph_search().

=#

function or_search{T <: AbstractProblem}(problem::T, state::AbstractVector, path::AbstractVector)
    if (goal_test(problem, state))
        return [];
    end
    if (state in path)
        return nothing;
    end
    for action in actions(problem, state)
        local plan = and_search(get_result(problem, state, action), vcat(path, [state,]));
        if (plan != nothing)
            return [action, plan];
        end
    end
    return nothing;
end

function or_search{T <: AbstractProblem}(problem::T, state::String, path::AbstractVector)
    if (goal_test(problem, state))
        return [];
    end
    if (state in path)
        return nothing;
    end
    for action in actions(problem, state)
        local plan = and_search(problem, get_result(problem, state, action), vcat(path, [state,]));
        if (plan != nothing)
            return [action, plan];
        end
    end
    return nothing;
end

function and_search{T <: AbstractVector}(problem::T, states::AbstractVector, path::AbstractVector)
    local plan = Dict{Any, Any}();
    for state in states
        plan[state] = or_search(problem, state, path);
        if (plan[state] == nothing)
            return nothing;
        end
    end
    return plan;
end

function and_or_graph_search{T <: AbstractProblem}(problem::T)
    return or_search(problem, problem.initial, []);
end

function genetic_search{T <: AbstractProblem}(problem::T; ngen::Int64=1000, pmut::Float64=0.1, n::Int64=20)
    local s = problem.initial;
    local states = [result(s, action) for action in actions(problem, s)];
    shuffle!(RandomDeviceInstance, states);
    if (length(states) < n)
        n = length(states);
    end
    return genetic_algorithm(states[1:n], value, ngen=ngen, pmut=pmut);
end

function genetic_algorithm{T <: AbstractVector}(population::T, fitness::Function; ngen::Int64=1000, pmut::Float64=0.1)
    for i in 1:ngen
        local new_population = Array{Any, 1}();
        for j in 1:length(population)
            local fitnesses = map(fitness, population);
            p1, p2 = weighted_sample_with_replacement(population, fitnesses, 2);
            local child = mate(p1, p2);
            if (rand(RandomDeviceInstance) < pmut)
                mutate(child);
            end
            push!(new_population, child);
        end
        population = new_population;
    end
    return argmax(population, fitness);
end

romania = UndirectedGraph(Dict(
                            Pair("A", Dict("Z"=>75, "S"=>140, "T"=>118)),
                            Pair("B", Dict("U"=>85, "P"=>101, "G"=>90, "F"=>211)),
                            Pair("C", Dict("D"=>120, "R"=>146, "P"=>138)),
                            Pair("D", Dict("M"=>75)),
                            Pair("E", Dict("H"=>86)),
                            Pair("F", Dict("S"=>99)),
                            Pair("H", Dict("U"=>98)),
                            Pair("I", Dict("V"=>92, "N"=>87)),
                            Pair("L", Dict("T"=>111, "M"=>70)),
                            Pair("O", Dict("Z"=>71, "S"=>151)),
                            Pair("P", Dict("R"=>97)),
                            Pair("R", Dict("S"=>80)),
                            Pair("U", Dict("V"=>142)),
                                ),
                            Dict{String, Tuple{Any, Any}}(
                                "A"=>( 91, 492), "B"=>(400, 327), "C"=>(253, 288), "D"=>(165, 299),
                                "E"=>(562, 293), "F"=>(305, 449), "G"=>(375, 270), "H"=>(534, 350),
                                "I"=>(473, 506), "L"=>(165, 379), "M"=>(168, 339), "N"=>(406, 537),
                                "O"=>(131, 571), "P"=>(320, 368), "R"=>(233, 410), "S"=>(207, 457),
                                "T"=>( 94, 410), "U"=>(456, 350), "V"=>(509, 444), "Z"=>(108, 531),
                                )
                            );

australia = UndirectedGraph(Dict(
                                Pair("T",   Dict()),
                                Pair("SA",  Dict("WA"=>1, "NT"=>1, "Q"=>1, "NSW"=>1, "V"=>1)),
                                Pair("NT",  Dict("WA"=>1, "Q"=>1)),
                                Pair("NSW", Dict("Q"=>1, "V"=>1)),
                            ),
                            Dict{String, Tuple{Any, Any}}("WA"=>(120, 24), "NT"=>(135, 20), "SA"=>(135, 30),
                                "Q"=>(145, 20), "NSW"=>(145, 32), "T"=>(145, 42), "V"=>(145, 37),
                                )
                            );

type NQueensProblem <: AbstractProblem
    N::Int64
    initial::Array{Nullable{Int64}, 1}

    function NQueensProblem(n::Int64)
        return new(n, fill(Nullable{Int64}(nothing), n));
    end
end

function actions(problem::NQueensProblem, state::AbstractVector)
    if (!isnull(state[length(state)]))
        return Array{Any, 1}([]);
    else
        local col = utils.null_index(state);
        return collect(row for row in 1:problem.N if (!conflicted(problem, state, row, col)));
    end
end

function conflict(problem::NQueensProblem, row1::Int64, col1::Int64, row2::Int64, col2::Int64)
    return ((row1 == row2) ||
            (col1 == col2) ||
            (row1 - col1 == row2 - col2) ||
            (row1 + col1 == row2 + col2));
end

function conflict(problem::NQueensProblem, row1::Int64, col1::Int64, row2::Nullable{Int64}, col2::Int64)
    if (isnull(row2))
        row2 = typemin(Int64);
    else
        row2 = get(row2);
    end
    return ((row1 == row2) ||
            (col1 == col2) ||
            (row1 - col1 == row2 - col2) ||
            (row1 + col1 == row2 + col2));
end

function conflicted(problem::NQueensProblem, state::AbstractVector, row::Int64, col::Int64)
    return any(conflict(problem, row, col, state[i], i) for i in 1:(col-1));
end

function get_result(problem::NQueensProblem, state::AbstractVector, row::Int64)
    local col = utils.null_index(state);
    local new_result = deepcopy(state);
    new_result[col] = row;
    return new_result;
end

function goal_test(problem::NQueensProblem, state::AbstractVector)
    if (isnull(state[length(state)]))
        return false;
    end
    return !any(conflicted(problem, state, get(state[col]), col) for col in 1:length(state));
end

capital_case_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

cubes16 = Array{String, 1}(["FORIXB", "MOQABJ", "GURILW", "SETUPL",
                            "CMPDAE", "ACITAO", "SLCRAE", "ROMASH",
                            "NODESW", "HEFIYE", "ONUDTK", "TEVIGN",
                            "ANEDVZ", "PINESH", "ABILYT", "GKYLEU"]);

function random_boggle(;n::Int64=4)
    local cubes = collect(cubes16[(i % 16) + 1] for i in 0:((n * n) - 1));
    shuffle!(RandomDeviceInstance, cubes);
    return map((function(array::String)
                    return rand(RandomDeviceInstance, collect(array));
                end), cubes);
end

boyan_best = collect("RSTCSDEIAEGNLRPEATESMSSID");

function print_boggle(board::Array{Char, 1})
    local nn = length(board);
    local n = int_sqrt(nn);
    local board_str = "";
    for i in 0:(nn - 1)
        if ((i % n == 0) && (i > 0))
            board_str = board_str * "\n";
        end
        if (board[i + 1] == 'Q')
            board_str = board_str * "Qu ";
        else
            board_str = board_str * String([board[i + 1]]) * "  ";
        end
    end
    print(board_str);
    nothing;
end

function boggle_neighbors(nn::Int64; cache::Dict=Dict{Any, Any}())
    if haskey(cache, nn)
        return cache[nn];
    end
    local n = int_sqrt(nn)
    local neighbors = Array(Any, nn);
    for i in 0:(nn - 1)
        neighbors[i + 1] = Array{Int64, 1}([]);
        on_top::Bool = (i < n);
        on_bottom::Bool = (i >= (nn - n));
        on_left::Bool = (i % n == 0);
        on_right::Bool = ((i + 1) % n == 0);
        if (!on_top)
            push!(neighbors[i + 1], (i + 1) - n);
            if (!on_left)
                push!(neighbors[i + 1], (i + 1) - n - 1);
            end
            if (!on_right)
                push!(neighbors[i + 1], (i + 1) - n + 1);
            end
        end
        if (!on_bottom)
            push!(neighbors[i + 1], (i + 1) + n);
            if (!on_left)
                push!(neighbors[i + 1], (i + 1) + n - 1);
            end
            if (!on_right)
                push!(neighbors[i + 1], (i + 1) + n + 1);
            end
        end
        if (!on_left)
            push!(neighbors[i + 1], (i + 1) - 1);
        end
        if (!on_right)
            push!(neighbors[i + 1], (i + 1) + 1);
        end
    end
    cache[nn] = neighbors;
    return neighbors;
end

function int_sqrt(n::Number)
    return Int64(sqrt(n));
end

type WordList
    words::Array{String, 1}
    bounds::Dict{Char, Tuple{Any, Any}}

    function WordList(filename::String; min_len::Int64=3)
        local wlba = read(filename);
        local wls = uppercase(String(wlba));
        local wlsa = sort(map(strip, split(wls, '\n')));
        local wlsa_filtered = collect(s for s in wlsa if (length(s) >= min_len));
        nwl::WordList = new(wlsa_filtered, Dict{Char, Tuple{Any, Any}}());
        for c in capital_case_alphabet
            fc::Char = c + 1;   #following character
            nwl.bounds[c] = (searchsorted(nwl.words, String([c]), 1, length(nwl.words), Base.Order.Forward).stop + 1,
                        searchsorted(nwl.words, String([fc]), 1, length(nwl.words), Base.Order.Forward).stop + 1);
        end
        return nwl;
    end
end

function lookup(wl::WordList, prefix::String; lo::Int64=1, hi::Union{Void, Int64}=nothing)
    local words = wl.words;
    if (typeof(hi) <: Void)
        hi = length(words);
    end
    local i::Int64 = searchsorted(words, prefix, lo, hi, Base.Order.Forward).start;

    #'i' is only larger than length of words when the returned index is not in WordList.
    if (i <= length(words) && startswith(words[i], prefix))
        return i, (words[i] == prefix);
    else
        return nothing, false;
    end
end

length(wl::WordList) = length(wl.words);

in(prefix::String, wl::WordList) = getindex(lookup(wl, prefix), 2);


type BoggleFinder
    wordlist::WordList
    scores::AbstractVector
    found::Dict
    board::AbstractVector
    neighbors::AbstractVector

    function BoggleFinder(;board::Union{Void, AbstractVector}=nothing, fn::Union{Void, String}=nothing)
        local wlfn::String;
        if (typeof(fn) <: Void)
            if (is_windows())
                wlfn = "..\\aima-data\\EN-text\\wordlist.txt";
            elseif (is_apple() || is_unix())
                wlfn = "../aima-data/EN-text/wordlist.txt";
            end
        else
            wlfn = fn;
        end
        nbf = new(WordList(wlfn),
                vcat([0, 0, 0, 0, 1, 2, 3, 5], fill(11, 100)),
                Dict{Any, Any}())
        if (!(typeof(board) <: Void))
            set_board(nbf, board=board);
        end
        return nbf;
    end
end

function set_board(bf::BoggleFinder; board::Union{Void, AbstractVector}=nothing)
    if (typeof(board) <: Void)
        board = random_boggle();
    end
    bf.board = board;
    bf.neighbors = boggle_neighbors(length(board));
    bf.found = Dict{Any, Any}();
    for i in 1:length(board)
        lo::Int64, hi::Int64 = bf.wordlist.bounds[board[i]];
        find(bf, lo, hi, i, [], "");
    end
    return bf;
end

function find(bf::BoggleFinder, lo::Int64, hi::Int64, i::Int64, visited::AbstractVector, prefix::String)
    if i in visited
        return nothing;
    end
    wordpos, is_word::Bool = lookup(bf.wordlist, prefix, lo=lo, hi=hi);
    if (!(typeof(wordpos) <: Void))
        if (is_word)
            bf.found[prefix] = true;
        end
        push!(visited, i);
        local c = bf.board[i];
        if (c == 'Q')
            c = "QU";
        else
            c = String([c]);
        end
        prefix = prefix * c;
        for j in bf.neighbors[i]
            find(bf, wordpos, hi, j, visited, prefix);
        end
        pop!(visited);
    end
    return nothing;
end

function words(bf::BoggleFinder)
    return collect(keys(bf.found));
end

function score(bf::BoggleFinder)
    return sum(collect(scores[len(w)] for w in words(bf)));
end

length(bf::BoggleFinder) = length(bf.found);

function boggle_hill_climbing(;board::Union{Void, AbstractVector}=nothing, ntimes::Int64=100, verbose::Bool=true)
    finder = BoggleFinder();
    if (typeof(board) <: Void)
        board = random_boggle();
    end
    local best_length::Int64 = length(set_board(finder, board=board));
    for t in 1:ntimes
        i, old_char = mutate_boggle(board);
        local new_length::Int64 = length(set_board(finder, board=board));
        if (new_length > best_length)
            best_length = new_length;
            if (verbose)
                println(best_length, " ", t, " ", board);
            end
        else
            board[i] = old_char;
        end
    end
    if (verbose)
        print_boggle(board);
    end
    return board, best_length;
end


function mutate_boggle(board::AbstractArray)
    local i::Int64 = rand(RandomDeviceInstance, 1:length(board));
    local old_char::Char = board[i];
    board[i] = rand(RandomDeviceInstance, collect(rand(RandomDeviceInstance, cubes16)));
    return i, old_char;
end

function execute_searcher{T <: AbstractProblem}(searcher::Function, problem::T)
    local p = InstrumentedProblem(problem);
    searcher(p);
    return p;
end

function compare_searchers{T <: AbstractProblem}(problems::Array{T, 1},
                                                header::Array{String, 1};
                                                searchers::Array{Function, 1}=[breadth_first_tree_search,
                                                                            breadth_first_search,
                                                                            depth_first_graph_search,
                                                                            iterative_deepening_search,
                                                                            depth_limited_search,
                                                                            recursive_best_first_search])
    local table = vcat(permutedims(hcat(header), [2, 1]), 
                        hcat(map(string, searchers), 
                            permutedims(reduce(hcat,
                                collect(
                                    collect(format_instrumented_results(execute_searcher(s, p)) for p in problems)
                                for s in searchers)),
                            [2,1])));
    return table;
end

function beautify_node(n::Node)
    return @sprintf("%s%s%s", "<Node ", string(n.state), ">");
end

