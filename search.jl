import Base: ==, length, in, eltype;

export Problem, InstrumentedProblem,
        actions, get_result, goal_test, path_cost, value,
        format_instrumented_results,
        Node, expand, child_node, solution, path, ==,
        search,
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
        OnlineDFSAgentProgram, update_state, execute,
        OnlineSearchProblem, LRTAStarAgentProgram,
        learning_realtime_astar_cost,
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
mutable struct Problem <: AbstractProblem
    initial::String
    goal::Union{Nothing, String}

    function Problem(initial_state::String; goal_state::Union{Nothing, String}=nothing)
        return new(initial_state, goal_state);
    end
end

"""
    actions(ap::T, state::String) where {T <: AbstractProblem}

Return an array of possible actions that can be executed in the given state 'state'.
"""
function actions(ap::T, state::String) where {T <: AbstractProblem}
    println("actions() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

"""
    get_result(ap::T, state::String, action::String) where {T <: AbstractProblem}

Return the resulting state from executing the given action 'action' in the given state 'state'.
"""
function get_result(ap::T, state::String, action::String) where {T <: AbstractProblem}
    println("get_result() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

"""
    goal_test(ap::T, state::String) where {T <: AbstractProblem}

Return a boolean value representing whether the given state 'state' is a goal state in the given
problem 'ap'.
"""
function goal_test(ap::T, state::String) where {T <: AbstractProblem}
    return ap.goal == state;
end

"""
    path_cost(ap::T, cost::Float64, state1::String, action::String, state2::String) where {T <: AbstractProblem}
    path_cost(ap::T, cost::Float64, state1::AbstractVector, action::Int64, state2::AbstractVector) where {T <: AbstractProblem}

Return the cost of a solution path arriving at 'state2' from 'state1' with the given action 'action' and
cost 'cost' to arrive at 'state1'. The default path_cost() method costs 1 for every step in a path.
"""
function path_cost(ap::T, cost::Float64, state1::String, action::String, state2::String) where {T <: AbstractProblem}
    return cost + 1;
end

function path_cost(ap::T, cost::Float64, state1::AbstractVector, action::Int64, state2::AbstractVector) where {T <: AbstractProblem}
    return cost + 1;
end

"""
    value(ap::T, state::String) where {T <: AbstractProblem}

Return a value for the given state 'state' in the given problem 'ap'.

This value is used in optimization problems such as hill climbing or simulated annealing.
"""
function value(ap::T, state::String) where {T <: AbstractProblem}
    println("value() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

#=

    InstrumentedProblem is a AbstractProblem implementation that wraps another AbstractProblem

    implementation and tracks the number of function calls made. This problem is used in

    compare_searchers() and execute_searcher().

=#
mutable struct InstrumentedProblem <: AbstractProblem
    problem::AbstractProblem
    actions::Int64
    results::Int64
    goal_tests::Int64
    found   # can be any DataType, but check for Nothing DataType later

    function InstrumentedProblem(ap::T) where {T <: AbstractProblem}
        return new(ap, Int64(0), Int64(0), Int64(0), nothing);
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
        ap.found = state;
    end
    return result;
end

function goal_test(ap::InstrumentedProblem, state::AbstractVector)
    ap.goal_tests = ap.goal_tests + 1;
    local result::Bool = goal_test(ap.problem, state);
    if (result)
        ap.found = state;
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
    return @sprintf("<%4d/%4d/%4d/%s>", ap.actions, ap.goal_tests, ap.results, string(ap.found));
end

# A node should not exist without a state.
mutable struct Node{T}
    state::T
    path_cost::Float64
    depth::UInt32
    action::Union{Nothing, String, Int64, Tuple}
    parent::Union{Nothing, Node}
    f::Float64

    function Node{T}(state::T; parent::Union{Nothing, Node}=nothing, action::Union{Nothing, String, Int64, Tuple}=nothing, path_cost::Float64=0.0, f::Union{Nothing, Float64}=nothing) where T
        nn = new(state, path_cost, UInt32(0), action, parent);
        if (typeof(parent) <: Node)
            nn.depth = UInt32(parent.depth + 1);
        end
        if (typeof(f) <: Float64)
            nn.f = f;
        end
        return nn;
    end
end

"""
    expand(n::Node, ap::T) where {T <: AbstractProblem}

Return an array of nodes reachable by 1 step from the given node 'n' in the problem 'ap'.
"""
function expand(n::Node, ap::T) where {T <: AbstractProblem}
    return collect(child_node(n, ap, act) for act in actions(ap, n.state));
end

"""
    child_node(n::Node, ap::T, action::String) where {T <: AbstractProblem}

Return a child node for the given node 'n' in problem 'ap' after executing the action 'action' (Fig. 3.10).
"""
function child_node(n::Node, ap::T, action::String) where {T <: AbstractProblem}
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

function child_node(n::Node, ap::T, action::Int64) where {T <: AbstractProblem}
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

function child_node(n::Node, ap::T, action::Tuple) where {T <: AbstractProblem}
    local next_node = get_result(ap, n.state, action);
    return Node{typeof(next_node)}(next_node, parent=n, action=action, path_cost=path_cost(ap, n.path_cost, n.state, action, next_node));
end

"""
    solution(n::Node)

Return an array of actions to get from the root node of node 'n' to the given node 'n'.
"""
function solution(n::Node)
    local path_sequence = path(n);
    return [node.action for node in path_sequence[2:length(path_sequence)]];
end

"""
    path(n::Node)

Return the path between the root node of node 'n' to the given node 'n' as an array of nodes.
"""
function path(n::Node)
    local node = n;
    local path_back = [];
    while true
        push!(path_back, node);
        if (!(node.parent === nothing))
            node = node.parent;
        else
            # The root node does not have a parent node.
            break;
        end
    end
    path_back = reverse(path_back);
    return path_back;
end

function ==(n1::Node, n2::Node)
    return (n1.state == n2.state);
end

#=

    SimpleProblemSolvingAgentProgram is a abstract problem solving agent (Fig. 3.1).

=#
mutable struct SimpleProblemSolvingAgentProgram <: AgentProgram
    state::Union{Nothing, String}
    goal::Union{Nothing, String}
    seq::Array{String, 1}
    problem::Union{Nothing, Problem}

    function SimpleProblemSolvingAgentProgram(;initial_state::Union{Nothing, String}=nothing)
        return new(initial_state, nothing, Array{String, 1}(), nothing);
    end
end

function execute(spsap::SimpleProblemSolvingAgentProgram, percept::Tuple{Any, Any})
    spsap.state = update_state(spsap, spsap.state, percept);
    if (length(spsap.seq) == 0)
        spsap.goal = formulate_problem(spsap, spsap.state);
        spsap.problem = forumate_problem(spsap, spsap.state, spsap.goal);
        spsap.seq = search(spsap, spsap.problem);
        if (length(spsap.seq) == 0)
            return Nothing;
        end
    end
    local action = popfirst!(spsap.seq);
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

function search(spsap::SimpleProblemSolvingAgentProgram, problem::T) where {T <: AbstractProblem}
    println("search() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

struct GAState
    genes::Array{Any, 1}

    function GAState(genes::Array{Any, 1})
        return new(Array{Any,1}(deepcopy(genes)));
    end
end

function mate(ga_state::T, other::T) where {T <: GAState}
    local c = rand(RandomDeviceInstance, range(1, stop=length(ga_state.genes)));
    local new_ga_state = deepcopy(ga_state[1:c]);
    for element in other.genes[(c + 1):length(other.genes)]
        push!(new_ga_state, element);
    end
    return new_ga_state;
end

function mutate(ga_state::T) where {T <: GAState}
    println("mutate() is not implemented yet for ", typeof(ga_state), "!");
    nothing;
end

"""
    tree_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)

Search the given problem by using the general tree search algorithm (Fig. 3.7) and return the node solution.
"""
function tree_search(problem::T1, frontier::T2) where {T1 <: AbstractProblem, T2 <: Queue}
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

function tree_search(problem::InstrumentedProblem, frontier::T) where {T <: Queue}
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
function graph_search(problem::T1, frontier::T2) where {T1 <: AbstractProblem, T2 <: Queue}
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

function graph_search(problem::InstrumentedProblem, frontier::T) where {T <: Queue}
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

"""
    breadth_first_tree_search(problem::T) where {T <: AbstractProblem}

Search the shallowest nodes in the search tree first.
"""
function breadth_first_tree_search(problem::T) where {T <: AbstractProblem}
    return tree_search(problem, FIFOQueue());
end

"""
    depth_first_tree_search(problem::T) where {T <: AbstractProblem}

Search the deepest nodes in the search tree first.
"""
function depth_first_tree_search(problem::T) where {T <: AbstractProblem}
    return tree_search(problem, Stack());
end

"""
    depth_first_graph_search(problem::T) where {T <: AbstractProblem}

Search the deepest nodes in the search tree first.
"""
function depth_first_graph_search(problem::T) where {T <: AbstractProblem}
    return graph_search(problem, Stack());
end

"""
    breadth_first_search(problem::T) where {T <: AbstractProblem}
    breadth_first_search(problem::InstrumentedProblem)

Return a solution by using the breadth-first search algorithm (Fig. 3.11)
on the given problem 'problem'. Otherwise, return 'nothing' on failure.
"""
function breadth_first_search(problem::T) where {T <: AbstractProblem}
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

"""
    best_first_graph_search(problem::T, f::Function) where {T <: AbstractProblem}

Search the nodes in the given problem 'problem' by visiting the nodes with the lowest
scores returned by f(). If f() is a heuristics estimate function to the goal state, then
this function becomes greedy best first search. If f() is a function that gets the node's
depth, then this function becomes breadth-first search.

Returns a solution if found, otherwise returns 'nothing' on failure.

This function uses f as a Function, because using f as an MemoizedFunction exhibits unusual
behavior when relying on MemoizedFunction by producing unexpected results.
"""
function best_first_graph_search(problem::T, f::Function) where {T <: AbstractProblem}
    local node = Node{typeof(problem.initial)}(problem.initial);
    if (goal_test(problem, node.state))
        return node;
    end
    local frontier = PQueue();
    push!(frontier, node, f);
    local explored = Set{typeof(problem.initial)}();
    while (length(frontier) != 0)
        node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        push!(explored, node.state);
        for child_node in expand(node, problem)
            if (!(child_node.state in explored) &&
                !(child_node in collect(getindex(x, 2) for x in frontier.array)))
                push!(frontier, child_node, f);
            elseif (child_node in [getindex(x, 2) for x in frontier.array])
                # Recall that Nodes can share the same state and different values for other fields.
                local existing_node = pop!(collect(getindex(x, 2)
                                                    for x in frontier.array
                                                    if (getindex(x, 2) == child_node)));
                if (f(child_node) < f(existing_node))
                    delete!(frontier, existing_node);
                    push!(frontier, child_node, f);
                end
            end
        end
    end
    return nothing;
end


"""
    uniform_cost_search(problem::T) where {T <: AbstractProblem}

Search the given problem by using the uniform cost algorithm (Fig. 3.14) and return the node solution.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function uniform_cost_search(problem::T) where {T <: AbstractProblem}
    return best_first_graph_search(problem, (function(n::Node)return n.path_cost;end));
end

function recursive_dls(node::Node, problem::T, limit::Int64) where {T <: AbstractProblem}
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
            elseif (!(typeof(result) <: Nothing))
                return result;
            end
        end
        return if_(cutoff_occurred, "cutoff", nothing);
    end
end;

"""
    depth_limited_search(problem::T; limit::Int64) where {T <: AbstractProblem}

Search the given problem by using the depth limited tree search algorithm (Fig. 3.17)
and return the node solution if a solution was found. Otherwise, this function returns 'nothing'.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function depth_limited_search(problem::T; limit::Int64=50) where {T <: AbstractProblem}
    return recursive_dls(Node{typeof(problem.initial)}(problem.initial), problem, limit);
end

function depth_limited_search(problem::InstrumentedProblem; limit::Int64=50)
    return recursive_dls(Node{typeof(problem.problem.initial)}(problem.problem.initial), problem, limit);
end

"""
    iterative_deepening_search(problem::T) where {T <: AbstractProblem}

Search the given problem by using the iterative deepening search algorithm (Fig. 3.18)
and return the node solution if a solution was found. Otherwise, this function returns 'nothing'.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function iterative_deepening_search(problem::T) where {T <: AbstractProblem}
    for depth in 1:typemax(Int64)
        local result = depth_limited_search(problem, limit=depth)
        if (result != "cutoff")
            return result;
        end
    end
    return nothing;
end

const greedy_best_first_graph_search = best_first_graph_search;

#=

    Graph is a graph that consists of nodes (vertices) and edges (links).

    The Graph constructor uses the keyword 'directed' to specify if the graph

    is directed or undirected.


    For an example Graph instance:

        Graph(dict=Dict([("A", Dict([("B", 1), ("C", 2)]))]))

    The example Graph is a directed graph with 3 vertices ("A", "B", and "C")

    and link "A"=>"B" (length 1) and "A"=>"C" (length 2).

=#
struct Graph{N}
    dict::Dict{N, Any}
    locations::Dict{N, Tuple{Any, Any}}
    directed::Bool

    function Graph{N}(;dict::Union{Nothing, Dict{N, }}=nothing, locations::Union{Nothing, Dict{N, Tuple{Any, Any}}}=nothing, directed::Bool=true) where N
        local ng::Graph;
        if ((typeof(dict) <: Nothing) && (typeof(locations) <: Nothing))
            ng = new(Dict{Any, Any}(), Dict{Any, Tuple{Any, Any}}(), Bool(directed));
        elseif (typeof(locations) <: Nothing)
            ng = new(Dict{eltype(dict.keys), Any}(dict), Dict{Any, Tuple{Any, Any}}(), Bool(directed));
        else
            ng = new(Dict{eltype(dict.keys), Any}(dict), Dict{eltype(locations.keys), Tuple{Any, Any}}(locations), Bool(directed));
        end
        if (!ng.directed)
            make_undirected(ng);
        end
        return ng;
    end

    function Graph{N}(graph::Graph{N}) where N
        return new(Dict{Any, Any}(graph.dict), Dict{String, Tuple{Any, Any}}(graph.locations), Bool(graph.directed));
    end
end

eltype(::Type{<:Graph{T}}) where {T} = T

function make_undirected(graph::Graph)
    for location_A in keys(graph.dict)
        for (location_B, d) in graph.dict[location_A]
            connect_nodes(graph, location_B, location_A, distance=d);
        end
    end
end

"""
    connect_nodes(graph::Graph{N}, A::N, B::N; distance::Int64=Int64(1)) where N

Add a link between Node 'A' to Node 'B'. If the graph is undirected, then add
the inverse link from Node 'B' to Node 'A'.
"""
function connect_nodes(graph::Graph{N}, A::N, B::N; distance::Int64=Int64(1)) where N
    get!(graph.dict, A, Dict{String, Int64}())[B]=distance;
    if (!graph.directed)
        get!(graph.dict, B, Dict{String, Int64}())[A]=distance;
    end
    nothing;
end

"""
    get_linked_nodes(graph::Graph{N}, a::N; b::Union{Nothing, N}=nothing) where N

Return a dictionary of nodes and their distances if the 'b' keyword is not given.
Otherwise, return the distance between 'a' and 'b'.
"""
function get_linked_nodes(graph::Graph{N}, a::N; b::Union{Nothing, N}=nothing) where N
    local linked = get!(graph.dict, a, Dict{Any, Any}());
    if (typeof(b) <: Nothing)
        return linked;
    else
        return get(linked, b, nothing);
    end
end

function get_nodes(graph::Graph)
    return collect(keys(graph.dict));
end

"""
    UndirectedGraph(dict::Dict{T, }, locations::Dict{T, Tuple{Any, Any}}) where T
    UndirectedGraph()

Return an undirected graph from the given dictionary of links 'dict' and dictionary 
of locations 'locations' if given.
"""
function UndirectedGraph(dict::Dict{T, }, locations::Dict{T, Tuple{Any, Any}}) where T
    return Graph{eltype(dict.keys)}(dict=dict, locations=locations, directed=false);
end

function UndirectedGraph()
    return Graph{Any}(directed=false);
end

"""
    RandomGraph()

Return a random graph with the specified nodes and number of links.
"""
function RandomGraph(;nodes::UnitRange=1:10,
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

#=

    GraphProblem is the problem of searching a graph from one node to another node.

=#
struct GraphProblem <: AbstractProblem
    initial::String
    goal::String
    graph::Graph
    h::MemoizedFunction


    function GraphProblem(initial_state::String, goal_state::String, graph::Graph)
        return new(initial_state, goal_state, Graph{eltype(graph)}(graph), MemoizedFunction(initial_to_goal_distance));
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

"""
    astar_search(problem::GraphProblem; h::Union{Nothing, Function}=nothing)

Apply the A* search (best-first graph search with f(n)=g(n)+h(n)) to the given problem 'problem'.
If the 'h' keyword is not used, this function uses the function problem.h.

This function uses mh as a Function, because using mh as an MemoizedFunction exhibits unusual
behavior when relying on MemoizedFunction by producing unexpected results.
"""
function astar_search(problem::GraphProblem; h::Union{Nothing, Function}=nothing)
    local mh::Function;
    if (!(typeof(h) <: Nothing))
        mh = h;
    else
        mh = problem.h.f;
    end
    return best_first_graph_search(problem,
                                    (function(node::Node; h::Function=mh, prob::GraphProblem=problem)
                                        return node.path_cost + h(prob, node);
                                    end));
end

"""
    RBFS(problem::T1, node::T2, flmt::Float64, h::MemoizedFunction) where {T1 <: AbstractProblem, T2 <: Node}

Recursively calls RBFS() with a new 'flmt' value and returns its solution to recursive_best_first_search().
"""
function RBFS(problem::T1, node::T2, flmt::Float64, h::MemoizedFunction) where {T1 <: AbstractProblem, T2 <: Node}
    if (goal_test(problem, node.state))
        return node, 0.0;
    end
    local successors = expand(node, problem);
    if (length(successors) == 0);
        return node, Inf;
    end
    for successor in successors
        successor.f = max(successor.path_cost + eval_memoized_function(h, problem, successor), node.f);
    end
    while (true)
        sort!(successors, lt=(function(n1::Node, n2::Node)return isless(n1.f, n2.f);end));
        local best::Node = successors[1];
        if (best.f > flmt)
            return nothing, best.f;
        end
        local alternative::Float64;
        if (length(successors) > 1)
            alternative = successors[1].f;
        else
            alternative = Inf;
        end
        result, best.f = RBFS(problem, best, min(flmt, alternative), h);
        if (!(result === nothing))
            return result, best.f;
        end
    end
end

"""
    recursive_best_first_search(problem::T; h::Union{Nothing, MemoizedFunction}) where {T <: AbstractProblem}

Search the given problem by using the recursive best first search algorithm (Fig. 3.26)
and return the node solution.

solution() can be used on the node solution to reconstruct the path taken to the solution.
"""
function recursive_best_first_search(problem::T; h::Union{Nothing, MemoizedFunction}=nothing) where {T <: AbstractProblem}
    local mh::MemoizedFunction; #memoized h(n) function
    if (!(typeof(h) <: Nothing))
        mh = MemoizedFunction(h);
    else
        mh = problem.h;
    end

    local node = Node{typeof(problem.initial)}(problem.initial);
    node.f = eval_memoized_function(mh, problem, node);
    result, bestf = RBFS(problem, node, Inf, mh);
    return result;
end

function recursive_best_first_search(problem::InstrumentedProblem; h::Union{Nothing, MemoizedFunction}=nothing)
    local mh::MemoizedFunction; #memoized h(n) function
    if (!(typeof(h) <: Nothing))
        mh = MemoizedFunction(h);
    else
        mh = problem.problem.h;
    end

    local node = Node{typeof(problem.problem.initial)}(problem.problem.initial);
    node.f = eval_memoized_function(mh, problem, node);
    result, bestf = RBFS(problem, node, Inf, mh);
    return result;
end

"""
    hill_climbing(problem::T) where {T <: AbstractProblem}

Return a state that is a local maximum for the given problem 'problem' by using
the hill-climbing search algorithm (Fig. 4.2) on the initial state of the problem.
"""
function hill_climbing(problem::T) where {T <: AbstractProblem}
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

"""
    exp_schedule(;kvar::Int64=20, delta::Float64=0.005, lmt::Int64=100)

Return a scheduled time for simulated annealing.
"""
function exp_schedule(;kvar::Int64=20, delta::Float64=0.005, lmt::Int64=100)
    return (function(t::Real; k=kvar, d=delta, limit=lmt)
                return if_((t < limit), (k * exp(-d * t)), 0);
            end);
end

"""
    simulated_annealing(problem::T; schedule::Function=exp_schedule()) where {T <: AbstractProblem}

Return the solution node by applying the simulated annealing algorithm (Fig. 4.5) on the given
problem 'problem' and schedule function 'schedule'. If a solution node can't be found,
this function returns 'nothing' on failure.
"""
function simulated_annealing(problem::T; schedule::Function=exp_schedule()) where {T <: AbstractProblem}
    local current_node = Node{typeof(problem.initial)}(problem.initial);
    for t in 0:(typemax(Int64) - 1)
        local temperature::Float64 = schedule(t);
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
function or_search(problem::T, state::AbstractVector, path::AbstractVector) where {T <: AbstractProblem}
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

function or_search(problem::T, state::String, path::AbstractVector) where {T <: AbstractProblem}
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

function and_search(problem::T, states::AbstractVector, path::AbstractVector) where {T <: AbstractVector}
    local plan = Dict{Any, Any}();
    for state in states
        plan[state] = or_search(problem, state, path);
        if (plan[state] == nothing)
            return nothing;
        end
    end
    return plan;
end

"""
    and_or_graph_search(problem::T) where {T <: AbstractProblem}

Return a conditional plan by using the algorithm for searching and-or graphs (Fig. 4.11)
on the given problem 'problem'. This function returns 'nothing' on failure.
"""
function and_or_graph_search(problem::T) where {T <: AbstractProblem}
    return or_search(problem, problem.initial, []);
end

#=

    OnlineDFSAgentProgram is a online depth first search agent (Fig. 4.21)

    implementation of AgentProgram.

=#
mutable struct OnlineDFSAgentProgram <: AgentProgram
    result::Dict
    untried::Dict
    unbacktracked::Dict
    state::Union{Nothing, String}
    action::Union{Nothing, String}
    problem::AbstractProblem

    function OnlineDFSAgentProgram(problem::T) where {T <: AbstractProblem}
        return new(Dict(), Dict(), Dict(), nothing, nothing, problem);
    end
end

function update_state(odfsap::OnlineDFSAgentProgram, percept::String)
    return percept;
end

function execute(odfsap::OnlineDFSAgentProgram, percept::String)
    local s_prime::String = update_state(odfsap, percept);
    if (goal_test(odfsap.problem, s_prime))
        odfsap.action = nothing;
    else
        if (!(s_prime in keys(odfsap.untried)))
            odfsap.untried[s_prime] = actions(odfsap.problem, s_prime);
        end
        if (!(odfsap.state === nothing))
            if (haskey(odfsap.result, (odfsap.state, odfsap.action)))
                if (s_prime != odfsap.result[(odfsap.state, odfsap.action)])
                    odfsap.result[(odfsap.state, odfsap.action)] = s_prime;
                    pushfirst!(odfsap.unbacktracked[s_prime], odfsap.state);
                end
            else
                if (s_prime != [])
                    odfsap.result[(odfsap.state, odfsap.action)] = s_prime;
                    pushfirst!(odfsap.unbacktracked[s_prime], odfsap.state);
                end
            end
        end
        if (length(odfsap.untried[s_prime]) == 0)
            if (length(odfsap.unbacktracked[s_prime]) == 0)
                odfsap.action = nothing;
            else
                first_item = popfirst!(odfsap.unbacktracked[s_prime]);
                for (state, b) in keys(odfsap.result)
                    if (odfsap.result[(state, b)] == first_item)
                        odfsap.action = b;
                        break;
                    end
                end
            end
        else
            odfsap.action = popfirst!(odfsap.untried[s_prime]);
        end
    end
    odfsap.state = s_prime;
    return odfsap.action;
end

#=

    OnlineSearchProblem is a AbstractProblem implementation of a online search problem

    that can be solved by a online search agent.

=#
struct OnlineSearchProblem <: AbstractProblem
    initial::String
    goal::String
    graph::Graph
    least_costs::Dict
    h::Function

    function OnlineSearchProblem(initial::String, goal::String, graph::Graph, least_costs::Dict)
        return new(initial, goal, graph, least_costs, online_search_least_cost);
    end
end

function actions(osp::OnlineSearchProblem, state::String)
    return collect(keys(osp.graph.dict[state]));
end

function get_result(osp::OnlineSearchProblem, state::String, action::String)
    return osp.graph.dict[state][action];
end

function online_search_least_cost(osp::OnlineSearchProblem, state::String)
    return osp.least_costs[state];
end

function path_cost(osp::OnlineSearchProblem, state1::String, action::String, state2::String)
    return 1;
end

function goal_test(osp::OnlineSearchProblem, state::String)
    if (state == osp.goal)
        return true;
    else
        return false;
    end
end

#=

    LRTAStarAgentProgram is an AgentProgram implementation of LRTA*-Agent (Fig. 4.24).

    The 'result' field is not necessary as the given problem contains the results table.

=#
mutable struct LRTAStarAgentProgram <: AgentProgram
    H::Dict
    state::Union{Nothing, String}
    action::Union{Nothing, String}
    problem::AbstractProblem

    function LRTAStarAgentProgram(problem::T) where {T <: AbstractProblem}
        return new(Dict(), nothing, nothing, problem);
    end
end

function learning_realtime_astar_cost(lrtaap::LRTAStarAgentProgram, state::String, action::String, s_prime::String, H::Dict)
    if (haskey(lrtaap.H, s_prime))
        return path_cost(lrtaap.problem, state, action, s_prime) + lrtaap.H[s_prime];
    else
        return path_cost(lrtaap.problem, state, action, s_prime) + lrtaap.problem.h(lrtaap.problem, s_prime);
    end
end

"""
    execute(lrtaap::LRTAStarAgentProgram, s_prime::String)

Return an action given the percept 's_prime' and by using the LRTA*-Agent
program (Fig. 4.24). If the current state of the agent is at the goal
state, return 'nothing'.
"""
function execute(lrtaap::LRTAStarAgentProgram, s_prime::String)
    if (goal_test(lrtaap.problem, s_prime))
        lrtaap.action = nothing;
        return nothing;
    else
        if (!haskey(lrtaap.H, s_prime))
            lrtaap.H[s_prime] = lrtaap.problem.h(lrtaap.problem, s_prime);
        end
        if (!(lrtaap.state === nothing))
            lrtaap.H[lrtaap.state] = reduce(min, learning_realtime_astar_cost(lrtaap,
                                                                        lrtaap.state,
                                                                        b,
                                                                        get_result(lrtaap.problem, lrtaap.state, b),
                                                                        lrtaap.H)
                                        for b in actions(lrtaap.problem, lrtaap.state));
        end
        lrtaap.action = argmin(actions(lrtaap.problem, s_prime),
                                (function(b::String)
                                    return learning_realtime_astar_cost(lrtaap,
                                                                        s_prime,
                                                                        b,
                                                                        get_result(lrtaap.problem, s_prime, b),
                                                                        lrtaap.H);
                                end));
        lrtaap.state = s_prime;
        return lrtaap.action;
    end
end

function genetic_search(problem::T; ngen::Int64=1000, pmut::Float64=0.1, n::Int64=20) where {T <: AbstractProblem}
    local s = problem.initial;
    local states = [result(s, action) for action in actions(problem, s)];
    shuffle!(RandomDeviceInstance, states);
    if (length(states) < n)
        n = length(states);
    end
    return genetic_algorithm(states[1:n], value, ngen=ngen, pmut=pmut);
end

function genetic_algorithm(population::T, fitness::Function; ngen::Int64=1000, pmut::Float64=0.1) where {T <: AbstractVector}
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

# Simplified road map of Romania example (Fig. 3.2)
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

# One-dimensional state space example (Fig. 4.23)
one_dim_state_space = Graph{String}(dict=Dict{String, Dict{String, String}}([Pair("State_1", Dict([Pair("Right", "State_2")])),
                                        Pair("State_2", Dict([Pair("Right", "State_3"),
                                                            Pair("Left", "State_1")])),
                                        Pair("State_3", Dict([Pair("Right", "State_4"),
                                                            Pair("Left", "State_2")])),
                                        Pair("State_4", Dict([Pair("Right", "State_5"),
                                                            Pair("Left", "State_3")])),
                                        Pair("State_5", Dict([Pair("Right", "State_6"),
                                                            Pair("Left", "State_4")])),
                                        Pair("State_6", Dict([Pair("Left", "State_5")]))]));

one_dim_state_space_least_costs = Dict([Pair("State_1", 8),
                                        Pair("State_2", 9),
                                        Pair("State_3", 2),
                                        Pair("State_4", 2),
                                        Pair("State_5", 4),
                                        Pair("State_6", 3)]);

# Principal states and territories of Australia example (Fig. 6.1)
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

#=

    NQueensProblem is the problem of placing 'N' non-attacking queens on a 'N'x'N' chess board.

    Each state is represented as an 'N' element array where the value of 'r' at index 'c' implies

    that a queen occupies the position at row 'r' and column 'c'. The columns are values are filled

    from left to right.

=#
struct NQueensProblem <: AbstractProblem
    N::Int64
    initial::Array{Union{Nothing, Int64}, 1}

    function NQueensProblem(n::Int64)
        return new(n, fill(nothing, n));
    end
end

function actions(problem::NQueensProblem, state::AbstractVector)
    if (!(state[length(state)] === nothing))
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

function conflict(problem::NQueensProblem, row1::Int64, col1::Int64, row2::Nothing, col2::Int64)
    error("conflict(): 'row2' is not initialized!");
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
    if ((state[length(state)] === nothing))
        return false;
    end
    return !any(conflicted(problem, state, state[col], col) for col in 1:length(state));
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
    local nn::Int64 = length(board);
    local n::Int64 = int_sqrt(nn);
    local board_str::String = "";
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
    local n::Int64 = int_sqrt(nn)
    local neighbors::AbstractVector = Array{Any, 1}(undef, nn);
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

#=

    WordList contains an array of words.

=#
struct WordList
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

function lookup(wl::WordList, prefix::String; lo::Int64=1, hi::Union{Nothing, Int64}=nothing)
    local words = wl.words;
    if (typeof(hi) <: Nothing)
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

#=

    BoggleFinder contains the words found on the Boggle board

    and the array of possible words for the Boggle board.

=#
mutable struct BoggleFinder
    wordlist::WordList
    scores::AbstractVector
    found::Dict
    board::AbstractVector
    neighbors::AbstractVector

    function BoggleFinder(;board::Union{Nothing, AbstractVector}=nothing, fn::Union{Nothing, String}=nothing)
        local wlfn::String;
        if (typeof(fn) <: Nothing)
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
        if (!(typeof(board) <: Nothing))
            set_board(nbf, board=board);
        end
        return nbf;
    end
end

function set_board(bf::BoggleFinder; board::Union{Nothing, AbstractVector}=nothing)
    if (typeof(board) <: Nothing)
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
    if (!(typeof(wordpos) <: Nothing))
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

"""
    boggle_hill_climbing(;board::Union{Nothing, AbstractVector}=nothing, ntimes::Int64=100, verbose::Bool=true)

Solve the inverse Boggle by using hill climbing (initially use a random Boggle board and changing it).

Return the best Boggle board and its length.
"""
function boggle_hill_climbing(;board::Union{Nothing, AbstractVector}=nothing, ntimes::Int64=100, verbose::Bool=true)
    finder = BoggleFinder();
    if (typeof(board) <: Nothing)
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

function execute_searcher(searcher::Function, problem::T) where {T <: AbstractProblem}
    local p = InstrumentedProblem(problem);
    searcher(p);
    return p;
end

function compare_searchers(problems::Array{T, 1},
                            header::Array{String, 1};
                            searchers::Array{Function, 1}=[breadth_first_tree_search,
                                                        breadth_first_search,
                                                        depth_first_graph_search,
                                                        iterative_deepening_search,
                                                        depth_limited_search,
                                                        recursive_best_first_search]) where {T <: AbstractProblem}
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

