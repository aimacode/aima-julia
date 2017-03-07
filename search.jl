
import Base: ==;

typealias Action String;

typealias Percept Tuple{Any, Any};

abstract AbstractProblem;

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

function result{T <: AbstractProblem}(ap::T, state::String, action::Action)
    println("result() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function goal_test{T <: AbstractProblem}(ap::T, state::String)
    return ap.goal == state;
end

function path_cost{T <: AbstractProblem}(ap::T, cost::Float64, state1::String, action::Action, state2::String)
    return cost + 1;
end

function value{T <: AbstractProblem}(ap::T, state::String)
    println("value() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

type Node
    state::String
    path_cost::Float64
    depth::UInt32
    action::Nullable{Action}
    parent::Nullable{Node}

    function Node(state::String; parent::Union{Void, Node}=nothing, action::Union{Void, Action}=nothing, path_cost::Float64=0.0)
        nn = new(state, path_cost, UInt32(0), Nullable{Action}(action), Nullable{Node}(parent));
        if (typeof(parent) <: Node)
            nn.depth = UInt32(parent.depth + 1);
        end
        return nn;
    end
end

function expand{T <: AbstractProblem}(n::Node, ap::T)
    return [child_node(n, ap, act) for act in actions(ap, n.state)];
end

function child_node{T <: AbstractProblem}(n::Node, ap::T, action::Action)
    local next_node = result(ap, n.state, action);
    return Node(next_node, n, action, ap.path_cost(n.path_cost, n.state, action, next_node));
end

function solution(n::Node)
    local path_sequence = path(n);
    return [node.action for node in path_sequence[2:length(path_sequence)]];
end

function path(n::Node)
    local node = n;
    local path_back = [];
    while true
        push!(path_back, node);
        if (!isnull(node.parent))
            node = node.parent;
        else     #the root node does not have a parent node
            break;
        end
    end
    path_back = reverse(path_back);
    return path_back;
end

==(n1::Node, n2::Node) = (n1.state == n2.state);

type SimpleProblemSolvingAgentProgram
    state::Nullable{String}
    goal::Nullable{String}
    seq::Array{Action, 1}
    problem::Nullable{Problem}

    function SimpleProblemSolvingAgentProgram(;initial_state::Union{Void, String}=nothing)
        return new(Nullable{String}(initial_state), Nullable{String}(), Array{Action, 1}(), Nullable{Problem}());
    end
end

function execute{T <: SimpleProblemSolvingAgentProgram}(spsap::T, percept::Percept)
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

function update_state{T <: SimpleProblemSolvingAgentProgram}(spsap::T, state::String, percept::Percept)
    println("update_state() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function formulate_goal{T <: SimpleProblemSolvingAgentProgram}(spsap::T, state::String)
    println("formulate_goal() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function formulate_problem{T <: SimpleProblemSolvingAgentProgram}(spsap::T, state::String, goal::String)
    println("formulate_problem() is not implemented yet for ", typeof(spsap), "!");
    nothing;
end

function search{T <: SimpleProblemSolvingAgentProgram}(spsap::T, problem::Problem)
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
    local c = rand(RandomDevice(), range(1, length(ga_state.genes)));
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

Search the given problem by using the general tree search algorithm (Fig. 3.7).
"""
function tree_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)
    push!(frontier, Node(problem.initial));
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

Search the given problem by using the general graph search algorithm (Fig. 3.7).

The uniform cost algorithm (Fig. 3.14) should be used when the frontier is a priority queue.
"""
function graph_search{T1 <: AbstractProblem, T2 <: Queue}(problem::T1, frontier::T2)
    local explored = Set{String}();
    push!(frontier, Node(problem.initial));
    while (length(frontier) != 0)
        local node = pop!(frontier);
        if (goal_test(problem, node.state))
            return node;
        end
        push!(explored, node.state);
        extend!(frontier, [child_node for child_node in expand(node, problem) 
                            if (!(child_node.state in explored) && !(child_node in frontier))])
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
    local node = Node(problem.initial);
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
