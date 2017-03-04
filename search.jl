
import Base: ==;

typealias Action String;

typealias Percept Tuple{Any, Any};

abstract AbstractProblem;

type Problem <: AbstractProblem
    initial::String
    goal::String

    function Problem(initial_state; goal_state=Void)
        if (typeof(goal_state) <: String)
            return new(initial_state, goal_state);
        else
            return new(initial_state, "");
        end
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
    action::Action
    parent::Node

    function Node(state; parent=Void, action=Void, path_cost=0)
        if (typeof(parent) <: Node && typeof(action) <: Action)
            nn = new(state, path_cost, UInt32(0), action);
            nn.parent = parent;
            nn.depth = parent.depth + 1;
            return nn;
        else
            if (typeof(parent) <: Node)
                nn = new(state, path_cost, UInt32(0), "");
                nn.parent = parent;
                nn.depth = parent.depth + 1;
                return nn;
            elseif (typeof(action) <: Action)
                nn = new(state, path_cost, UInt32(0), action);
                return nn;  #leave parent undefined
            else
                nn = new(state, path_cost, UInt32(0), "");
                return nn;  #leave parent undefined
            end
        end
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
        try
            node = node.parent;
        catch e     #our root node's parent is undefined
            break;
        end
    end
    path_back = reverse(path_back);
    return path_back;
end

==(n1::Node, n2::Node) = (n1.state == n2.state);

type SimpleProblemSolvingAgentProgram
    state::String
    goal::String
    seq::Array{Action, 1}
    problem::Problem

    function SimpleProblemSolvingAgentProgram(;initial_state=Void)
        if (typeof(initial_state) <: String)
            return new(initial_state, "", Array{Action, 1}());
        else
            return new("", "", Array{Action, 1}());
        end
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

    function GAState(genes)
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
