include("../aimajulia.jl");
using aimajulia;
using aimajulia.utils;

function beautify_node_args(n)
    if (typeof(n) <: Node)
        return beautify_node(n);
    elseif (typeof(n) <: Tuple)
        return map(beautify_node_args, n);
    else
        return n;
    end
end

function bfgs{T <: AbstractProblem}(problem::T, f::Function)
    mf = MemoizedFunction(f);
    local node = Node{typeof(problem.initial)}(problem.initial);
    if (goal_test(problem, node.state))
        return node;
    end
    local frontier = PQueue();
    push!(frontier, node, mf);
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
                push!(frontier, child_node, mf);
            elseif (child_node in [getindex(x, 2) for x in frontier.array])
            #Recall that Nodes can share the same state and different values for other fields.
                local existing_node = pop!(collect(getindex(x, 2)
                                                    for x in frontier.array
                                                    if (getindex(x, 2) == child_node)));

                eval_memoized_function(mf, child_node);
                eval_memoized_function(mf, existing_node);

                if (eval_memoized_function(mf, child_node) < eval_memoized_function(mf, existing_node))
                    delete!(frontier, existing_node);
                    push!(frontier, child_node, mf);
                end
            end
        end
        print("length of memoization dictionary: ", length(mf.values), " ");
        println(map(beautify_node_args, collect(keys(mf.values)))...);
    end
    return nothing;
end

function astar_memoized_search(problem::GraphProblem; h::Union{Void, Function}=nothing)
    local mh::MemoizedFunction; #memoized h(n) function
    if (!(typeof(h) <: Void))
        mh = h;
    else
        mh = problem.h;
    end
    return bfgs(problem,
                                    (function(node::Node; h::MemoizedFunction=mh, prob::GraphProblem=problem)
                                        return node.path_cost + eval_memoized_function(h, prob, node);end));
end

astar_str="";for i in 1:5
	if (Node{String}("P")==get(astar_memoized_search(GraphProblem("A", "B", aimajulia.romania)).parent))
		astar_str = astar_str * "1";
        println("Test: Passed!");
	else
		astar_str = astar_str * "0";
        println("Test: Failed!");
	end
end;println(count(i->(i=='1'), astar_str), " of 5 tries passed!");
