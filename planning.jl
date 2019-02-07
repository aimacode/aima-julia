
export AbstractPDDL,
        PDDL, goal_test, execute_action,
        AbstractPlanningAction, PlanningAction,
        substitute, check_precondition,
        air_cargo_pddl, air_cargo_pddl_goal_test,
        spare_tire_pddl, spare_tire_pddl_goal_test,
        three_block_tower_pddl, three_block_tower_pddl_goal_test,
        have_cake_and_eat_cake_too_pddl, have_cake_and_eat_cake_too_pddl_goal_test,
        PlanningLevel,
        find_mutex_links, build_level_links, build_level_links_permute_arguments, perform_actions,
        planning_combinations,
        PlanningGraph, expand_graph, non_mutex_goals,
        GraphPlanProblem, check_level_off, extract_solution,
        graphplan,
        doubles_tennis_pddl, doubles_tennis_pddl_goal_test,
        PlanningHighLevelAction, check_resource, check_job_order,
        HighLevelPDDL, refinements, hierarchical_search,
        job_shop_scheduling_pddl, job_shop_scheduling_pddl_goal_test;

abstract type AbstractPDDL end;

abstract type AbstractPlanningAction end;

#=

    PlanningAction is an action schema defined by the action's name, preconditions, and effects.

    Preconditions and effects consists of either positive and negated literals.

=#
struct PlanningAction <: AbstractPlanningAction
    name::String
    arguments::Tuple
    precondition_positive::Array{Expression, 1}
    precondition_negated::Array{Expression, 1}
    effect_add_list::Array{Expression, 1}
    effect_delete_list::Array{Expression, 1}

    function PlanningAction(action::Expression, precondition::Tuple{Vararg{Array, 2}}, effect::Tuple{Vararg{Array, 2}})
        return new(action.operator, action.arguments, precondition[1], precondition[2], effect[1], effect[2]);
    end
end

function substitute(action::T, e::Expression, arguments::Tuple{Vararg{Expression}}) where {T <: AbstractPlanningAction}
    local new_arguments::AbstractVector = collect(e.arguments);
    for (index_1, argument) in enumerate(e.arguments)
        for index_2 in 1:length(action.arguments)
            if (action.arguments[index_2] == argument)
                new_arguments[index_1] = arguments[index_2];
            end
        end
    end
    return Expression(e.operator, Tuple((new_arguments...,)));
end

function check_precondition(action::T1, kb::T2, arguments::Tuple) where {T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}
    # Check for positive clauses.
    for clause in action.precondition_positive
        if (!(substitute(action, clause, arguments) in kb.clauses))
            return false;
        end
    end
    # Check for negated clauses.
    for clause in action.precondition_negated
        if (substitute(action, clause, arguments) in kb.clauses)
            return false;
        end
    end
    return true;
end

"""
    execute_action(action, kb, arguments)

Execute the given action 'action' on the knowledge base of the planning problem.
"""
function execute_action(action::T1, kb::T2, arguments::Tuple) where {T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}
    if (!(check_precondition(action, kb, arguments)))
        error(@sprintf("execute_action(): Action \"%s\" preconditions are not satisfied!", action.name));
    end
    # Retract negated literals to knowledge base 'kb'.
    for clause in action.effect_delete_list
        retract(kb, substitute(action, clause, arguments));
    end
    # Add positive literals to knowledge base 'kb'.
    for clause in action.effect_add_list
        tell(kb, substitute(action, clause, arguments));
    end
    nothing;
end

#=

    The Planning Domain Definition Language (PDDL) is used to define a search problem.

    The states (starting from the initial state) are represented as the conjunction of

    the statements in 'kb' (a FirstOrderLogicKnowledgeBase). The actions are described

    by 'actions' (an array of action schemas). The 'goal_test' is a function that checks

    if the current state of the problem is at the goal state.

=#
struct PDDL <: AbstractPDDL
    kb::FirstOrderLogicKnowledgeBase
    actions::Array{PlanningAction, 1}
    goal_test::Function

    function PDDL(initial_state::Array{Expression, 1}, actions::Array{PlanningAction, 1}, goal_test::Function)
        return new(FirstOrderLogicKnowledgeBase(initial_state), actions, goal_test);
    end
end

function goal_test(plan::T) where {T <: AbstractPDDL}
    return plan.goal_test(plan.kb);
end

"""
    execute_action(plan, action)

Execute the first relevant PlanningAction associated with action 'action'
on the given planning problem's knowledgebase.
"""
function execute_action(plan::T, action::Expression) where {T <: AbstractPDDL}
    local action_name::String = action.operator;
    local arguments::Tuple = action.arguments;
    local relevant_actions::AbstractVector = collect(a for a in plan.actions if (a.name == action_name));
    if (length(relevant_actions) == 0)
        error(@sprintf("execute_action(): Action \"%s\" not found!", action_name));
    else
        local first_relevant_action::PlanningAction = relevant_actions[1];
        if (!check_precondition(first_relevant_action, plan.kb, arguments))
            error(@sprintf("execute_action(): Action \"%s\" preconditions are not satisfied!", repr(action)));
        else
            execute_action(first_relevant_action, plan.kb, arguments);
        end
    end
    nothing;
end

function air_cargo_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("At(C1, JFK)"), expr("At(C2, SFO)"))));
end

"""
    air_cargo_pddl()

Return a PDDL representing the air cargo transportation planning problem (Fig. 10.1).
"""
function air_cargo_pddl()
    local initial::Array{Expression, 1} = map(expr, ["At(C1, SFO)",
                                                "At(C2, JFK)",
                                                "At(P1, SFO)",
                                                "At(P2, JFK)",
                                                "Cargo(C1)",
                                                "Cargo(C2)",
                                                "Plane(P1)",
                                                "Plane(P2)",
                                                "Airport(JFK)",
                                                "Airport(SFO)"]);
    # Load Action Schema
    local precondition_positive::Array{Expression, 1} = map(expr, ["At(c, a)",
                                                            "At(p, a)",
                                                            "Cargo(c)",
                                                            "Plane(p)",
                                                            "Airport(a)"]);
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("In(c, p)")];
    local effect_delete_list::Array{Expression, 1} = [expr("At(c, a)")];
    local load::PlanningAction = PlanningAction(expr("Load(c, p, a)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # Unload Action Schema
    precondition_positive = map(expr, ["In(c, p)", "At(p, a)", "Cargo(c)", "Plane(p)", "Airport(a)"]);
    precondition_negated = [];
    effect_add_list = [expr("At(c, a)")];
    effect_delete_list = [expr("In(c, p)")];
    local unload::PlanningAction = PlanningAction(expr("Unload(c, p, a)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # Fly Action Schema
    precondition_positive = map(expr, ["At(p, f)", "Plane(p)", "Airport(f)", "Airport(to)"]);
    precondition_negated = [];
    effect_add_list = [expr("At(p, to)")];
    effect_delete_list = [expr("At(p, f)")];
    local fly::PlanningAction = PlanningAction(expr("Fly(p, f, to)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    return PDDL(initial, [load, unload, fly], air_cargo_pddl_goal_test);
end

function spare_tire_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("At(Spare, Axle)"),)));
end

"""
    spare_tire_pddl()

Return a PDDL representing the spare tire planning problem (Fig. 10.2).
"""
function spare_tire_pddl()
    local initial::Array{Expression, 1} = map(expr, ["Tire(Flat)",
                                                    "Tire(Spare)",
                                                    "At(Flat, Axle)",
                                                    "At(Spare, Trunk)"]);
    # Remove Action Schema
    local precondition_positive::Array{Expression, 1} = [expr("At(obj, loc)")];
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("At(obj, Ground)")];
    local effect_delete_list::Array{Expression, 1} = [expr("At(obj, loc)")];
    local remove::PlanningAction = PlanningAction(expr("Remove(obj, loc)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # PutOn Action Schema
    precondition_positive = map(expr, ["Tire(t)", "At(t, Ground)"]);
    precondition_negated = [expr("At(Flat, Axle)")];
    effect_add_list = [expr("At(t, Axle)")];
    effect_delete_list = [expr("At(t, Ground)")];
    local put_on::PlanningAction = PlanningAction(expr("PutOn(t, Axle)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    # LeaveOvernight Action Schema
    precondition_positive = [];
    precondition_negated = [];
    effect_add_list = [];
    effect_delete_list = map(expr, ["At(Spare, Ground)", "At(Spare, Axle)", "At(Spare, Trunk)",
                                    "At(Flat, Ground)", "At(Flat, Axle)", "At(Flat, Trunk)"]);
    local leave_overnight::PlanningAction = PlanningAction(expr("LeaveOvernight"),
                                                            (precondition_positive, precondition_negated),
                                                            (effect_add_list, effect_delete_list));
    return PDDL(initial, [remove, put_on, leave_overnight], spare_tire_pddl_goal_test);
end

function three_block_tower_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   #length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("On(A, B)"), expr("On(B, C)"))));
end

"""
    three_block_tower_pddl()

Return a PDDL representing the building of a three-block tower planning problem (Fig. 10.3).
"""
function three_block_tower_pddl()
    local initial::Array{Expression, 1} = map(expr, ["On(A, Table)",
                                                    "On(B, Table)",
                                                    "On(C, A)",
                                                    "Block(A)",
                                                    "Block(B)",
                                                    "Block(C)",
                                                    "Clear(B)",
                                                    "Clear(C)"]);
    # Move Action Schema
    local precondition_positive::Array{Expression, 1} = map(expr, ["On(b, x)", "Clear(b)", "Clear(y)", "Block(b)", "Block(y)"]);
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("On(b, y)"), expr("Clear(x)")];
    local effect_delete_list::Array{Expression, 1} = [expr("On(b, x)"), expr("Clear(y)")];
    local move::PlanningAction = PlanningAction(expr("Move(b, x, y)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # MoveToTable Action Schema
    precondition_positive = map(expr, ["On(b, x)", "Clear(b)", "Block(b)"]);
    precondition_negated = [];
    effect_add_list = [expr("On(b, Table)"), expr("Clear(x)")];
    effect_delete_list = [expr("On(b, x)")];
    local move_to_table::PlanningAction = PlanningAction(expr("MoveToTable(b, x)"),
                                                        (precondition_positive, precondition_negated),
                                                        (effect_add_list, effect_delete_list));
    return PDDL(initial, [move, move_to_table], three_block_tower_pddl_goal_test);
end

function have_cake_and_eat_cake_too_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("Have(Cake)"), expr("Eaten(Cake)"))));
end

"""
    have_cake_and_eat_cake_too_pddl()

Return a PDDL representing the 'have cake and eat cake too' planning problem (Fig. 10.7).
"""
function have_cake_and_eat_cake_too_pddl()
    local initial::Array{Expression, 1} = [expr("Have(Cake)")];
    # Eat Cake Action Schema
    local precondition_positive::Array{Expression, 1} = [expr("Have(Cake)")];
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("Eaten(Cake)")];
    local effect_delete_list::Array{Expression, 1} = [expr("Have(Cake)")];
    local eat_cake::PlanningAction = PlanningAction(expr("Eat(Cake)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    # Bake Cake Action Schema
    precondition_positive = [];
    precondition_negated = [expr("Have(Cake)")];
    effect_add_list = [expr("Have(Cake)")];
    effect_delete_list = [];
    local bake_cake::PlanningAction = PlanningAction(expr("Bake(Cake)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    return PDDL(initial, [eat_cake, bake_cake], have_cake_and_eat_cake_too_pddl_goal_test);
end

mutable struct PlanningLevel
    positive_kb::FirstOrderLogicKnowledgeBase
    current_state_positive::Array{Expression, 1}    #current state of the planning problem
    current_state_negated::Array{Expression, 1}     #current state of the planning problem
    current_action_links_positive::Dict             #current actions to current state link
    current_action_links_negated::Dict              #current actions to current state link
    current_state_links_positive::Dict              #current state to action link
    current_state_links_negated::Dict               #current state to action link
    next_action_links::Dict                         #current action to next state link
    next_state_links_positive::Dict                 #next state to current action link
    next_state_links_negated::Dict                  #next state to current action link
    mutex_links::Array{Set, 1}                      #each mutex relation is a Set of 2 actions/literals

    function PlanningLevel(p_kb::FirstOrderLogicKnowledgeBase, n_kb::FirstOrderLogicKnowledgeBase)
        return new(p_kb, p_kb.clauses, n_kb.clauses, Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), []);
    end
end

function find_mutex_links(level::PlanningLevel)
    # Inconsistent effects condition between 2 action schemas at a given level
    for positive_effect in level.next_state_links_positive
        negated_effect = positive_effect;
        if (haskey(level.next_state_links_negated, negated_effect))
            for a in level.next_state_links_positive[positive_effect]
                for b in level.next_state_links_negated[negated_effect]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Inference condition between 2 action schemas at a given level
    for positive_precondition in level.current_state_links_positive
        negated_effect = positive_precondition;
        if (haskey(level.next_state_links_negated, negated_effect))
            for a in level.current_state_links_positive[positive_precondition]
                for b in level.next_state_links_negated[negated_effect]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    for negated_precondition in level.current_state_links_negated
        positive_effect = negated_precondition;
        if (haskey(level.next_state_links_positive, positive_effect))
            for a in level.next_state_links_positive[positive_effect]
                for b in level.current_state_links_negated[negated_precondition]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Competing needs condition between 2 action schemas
    for positive_precondition in level. current_state_links_positive
        negated_precondition = positive_precondition;
        if (haskey(level.current_state_links_negated, negated_precondition))
            for a in level.current_state_links_positive[positive_precondition]
                for b in level.current_state_links_negated[negated_precondition]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Inconsistent support condition
    local state_mutex_links::AbstractVector = [];
    for pair in level.mutex_links
        collected_pair::AbstractVector = collect(pair);
        next_state_1 = level.next_action_links[collected_pair[1]];
        if (length(sorted_pair) == 2)
            next_state_2 = level.next_action_links[collected_pair[2]];
        else
            next_state_2 = level.next_action_links[collected_pair[1]];
        end
        if ((length(next_state_1) == 1) && (length(next_state_2) == 1))
            push!(state_mutex_links, Set([next_state_1[1], next_state_2[1]]));
        end
    end
    level.mutex_links = vcat(level.mutex_links, state_mutex_links);
    nothing;
end

function build_level_links_permute_arguments(depth::Int64, objects::AbstractVector, current_permutation::Tuple, permutations_array::AbstractVector)
    if (depth == 0)
        push!(permutations_array, current_permutation);
    elseif (depth < 0)
        error("build_level_links_permute_arguments(): Found negative depth!");
    else
        for (i, item) in enumerate(objects)
            build_level_links_permute_arguments((depth - 1),
                                                Tuple((objects[1:(i - 1)]..., objects[(i + 1):end]...,)),
                                                Tuple((current_permutation..., item)),
                                                permutations_array)
        end
    end
end

function build_level_links_permute_arguments(depth::Int64, objects::Tuple, current_permutation::Tuple, permutations_array::AbstractVector)
    if (depth == 0)
        push!(permutations_array, current_permutation);
    elseif (depth < 0)
        error("build_level_links_permute_arguments(): Found negative depth!");
    else
        for (i, item) in enumerate(objects)
            build_level_links_permute_arguments((depth - 1),
                                                Tuple((objects[1:(i - 1)]..., objects[(i + 1):end]...,)),
                                                Tuple((current_permutation..., item)),
                                                permutations_array)
        end
    end
end

function build_level_links(level::PlanningLevel, actions::AbstractVector, objects::Set)
    # Create persistence actions for positive states
    for clause in level.current_state_positive
        level.current_action_links_positive[Expression("Persistence", clause)] = [clause];
        level.next_action_links[Expression("Persistence", clause)] = [clause];
        level.current_state_links_positive[clause] = [Expression("Persistence", clause)];
        level.next_state_links_positive[clause] = [Expression("Persistence", clause)];
    end
    # Create persistence actions for negated states
    for clause in level.current_state_negated
        not_expression = Expression("not"*clause.operator, clause.arguments);
        level.current_action_links_negated[Expression("Persistence", not_expression)] = [clause];
        level.next_action_links[Expression("Persistence", not_expression)] = [clause];
        level.current_state_links_negated[clause] = [Expression("Persistence", not_expression)];
        level.next_state_links_negated[clause] = [Expression("Persistence", not_expression)];
    end
    # Recursively collect num_arg depth, collecting a Tuple of Tuples
    for action in actions
        local num_arguments::Int64 = length(action.arguments);
        local possible_arguments::AbstractVector = [];
        build_level_links_permute_arguments(num_arguments, collect(objects), (), possible_arguments);
        for argument in possible_arguments
            if (check_precondition(action, level.positive_kb, argument))
                for (number, symbol) in enumerate(action.arguments)
                    if (!all(islowercase, symbol.operator))
                        argument = Tuple((argument[1:(number - 1)]..., symbol, argument[(number + 1):end]...,));
                    end
                end
                local new_action::Expression = substitute(action, Expression(action.name, action.arguments), argument);
                level.current_action_links_positive[new_action] = [];
                level.current_action_links_negated[new_action] = [];
                local new_clause::Expression;
                for clause in action.precondition_positive
                    new_clause = substitute(action, clause, argument);
                    push!(level.current_action_links_positive[new_action], new_clause);
                    if (haskey(level.current_state_links_positive, new_clause))
                        push!(level.current_state_links_positive[new_clause], new_action);
                    else
                        level.current_state_links_positive[new_clause] = [new_action];
                    end
                end
                for clause in action.precondition_negated
                    new_clause = substitute(action, clause, argument);
                    push!(level.current_action_links_negated[new_action], new_clause);
                    if (haskey(level.current_state_links_negated, new_clause))
                        push!(level.current_state_links_negated[new_clause], new_action);
                    else
                        level.current_state_links_negated[new_clause] = [new_action];
                    end
                end
                level.next_action_links[new_action] = [];
                for clause in action.effect_add_list
                    new_clause = substitute(action, clause, argument);
                    push!(level.next_action_links[new_action], new_clause);
                    if (haskey(level.next_state_links_positive, new_clause))
                        push!(level.next_state_links_positive[new_clause], new_action);
                    else
                        level.next_state_links_positive[new_clause] = [new_action];
                    end
                end
                for clause in action.effect_delete_list
                    new_clause = substitute(action, clause, argument);
                    push!(level.next_action_links[new_action], new_clause);
                    if (haskey(level.next_state_links_negated, new_clause))
                        push!(level.next_state_links_negated[new_clause], new_action);
                    else
                        level.next_state_links_negated[new_clause] = [new_action];
                    end
                end
            end
        end
    end
    nothing;
end

function perform_actions(level::PlanningLevel)
    local new_kb_positive::FirstOrderLogicKnowledgeBase = FirstOrderLogicKnowledgeBase(Array{Expression, 1}(collect(Set(collect(keys(level.next_state_links_positive))))));
    local new_kb_negated::FirstOrderLogicKnowledgeBase = FirstOrderLogicKnowledgeBase(Array{Expression, 1}(collect(Set(collect(keys(level.next_state_links_negated))))));
    return PlanningLevel(new_kb_positive, new_kb_negated);
end

function planning_combinations(items::AbstractVector) #ordered permutations of length 2
    local combinations::AbstractVector = [];
    for (i, a) in enumerate(items)
        for b in items[(i + 1):end]
            push!(combinations, (a, b));
        end
    end
    return combinations;
end

#=

    PlanningGraph is an implementation of a planning graph data structure.

    The planning graph is organized into levels based on the given PDDL

    and negated knowledge base.

=#
struct PlanningGraph
    pddl::AbstractPDDL
    levels::Array{PlanningLevel, 1}
    objects::Set{Expression}

    function PlanningGraph(pddl::T, n_kb::FirstOrderLogicKnowledgeBase) where {T <: AbstractPDDL}
        return new(pddl, [PlanningLevel(pddl.kb, n_kb)], Set(collect(arg for clause in vcat(pddl.kb.clauses, n_kb.clauses) for arg in clause.arguments)));
    end
end

function expand_graph(pg::PlanningGraph)
    local last_level = pg.levels[length(pg.levels)];
    build_level_links(last_level, pg.pddl.actions, pg.objects);
    find_mutex_links(last_level);
    push!(pg.levels, perform_actions(last_level));
    nothing;
end

function non_mutex_goals(pg::PlanningGraph, goals::AbstractVector, index::Int64)
    local goal_combinations::AbstractVector = planning_combinations(goals);
    for goal in goal_combinations
        if (index < 0)
            if (Set(collect(goal)) in reverse(pg.levels)[abs(index)].mutex_links)
                return false;
            end
        else
            error("non_mutex_goals(): Expected negative index, got ", index, "!");
        end
    end
    return true;
end

#=

    GraphPlanProblem is a data structure that stores the 'graph', 'nogoods',

    and 'solution' variables of a given Graphplan planning problem.

=#
struct GraphPlanProblem
    graph::PlanningGraph
    nogoods::AbstractVector
    solution::AbstractVector

    function GraphPlanProblem(pddl::AbstractPDDL, n_kb::FirstOrderLogicKnowledgeBase)
        return new(PlanningGraph(pddl, n_kb), [], []);
    end
end

function check_level_off(gpp::GraphPlanProblem)
    local first_check::Bool = (Set(reverse(gpp.graph.levels)[1].current_state_positive) == Set(reverse(gpp.graph.levels)[2].current_state_positive));
    local second_check::Bool = (Set(reverse(gpp.graph.levels)[1].current_state_negated) == Set(reverse(gpp.graph.levels)[2].current_state_negated));
    return (first_check && second_check);
end

function extract_solution(gpp::GraphPlanProblem, goals_positive::AbstractVector, goals_negated::AbstractVector, index::Int64)
    local level::PlanningLevel;
    if (index < 0)
        level = reverse(gpp.graph.levels)[abs(index)];
    else
        error("extract_solution(): Expected negative index, got ", index, "!");
    end
    if (!non_mutex_goals(gpp.graph, vcat(goals_positive, goals_negated), index))
        push!(gpp.nogoods, (level, goals_positive, goals_negated));
        return nothing;
    end
    level = reverse(gpp.graph.levels)[abs(index - 1)];
    local actions::AbstractVector = [];
    for goal in goals_positive
        push!(actions, level.next_state_links_positive[goal]);
    end
    for goal in goals_negated
        push!(actions, level.next_state_links_negated[goal]);
    end

    # Create all possible combinations of actions by using finding the cartesian product.
    local action_combinations::AbstractVector = iterable_cartesian_product(actions);
    # Remove action combinations that contain mutexes.
    local non_mutex_actions::AbstractVector = [];
    for action_list in action_combinations
        action_pairs = planning_combinations(collect(Set(action_list)));
        push!(non_mutex_actions, collect(Set(action_list)));
        for pair in action_pairs
            if (Set(collect(pair)) in level.mutex_links)
                pop!(non_mutex_actions);
                break;
            end
        end
    end

    for action_list in non_mutex_actions
        if (!([action_list, index] in gpp.solution))
            push!(gpp.solution, [action_list, index]);
            local new_goals_positive::AbstractVector = [];
            local new_goals_negated::AbstractVector = [];
            for action in Set(action_list)
                if (haskey(level.current_action_links_positive, action))
                    new_goals_positive = vcat(new_goals_positive, level.current_action_links_positive[action]);
                end
                if (haskey(level.current_action_links_negated, action))
                    new_goals_negated = vcat(new_goals_negated, level.current_action_links_negated[action]);
                end
            end
            if ((abs(index) + 1) == length(gpp.graph.levels))
                return nothing;
            elseif ((level, new_goals_positive, new_goals_negated) in gpp.nogoods)
                return nothing;
            else
                extract_solution(gpp, new_goals_positive, new_goals_negated, (index - 1))
            end
        end
    end
    local solution::AbstractVector = [];
    for item in gpp.solution
        if (item[2] == -1)
            push!(solution, [item[1]]);
        else
            push!(reverse(solution)[1], item[1]);
        end
    end
    for (i, item) in enumerate(solution)
        solution[i] = reverse(item);
    end
    return solution;
end

function goal_test(gpp::GraphPlanProblem, goals::AbstractVector)    #goal_test() for graphplan()
    local positive_kb::FirstOrderLogicKnowledgeBase = reverse(gpp.graph.levels)[1].positive_kb;
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(positive_kb, goal) for goal in goals));
end

"""
    graphplan(gpp::GraphPlanProblem, goals::Tuple)


Apply the Graphplan algorithm (Fig. 10.9) to the given planning problem 'gpp' and goal state 'goals'.
Return the solution or 'nothing' on failure.
"""
function graphplan(gpp::GraphPlanProblem, goals::Tuple)
    local goals_positive::AbstractVector = goals[1];
    local goals_negated::AbstractVector = goals[2];
    while (true)
        if (goal_test(gpp, goals_positive) && non_mutex_goals(gpp.graph, vcat(goals_positive, goals_negated), -1))
            solution = extract_solution(gpp, goals_positive, goals_negated, -1);
            if (!(typeof(solution) <: Nothing))
                return solution;
            end
        end
        expand_graph(gpp.graph);
        if ((length(gpp.graph.levels) > 1) && check_level_off(gpp))
            return nothing;
        end
    end
    return nothing;
end

function doubles_tennis_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   #length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in map(expr, ["Returned(Ball)", "At(a, RightNet)", "At(a, LeftNet)"])));
end

"""
    doubles_tennis_pddl()

Return a PDDL representing the doubles tennis planning problem (Fig. 11.10).

The PlanningActions 'hit' and 'go' both take 3 arguments because their
preconditions will not be satisfied without the 'loc' variable.
This occurs when substituting propositional logic variables within
the check_precondition() function.
"""
function doubles_tennis_pddl()
    local initial::Array{Expression, 1} = map(expr, ["At(A, LeftBaseLine)",
                                                    "At(B, RightNet)",
                                                    "Approaching(Ball, RightBaseLine)",
                                                    "Partner(A, B)",
                                                    "Partner(B, A)"]);
    # Hit Action Schema
    local precondition_positive::Array{Expression, 1} = map(expr, ["Approaching(Ball, loc)", "At(actor, loc)"]);
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("Returned(Ball)")];
    local effect_delete_list::Array{Expression, 1} = [];
    local hit::PlanningAction = PlanningAction(expr("Hit(actor, loc, Ball)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # Go Action Schema
    precondition_positive = [expr("At(actor, loc)")];
    precondition_negated = [];
    effect_add_list = [expr("At(actor, to)")];
    effect_delete_list = [expr("At(actor, loc)")];
    local go::PlanningAction = PlanningAction(expr("Go(actor, loc, to)"),
                                            (precondition_positive, precondition_negated),
                                            (effect_add_list, effect_delete_list));
    return PDDL(initial, [hit, go], doubles_tennis_pddl_goal_test);
end

#=

    PlanningHighLevelAction (HLA) is an action schema with resource constraints and refinements.

=#
mutable struct PlanningHighLevelAction <: AbstractPlanningAction
    name::String
    arguments::Tuple
    precondition_positive::Array{Expression, 1}
    precondition_negated::Array{Expression, 1}
    effect_add_list::Array{Expression, 1}
    effect_delete_list::Array{Expression, 1}
    duration::Int64
    consumes::Dict
    uses::Dict
    completed::Bool
    priority::Int64     #undefined field used to schedule multiple HLAs
    job_group::Int64    #undefined field used to schedule multiple HLAs

    function PlanningHighLevelAction(action::Expression,
                                    precondition::Tuple{Vararg{Array{Expression, 1}, 2}},
                                    effect::Tuple{Vararg{Array{Expression, 1}, 2}};
                                    duration::Int64=0,
                                    consumes::Dict=Dict(),
                                    uses::Dict=Dict())
        return new(action.operator, action.arguments, precondition[1], precondition[2], effect[1], effect[2], duration, consumes, uses, false);
    end
end

"""
    execute_action(hla, job_order, available_resources, kb, arguments)

Execute the given action 'hla' on the knowledge base of the planning problem.
This function checks for resources and job order execution required for the action to be executed.
"""
function execute_action(hla::PlanningHighLevelAction, job_order::Array{Array{PlanningHighLevelAction, 1}, 1}, available_resources::Dict, kb::AbstractKnowledgeBase, arguments::Tuple)
    if (!check_consumes_resource(hla, available_resources))
        error(@sprintf("execute_action(): High-level action \"%s\"'s consumable constraint was not satisfied!", hla.name));
    elseif (!check_uses_resource(hla, available_resources))
        error(@sprintf("execute_action(): High-level action \"%s\"'s reusable constraint was not satisfied!", hla.name));
    elseif (!check_job_order(hla, job_order))
        error(@sprintf("execute_action(): High-level action \"%s\" is not scheduled to execute! All scheduled unexecuted actions should run first!", hla.name));
    end

    # Execute 'hla' like a primitive action within this method to avoid confusion of
    # calling execute_action() within execute_action().

    if (!(check_precondition(hla, kb, arguments)))
        error(@sprintf("execute_action(): High-level action \"%s\" preconditions are not satisfied!", hla.name));
    end
    # Retract negated literals to knowledge base 'kb'.
    for clause in hla.effect_delete_list
        retract(kb, substitute(hla, clause, arguments));
    end
    # Add positive literals to knowledge base 'kb'.
    for clause in hla.effect_add_list
        tell(kb, substitute(hla, clause, arguments));
    end

    for resource in keys(hla.consumes)
        available_resources[resource] = available_resources[resource] - hla.consumes[resource];
    end

    # Update current high-level action task as completed.
    hla.completed = true;
    nothing;
end

function check_consumes_resource(hla::PlanningHighLevelAction, available_resources::Dict)
    for resource in keys(hla.consumes)
        if (!haskey(available_resources, resource))
            return false;
        elseif (available_resources[resource] < (hla.consumes)[resource])
            return false;
        end
    end
    return true;
end

function check_uses_resource(hla::PlanningHighLevelAction, available_resources::Dict)
    for resource in keys(hla.uses)
        if (!haskey(available_resources, resource))
            return false;
        elseif (available_resources[resource] < (hla.uses)[resource])
            return false;
        end
    end
    return true;
end

function check_job_order(hla::PlanningHighLevelAction, job_order::Array{Array{PlanningHighLevelAction, 1}, 1})
    for jobs in job_order
        if (hla in jobs)
            for job in jobs
                if (job == hla)
                    return true;
                elseif (!job.completed)
                    return false;
                end
            end
        end
    end
    return true;
end

#=

    HighLevelPDDL is a high-level Planning Domain Definition Language (PDDL) and is used to

    define a search problem. The HighLevelPDDL includes job scheduling and resource constraints

    in addition to the states, action schemas, and the goal test found in the PDDL datatype.

=#
struct HighLevelPDDL <: AbstractPDDL 
    kb::FirstOrderLogicKnowledgeBase
    actions::Array{PlanningHighLevelAction, 1}
    goal_test::Function
    jobs::Array{Array{PlanningHighLevelAction, 1}, 1}
    resources::Dict

    function HighLevelPDDL(initial_state::Array{Expression, 1}, actions::Array{PlanningHighLevelAction, 1}, goal_test::Function; jobs::Array{Array{PlanningHighLevelAction, 1}, 1}=Array{Array{PlanningHighLevelAction, 1}, 1}(), resources::Dict=Dict())
        return new(FirstOrderLogicKnowledgeBase(initial_state), actions, goal_test, jobs, resources);
    end
end

"""
    execute_action(plan, action)

Execute the first relevant PlanningHighLevelAction associated with action 'action'
on the given planning problem's knowledge base.
"""
function execute_action(plan::HighLevelPDDL, action::PlanningHighLevelAction)
    local arguments::Tuple = action.arguments;
    local relevant_actions::AbstractVector = collect(a for a in plan.actions if (a.name == action.name));
    if (length(relevant_actions) == 0)
        error(@sprintf("execute_action(): High-level action \"%s\" not found!", action.name));
    else
        local first_relevant_action::PlanningHighLevelAction = relevant_actions[1];
        execute_action(first_relevant_action, plan.jobs, plan.resources, plan.kb, arguments);
    end
    nothing;
end

"""
    get_result(plan, action)

Return the planning problem after executing the given high-level action 'action'.
"""
function get_result(plan::HighLevelPDDL, action::PlanningHighLevelAction)
    execute_action(plan, action);
    return plan;
end

function get_result(plan::HighLevelPDDL, action::Nothing)
    return plan;
end

"""
    refinements(hla, state, dict)

Return a action sequence such that each refinement's preconditions of 'hla' are satisfied by
the given state of the planning problem 'state'.
"""
function refinements(hla::PlanningHighLevelAction, state::HighLevelPDDL, dict::Dict)
    local solution::Array{PlanningHighLevelAction, 1} = Array{PlanningHighLevelAction, 1}();
    local indices::AbstractVector = collect(i for (i, x) in enumerate(dict["HLA"]) if (expr(x).operator == hla.name));
    for i in indices
        local action::PlanningHighLevelAction = PlanningHighLevelAction(expr(dict["steps"][i][1]),
                                                                        (collect(expr(x) for x in dict["precondition_positive"][i]), collect(expr(x) for x in dict["precondition_negated"][i])),
                                                                        (collect(expr(x) for x in dict["effect_add_list"][i]), collect(expr(x) for x in dict["effect_delete_list"][i])));
        if (check_precondition(hla, state.kb, action.arguments))
            push!(solution, action);
        end
    end
    return solution;
end

"""
    hierarchical_search(problem, hierarchy)

Use the breadth-first implementation of the Hierarchical Search algorithm (Fig. 11.5) to the 
given high-level planning problem 'problem' and 'hierarchy'. Return the solution or 'nothing' 
on failure.
"""
function hierarchical_search(problem::HighLevelPDDL, hierarchy)
    local act::Node = Node(problem.actions[1]);
    local frontier::FIFOQueue();
    push!(frontier, act);
    while (true)
        if (length(frontier) == 0)
            return nothing;
        end
        local plan = pop!(frontier);
        hla = plan.state;
        if (!(plan.parent === nothing))
            prefix = plan.parent.action;
        else
            prefix = nothing;
        end
        local outcome::HighLevelPDDL = get_result(problem, prefix);
        if (typeof(hla) <: Nothing)
            if (goal_test(outcome))
                return path(plan);
            end
        else
            for sequence in refinements(hla, outcome, hierachy)
                push!(frontier, Node(plan.state, parent=plan.parent, action=sequence));
            end
        end
    end
    nothing;
end

function job_shop_scheduling_pddl_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in map(expr, ["Has(C1, W1)", "Has(C1, E1)", "Inspected(C1)",
                                                        "Has(C2, W2)", "Has(C2, E2)", "Inspected(C2)",
                                                        "Car(C1)", "Car(C2)",
                                                        "Wheels(W1)", "Wheels(W2)",
                                                        "Engine(E1)", "Engine(E2)"])));
end

"""
    job_shop_scheduling_pddl()

Return a HighLevelPDDL representing the job-shop scheduling planning problem (Fig. 11.1).
"""
function job_shop_scheduling_pddl()
    local initial::Array{Expression, 1} = map(expr, ["Car(C1)", "Car(C2)",
                                                    "Wheels(W1)", "Wheels(W2)",
                                                    "Engine(E1)", "Engine(E2)"]);
    local resources::Dict = Dict([Pair("EngineHoists", 1),
                                Pair("WheelStations", 2),
                                Pair("Inspectors", 2),
                                Pair("LugNuts", 500)]);
    # AddEngine1 high-level action schema
    local precondition_positive::Array{Expression, 1} = [];
    local precondition_negated::Array{Expression, 1} = [expr("Has(C1, E1)")];
    local effect_add_list::Array{Expression, 1} = [expr("Has(C1, E1)")];
    local effect_delete_list::Array{Expression, 1} = [];
    local add_engine_1::PlanningHighLevelAction = PlanningHighLevelAction(expr("AddEngine1"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=30, uses=Dict([Pair("EngineHoists", 1)]));
    # AddEngine2 high-level action schema
    precondition_positive = [];
    precondition_negated = [expr("Has(C2, E2)")];
    effect_add_list = [expr("Has(C2, E2)")];
    effect_delete_list = [];
    local add_engine_2::PlanningHighLevelAction = PlanningHighLevelAction(expr("AddEngine2"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=60, uses=Dict([Pair("EngineHoists", 1)]));
    # AddWheels1 high-level action schema
    precondition_positive = [];
    precondition_negated = [expr("Has(C1, W1)")];
    effect_add_list = [expr("Has(C1, W1)")];
    effect_delete_list = [];
    local add_wheels_1::PlanningHighLevelAction = PlanningHighLevelAction(expr("AddWheels1"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=30, consumes=Dict([Pair("LugNuts", 20)]), uses=Dict([Pair("WheelStations", 1)]));
    # AddWheels2 high-level action schema
    precondition_positive = [];
    precondition_negated = [expr("Has(C2, W2)")];
    effect_add_list = [expr("Has(C2, W2)")];
    effect_delete_list = [];
    local add_wheels_2::PlanningHighLevelAction = PlanningHighLevelAction(expr("AddWheels2"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=15, consumes=Dict([Pair("LugNuts", 20)]), uses=Dict([Pair("WheelStations", 1)]));
    # Inspect1 high-level action schema
    precondition_positive = [];
    precondition_negated = [expr("Inspected(C1)")];
    effect_add_list = [expr("Inspected(C1)")];
    effect_delete_list = [];
    local inspect_1::PlanningHighLevelAction = PlanningHighLevelAction(expr("Inspect1"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=10, uses=Dict([Pair("Inspectors", 1)]));
    # Inspect2 high-level action schema
    precondition_positive = [];
    precondition_negated = [expr("Inspected(C2)")];
    effect_add_list = [expr("Inspected(C2)")];
    effect_delete_list = [];
    local inspect_2::PlanningHighLevelAction = PlanningHighLevelAction(expr("Inspect2"),
                                                                        (precondition_positive, precondition_negated),
                                                                        (effect_add_list, effect_delete_list),
                                                                        duration=10, uses=Dict([Pair("Inspectors", 1)]));
    local job_group_1::Array{PlanningHighLevelAction, 1} = [add_engine_1, add_wheels_1, inspect_1];
    local job_group_2::Array{PlanningHighLevelAction, 1} = [add_engine_2, add_wheels_2, inspect_2];
    return HighLevelPDDL(initial, [add_engine_1, add_engine_2, add_wheels_1, add_wheels_2, inspect_1, inspect_2], job_shop_scheduling_pddl_goal_test, jobs=[job_group_1, job_group_2], resources=resources);
end
