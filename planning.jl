
export AbstractPDDL,
        PDDL, goal_test, execute_action,
        AbstractPlanningAction, PlanningAction,
        substitute, check_precondition;

abstract AbstractPDDL;

abstract AbstractPlanningAction;

#=

    PlanningAction is an action schema defined by the action's name, preconditions, and effects.

    Preconditions and effects consists of either positive and negated literals.

=#
type PlanningAction <: AbstractPlanningAction
    name::String
    arguments::Tuple
    precondition_positive::Array{Expression, 1}
    precondition_negated::Array{Expression, 1}
    effect_add_list::Array{Expression, 1}
    effect_delete_list::Array{Expression, 1}

    function PlanningAction(action::Expression, precondition::Tuple{Vararg{Array{Expression, 1}, 2}}, effect::Tuple{Vararg{Array{Expression, 1}, 2}})
        return new(action.operator, action.arguments, precondition[1], precondition[2], effect[1], effect[2]);
    end
end

function substitute{T <: AbstractPlanningAction}(action::T, e::Expression, arguments::Tuple{Vararg{Expression}})
    local new_arguments::AbstractVector = collect(e.arguments);
    for (index_1, argument) in enumerate(e.arguments)
        for index_2 in 1:length(action.arguments)
            if (action.arguments[index_2] == argument)
                new_arguments[index_1] = arguments[index_2];
            end
        end
    end
    return Expression(e.operator, Tuple((new_arguments...)));
end

function check_precondition{T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}(action::T1, kb::T2, arguments::Tuple)
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

function execute_action{T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}(action::T1, kb::T2, arguments::Tuple)
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
type PDDL <: AbstractPDDL
    kb::FirstOrderLogicKnowledgeBase
    actions::Array{PlanningAction, 1}
    goal_test::Function

    function PDDL(initial_state::Array{Expression, 1}, actions::Array{PlanningAction, 1}, goal_test::Function)
        return new(FirstOrderLogicKnowledgeBase(initial_state), actions, goal_test);
    end
end

function goal_test{T <: AbstractPDDL}(plan::T)
    return plan.goal_test(plan.kb);
end

function execute_action{T <: AbstractPDDL}(plan::T, action::Expression)
    local action_name::String = action.operator;
    local arguments::Tuple = action.arguments;
    local relevant_actions::AbstractVector = collect(a for a in plan.actions if (a.name == action_name));
    if (length(relevant_actions) == 0)
        error(@sprintf("execute_action(): Action \"%s\" not found!", action_name));
    else
        local first_relevant_action::PlanningAction = first(a for a in plan.actions if (a.name == action_name));
        if (!check_precondition(first_relevant_action, plan.kb, arguments))
            error(@sprintf("execute_action(): Action \"%s\" preconditions are not satisfied!", action_name));
        else
            execute_action(first_relevant_action, plan.kb, arguments);
        end
    end
    nothing;
end

