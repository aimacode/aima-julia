import Base: run;

using Compat

export Environment,
       Action,
          NoOpAction,
       Percept,
       AgentProgram,
          TableDrivenAgentProgram,
          ReflexVacuumAgentProgram,
          SimpleReflexAgentProgram,
          ModelBasedReflexAgentProgram

"""
An agent perceives an *environment* through sensors and acts with actuators.

Sensors provide agent the *percepts*, based on which the agent delivers
*actions*

Pg. 35, AIMA 3ed
"""
@compat abstract type Environment end

"""
*AgentProgram* is an internal representation of an agent function with an
concrete implementation. While *agent function* can be abstract *AgentProgram*
provides clear direction to the implementation.

Pg. 35, AIMA 3ed
"""
@compat abstract type AgentProgram end

"""
*Action* is an agent's response to the environment through actuators.

Although the representation of a string may suffice for more most sample
programs, an abstract type is introduced to emphacize the need for
providing a concrete type based on the environment or agent at hand.

In most problems we try to solve, the *Action* may be driven by the choice
of the *Environment*
"""
@compat abstract type Action end

"""
*NoOp* is a directive where the agent does not take any futher action.
"""
immutable NoOpActionType <: Action
  val::Symbol
  NoOpActionType()=new("NoOp")
end

const Action_NoOp = NoOpActionType()

"""
*Percept* is an input to the *Agent* from environment through sensors.

Although the representation of a Tuple may suffice for more most sample
programs, an abstract type is introduced to emphacize the need for
providing a concrete type based on the environment or agent at hand.

In most problems we try to solve, the *Percept* may be driven by the choice
of the *Environment*
"""
@compat abstract type Percept end

"""
Given a *Percept* returns an *Action* apt for the agent.

Depending on the agent program the function may respond with different *Action*
evaluation strategies.
"""

function execute{AP <: AgentProgram}(ap::AP, p::Percept)
    error(E_ABSTRACT)
end


"""
*TableDrivenAgentProgram* is a simple model of an agent program where all
percept sequences are well-known ahead in time and can be organized as a
mapping from percepts to action.

Look at the corresponding execute method for *Action* evaluation strategy.

The implementation must have the following methods:

1. append - percept to the list of percepts seen my the AgentProgram
2. lookup - the percepts in the tables of the AgentProgram

Fig 2.7 Pg. 47, AIMA 3ed
"""

@compat abstract type TableDrivenAgentProgram <: AgentProgram end

function execute(ap::TableDrivenAgentProgram, percept::Percept)
  append(ap.percepts, percept)
  action = lookup(ap.table, ap.percepts)
  return action
end

"""
*Rule* is an abstract representation of a framework that associates a *State*
condition to the appropriate action.

Definition of a condition can be implementation dependent.
"""

@compat abstract type Rule end

"""
*State* is an internal evaluated position of the Environment. In the context
of the problem the *Environment* can be one of the stated states. Any input or'
action may lead to change in *Environment* state.
"""
@compat abstract type State end

"""
*SimpleReflexAgentProgram* is a simple *Percept* to *Action* matching state
based rules.

It does not depend on the historical percept data.

It needs to implement two methods

1. interpret_input
2. rule_match

for all the concrete implementations.

3. rules - Will provide all the rules associated with the
AgentProgram.

Fig 2.10 Pg. 49, AIMA 3ed
"""
@compat abstract type SimpleReflexAgentProgram <: AgentProgram end

function execute(ap::SimpleReflexAgentProgram, percept::Percept)
    state = interpret_input(percept);
    rule = rule_match(state, ap.rules);
    action = rule.action;
    return action;
end

"""
Given a *State* to provide an *Action* that the agent must execute.

Matching is useful for both:

1. *SimpleReflexAgentProgram*
2. *ModelBasedReflexAgentProgram*

Both *AgentPrograms* have state models in-built, hence the rule matches the
relevant *Action* to be picked up.
"""
function rule_match(state::State, rules::Vector{Rule})
    error(E_ABSTRACT)
end

"""
*ModelBasedReflexAgentProgram* uses a model which is close to the
understanding of the world.

The *AgentProgram* updates the states based on the *Percepts* received.

"""
@compat abstract type ModelBasedReflexAgentProgram <: AgentProgram end

function execute(ap::ModelBasedReflexAgentProgram, percept::Percept)
    ap.state = update_state(ap.state, ap.action, percept, ap.model);
    rule = rule_match(state, ap.rules);
    action = rule.action
    return action
end

"""

"""
type Agent{T <: AgentProgram}
  ap::T
end

#=
Concrete instantiation of VacuumEnvironment using the models described above.
=#

"""
*VacuumEnvironment* has 2 locations:

1. loc_A
2. loc_B

adjacent to each other. loc_B to the *Right* of loc_A. This shall mean loc_A is
on the *Left* of loc_B.

A vacuum cleaner can sense *Dirt* is the location it's in.

If *Dirt* is found it will *Suck* the *Dirt* and *Clean* the location.

The *Environment* is still abstract as it gets expressed through its components.
"""
@compat abstract type VacuumEnvironment <: Environment end

"""
In a *VacuumEnvironment* a robot can only read where it's current location is
and whether the location has *Dirt* or is *Clean*.

There are 4 possible *Percept*.

(loc_A, Dirty)
(loc_A, Clean)
(loc_B, Dirty)
(loc_B, Clean)
"""
type VacuumPercept <: Percept
  location_status::Tuple{Symbol, Symbol}
  VacuumPercept(loc::AbstractString, cstate::AbstractString)=
    new(Tuple(Symbol(loc), Symbol(cstate)))
end

"""
The vacuum cleaner can do the following actions.

Move from loc_A to loc_B --> Right
Move from loc_B to loc_A --> Left
If Dirty, Suck the Dirt  --> Suck
"""
type VacuumAction <: Action
  sym::Symbol
  VacuumAction(str::AbstractString)=new(Symbol(str))
end

const Action_Left = VacuumAction("Left")
const Action_Right = VacuumAction("Right")
const Action_Suck = VacuumAction("Suck")

"""
Concrete implementation for the Vacuum Agent using the
*TableDrivenAgentProgram*
"""
type TableDrivenVacuumAgentProgram <: TableDrivenAgentProgram
    table::Dict{Vector{VacuumPercept}, Action}
    percepts::Vector{VacuumPercept}

    function TableDrivenVacuumAgentProgram()
      PAC = VacuumPercept("loc_A", "Clean")
      PAD = VacuumPercept("loc_A", "Dirty")
      PBC = VacuumPercept("loc_B", "Clean")
      PBD = VacuumPercept("loc_B", "Dirty")
      table =   Dict([PAC] => Action_Right,
                     [PAD] => Action_Suck,
                     [PBC] => Action_Left,
                     [PBD] => Action_Suck,
                     [PAC, PAC] => Action_Right,
                     [PAC, PAD] => Action_Suck,
                     [PAC, PAC, PAC] => Action_Right,
                     [PAC, PAC, PAD] => Action_Suck)
      return new(table, Vector{Percept}())
    end
end

function append(percepts::Vector{VacuumPercept},
                percept::VacuumPercept)
  push!(percepts, percept)
end

function lookup(table::Dict{Vector{VacuumPercept}, Action},
                percepts::Vector{VacuumPercept})
    if (haskey(table, ap.percept_sequence))
        action = ap.table[ap.percept_sequence]
        @printf("%s perceives %s and does %s\n",
                string(typeof(ap)), string(percept), action.name);
    else
        @printf("%s perceives %s but cannot execute as table does not have the percept sequence.\n",
              string(typeof(ap)), string(percept));
    end
    return action
end

"""
Technically the data in *Percept* is not very different from *State* as
the *SimpleReflexAgentProgram* contains no knowledge of overall model nor has
information of historical states.
"""
type ReflexVacuumState <: State
  location_status::Tuple{Symbol, Symbol}
  ReflexVacuumState(loc::AbstractString, cstate::AbstractString)=
    new((Symbol(loc), Symbol(cstate)))
end

const State_A_Clean=ReflexVacuumState("loc_A","Clean")
const State_B_Clean=ReflexVacuumState("loc_B","Clean")
const State_A_Dirty=ReflexVacuumState("loc_A","Dirty")
const State_B_Dirty=ReflexVacuumState("loc_B","Dirty")

"""
A method needed by the SimpleReflexAgentProgram abstraction to map

*Percept* to an internal *State* of the *AgentProgram*
"""
#=
In this case as the State datastructure is very similar to Persept mere
reinterpretatation carried out in reality there may be additional
transformations or data repurposing may be needed.
=#
function interpret_input(percept::VacuumPercept)
    return reinterpret(ReflexVacuumState, percept)
end

type MappingRule <: Rule
  state::State
  action::Action
end

const Rule_A_Clean=MappingRule(State_A_Clean,Action_Right)
const Rule_B_Clean=MappingRule(State_B_Clean,Action_Left)
const Rule_A_Dirty=MappingRule(State_A_Dirty,Action_Suck)
const Rule_B_Dirty=MappingRule(State_B_Dirty,Action_Suck)

type SimpleReflexVacuumAgentProgram <: SimpleReflexAgentProgram
  rules::Vector{MappingRule}
  SimpleReflexVacuumAgentProgram()=new([State_A_Clean, State_B_Clean, State_A_Dirty, State_B_Dirty])
end

function rule_match(state, rules)
  for rule in rules
    if (state == rule.state_val)
      return rule.action
    end
  end
end

"""
*ReflexVacuumAgentProgram* is a simple *Percept* to *Action* matching model
only catering to the vacuum robot environment.

It does not depend on the historical percept data.

Fig 2.8 Pg. 48, AIMA 3ed
"""

@compat abstract type ReflexVacuumAgentProgram <: AgentProgram end

function execute(ap::ReflexVacuumAgentProgram, percept::VacuumPercept)
    location = percept.location_status[1]
    status = percept.location_status[2]
    action = (status == Symbol("Dirty"))? Action_Suck:
             (location == loc_A)? Action_Right:
             (location == loc_B)? Action_Left : nothing
    return action
end

"""
Model is a theoretical representation of the system or world. Sensors of the
Agent are the eyes and ears of the system to update the model.

The model has its internal states which will vary for *AgentProgram*.

In the *VacuumEnvironment* we choose the following as a Model.

**Model**

Model|Loc_A|Loc_B|
=====|=====|=====|
Agent|  1  |  0  |
=====|=====|=====|
Dirty|  1  |  1  |
=====|=====|=====|

Existence of the agent or dirt is shown as    : 1
Non-Existence of the agent or dirt is shown as: 0
When status is unknown it's kept as           :-1

Hence, when the model is initialized it will be a 2x2 grid as below:

Model|Loc_A|Loc_B|
=====|=====|=====|
Agent| -1  |  -1 |
=====|=====|=====|
Dirty| -1  |  -1 |
=====|=====|=====|

Hence, effectively there are 3 states per slot leading to 3^4=81 states.

However, as  you can see some states are impossible for example we know there is
only one agent. Hence, when location of the agent is known it's fairly
deterministic.

Agent state can be: (1,0) for loc_A or (0,1) for loc_B.
Indeterminate state (-1,-1) can be only for the first time but never
subsequently.

For example, the following states are impossible in the model.
First 2 elements are agent states and second 2 are the dirt state.

-1, 0, 0, 0   <-I1. Agent state in one location tells the other location state.
-1, 1, 0, 0   <-I2. Same as S1
-1,-1,-1, 0   <-I3. Dirt state known means agent state cannot be unknown.
-1,-1,-1, 1   <-I4. Same as S3.
-1,-1, 0, 0   <-I5. Same as S3
-1,-1, 0, 1   <-I6. Same as S3
-1,-1, 1, 0   <-I7. Same as S3
-1,-1, 1, 1   <-I8. Same as S3
...

Effectively, the model has only following valid environment states.

-1,-1,-1,-1   <-R0. Initial state Agent has not received any Percept.
                    Ignore this state and move read next Percept
 1, 0, 1,-1   <-R1. Agent goes to loc_A first. sees dirt.    -> Suck
 1, 0, 0,-1   <-R2. Agent goes to loc_A first. sees no dirt. -> Right
 0, 1,-1, 1   <-R3. Agent goes to loc_B first. sees dirt.    -> Suck
 0, 1,-1, 0   <-R4. Agent goes to loc_B first. sees no dirt. -> Left
 1, 0, 1, 0   <-R5. Agent goes to loc_A second.sees dirt.    -> Suck
 1, 0, 0, 0   <-R6. Agent goes to loc_A second.sees no dirt. -> NoOp
 0, 1, 0, 1   <-R7. Agent goes to loc_B second.sees dirt.    -> Suck
 0, 1, 0, 0   <-R8. Agent goes to loc_B second.sees no dirt. -> NoOp

 When Agent goes second time the first location has to be clean and cannot be
 unknown or dirty.

Due to the presence of the model the previous states are captured and
decisions can be taken on NoOp. One can also see an inherent assumption in the
model, dirt does not get generated in the locations autonomously nor added by
someone external. Those will fail the model.

Note: States here are very different from that of the
*SimpleReflexVacuumAgentProgram*
"""

type ModelVacuumState <: State
  val::Tuple{Int,Int,Int,Int}
  ModelVacuumState(v::Tuple{Int,Int,Int,Int})=new(v)
end

ModelVacuumState(v::Vector{Int})=ModelVacuumState(tuple(v...))

const R1=MappingRule(ModelVacuumState([1, 0, 1,-1]), Action_Suck)
const R2=MappingRule(ModelVacuumState([1, 0, 0,-1]), Action_Right)
const R3=MappingRule(ModelVacuumState([0, 1,-1, 1]), Action_Suck)
const R4=MappingRule(ModelVacuumState([0, 1,-1, 0]), Action_Left)
const R5=MappingRule(ModelVacuumState([1, 0, 1, 0]), Action_Suck)
const R6=MappingRule(ModelVacuumState([1, 0, 0, 0]), Action_NoOp)
const R7=MappingRule(ModelVacuumState([0, 1, 0, 1]), Action_Suck)
const R8=MappingRule(ModelVacuumState([0, 1, 0, 0]), Action_NoOp)

type ModelBasedVacuumAgentProgram <: ModelBasedReflexAgentProgram
  model::Vector{Int}
  rules::Vector{Rule}
  state::ModelVacuumState
  action::Action

  function ModelBasedVacuumAgentProgram()
    model = [-1,-1,-1,-1]
    rules = [R1, R2, R3, R4, R5, R6, R7, R8]
    state = ModelVacuumState([-1,-1,-1,-1])
    action = Action_NoOp
    return new(model, rules, state, action)
  end
end

function update_state(state, action, percept, model)

  #Update model with previous state
  for i=1:4
    model[i] = state.val[i]
  end

  loc = percept[1]
  status = percept[2]

  if (loc == Symbol("loc_A"))
    model[1] = 1; model[2] = 0
    model[3] = (status == Symbol("Dirty"))?1:0
  else
    model[1] = 0; model[2] = 1
    model[4] = (status == Symbol("Dirty"))?1:0
  end

  return VacuumState(model)
end
