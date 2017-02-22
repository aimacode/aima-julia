typealias Percept Tuple{Any, Any}

abstract Action;		#declare Action as a supertype for Action implementations

type AAction <: Action
	isNoOp::Bool
	name::String

	function AAction(n::String)
		return new(Bool(false), n);
	end
end

type NoOpAction <: Action
	isNoOp::Bool
	name::String

	function NoOpAction()
		return new(Bool(true), "NoOp");
	end
end

abstract AgentProgram;		#declare AgentProgram as a supertype for AgentProgram implementations

#=

	Define a global execute() function to be implemented for each respective
	AgentProgram DataType implementation.
		
=#

function execute{T <: AgentProgram}(ap::T, p::Percept)		#implement functionality later
	println("execute() not yet implemented for ", typeof(ap), "!");
	nothing;
end

type TableDrivenAgentProgram <: AgentProgram
	isTracing::Bool
	percepts::Array{Percept, 1}
	table::Dict{Any, Any}

	function TableDrivenAgentProgram(;table_dict=C_NULL, trace=false)
		if (table_dict == C_NULL)			#no table given, create empty dictionary
			return new(Bool(trace), Array{Percept, 1}(), Dict{Any, Any}());
		else
			return new(Bool(trace), Array{Percept, 1}(), table_dict);
		end
	end
end

function execute(ap::TableDrivenAgentProgram, p::Percept)
	ap.percepts.push(p);
	action = ap.table[Tuple((ap.percepts...))]	#convert percept sequence to tuple
	if (ap.isTracing)
		@printf("%s perceives %s and does %s", string(typeof(ap)), string(p), action.name);
	end
	return action;
end

type ReflexVacuumAgentProgram <: AgentProgram
	isTracing::Bool

	function ReflexVacuumAgentProgram(;trace=false)
		return new(Bool(trace));
	end
end

type ModelBasedVacuumAgentProgram <: AgentProgram
	isTracing::Bool
	model::Dict{Any, Any}

	function ModelBasedVacuumAgentProgram(;trace=false, model=Void)
		if (typeof(model) <: Dict{Any, Any})
			new_ap = new(Bool(trace));
			new_ap.model = deepcopy(model);
			return new_ap;
		else
			return new(Bool(trace), Dict{Any, Any}());
		end
	end
end

#=

	Agents can interact with the environment through percepts and actions.

=#

abstract EnvironmentObject;			#declare EnvironmentObject as a supertype for EnvironmentObject implementations

type Agent <: EnvironmentObject		#the Agent exist in the environment like other environment objects such as gold
	alive::Bool
	performance::Float64
	program::AgentProgram
	location::Tuple{Any, Any}		#initialized when adding agent to environment

	function Agent()
		return new(Bool(true), Float64(0));
	end

	function Agent{T <: AgentProgram}(ap::T)
		new_agent = new(Bool(true), Float64(0));	#program is undefined
		new_agent.program = ap;
		return new_agent;
	end
end


function isAlive{T <: Agent}(a::T)
	return a.alive;
end

function setAlive{T <: Agent}(a::T, bv::Bool)
	a.alive = bv;
	nothing;
end

#=

	Implement execute() methods for implemented AgentPrograms.

=#

loc_A = (0, 0)
loc_B = (1, 0)

function execute(ap::ReflexVacuumAgentProgram, location_status::Percept)
	local location = location_status[1];
	local status = location_status[2];
	if (status == "Dirty")
		return "Suck";
	elseif (location == loc_A)
		return "Right";
	elseif (location == loc_B)
		return "Left";
	end
end

function execute(ap::ModelBasedVacuumAgentProgram, location_status::Percept)
	local location = location_status[1];
	local status = location_status[2];
	ap.model[location] = status;							#update existing model
	if (ap.model[loc_A] == ap.model[loc_B] == "Clean")		#return "NoOp" when no work is necessary
		return "NoOp";
	elseif (status == "Dirty")
		return "Suck";
	elseif (location == loc_A)
		return "Right";
	elseif (location == loc_B)
		return "Left";
	end
end

#=

	Load a implemented AgentProgram into a Agent.

=#

function TableDrivenVacuumAgent()
	#dictionary representation of table (Fig. 2.3) of percept sequences (key) mappings to actions (value).
	local table = Dict{Any, Any}([
				Pair(((loc_A, "Clean"),), "Right"),
				Pair(((loc_A, "Dirty"),), "Suck"),
				Pair(((loc_B, "Clean"),), "Left"),
				Pair(((loc_B, "Dirty"),), "Suck"),
				Pair(((loc_A, "Clean"), (loc_A, "Clean")), "Right"),
				Pair(((loc_A, "Clean"), (loc_A, "Dirty")), "Suck"),
				# ...
				Pair(((loc_A, "Clean"), (loc_A, "Clean"), (loc_A, "Clean")), "Right"),
				Pair(((loc_A, "Clean"), (loc_A, "Clean"), (loc_A, "Dirty")), "Suck"),
				# ...
				]);
    return Agent(TableDrivenAgentProgram(table_dict=table));
end

function ReflexVacuumAgent()
    #Return a reflex agent for the two-state vacuum environment (Fig. 2.8).
    return Agent(ReflexVacuumAgentProgram());
end

function ModelBasedVacuumAgent()
	#Return a agent that tracks statuses of clean and dirty locations.
	return Agent(ModelBasedVacuumAgentProgram(model=Dict{Any, Any}([
			Pair(loc_A, Void),
			Pair(loc_B, Void),
			])));
end

abstract Environment;				#declare Environment as a supertype for Environment implementations

type XYEnvironment <: Environment
	objects::Array{EnvironmentObject, 1}
	agents::Array{Agent, 1}					#agents found in this field should also be found in the objects field

	function XYEnvironment()
		return new(Array{EnvironmentObject, 1}(), Array{Agent, 1}());
	end
end

function percept{T1 <: Environment, T2 <: Action}(e::T1, a::Agent, act::T2)		#implement this later
	println("percept() not yet implemented for ", typeof(e), "!");
	nothing;
end
