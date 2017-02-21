type Percept
	attr::Dict{Any, Any}

	function Percept()
		return new(Dict{Any, Any}());
	end
end

abstract Action;		#declare Action as a supertype

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

abstract AgentProgram;

#=

	Define a global execute() function to be implemented for each respective
	AgentProgram DataType implementation.
		
=#

execute{T <: AgentProgram}(ap::T, p::Percept) = function() end		#implement functionality later

type TableDrivenAgentProgram <: AgentProgram
	isTracing::Bool
	percepts::Array{Percept, 1}
	table::Dict{Any, Any}

	function TableDrivenAgentProgram(;t=C_NULL, trace=false)
		if (t == C_NULL)			#no table given, create empty dictionary
			return new(Bool(trace), Array{Percept, 1}(), Dict{Any, Any}());
		else
			return new(Bool(trace), Array{Percept, 1}(), t);
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

#=

	Agents can interact with the environment through percepts and actions.

=#

type Agent
	alive::Bool
	program::AgentProgram

	function Agent()
		return new(Bool(true))
	end
end


function isAlive{T <: Agent}(a::T)
	return a.alive;
end

function setAlive{T <: Agent}(a::T, bv::Bool)
	a.alive = bv;
	nothing;
end
