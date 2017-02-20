type Agent
	alive::Bool

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