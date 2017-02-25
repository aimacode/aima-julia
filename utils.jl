module utils

export if_;

function if_(boolean_expression, ans1, ans2)
	if (boolean_expression)
		return ans1;
	else
		return ans2;
	end
end

end