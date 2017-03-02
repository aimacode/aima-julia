include("utils.jl");

using utils;

using Base.Test;

# The following queue tests are from the aima-python utils.py doctest

na = [1, 8, 2, 7, 5, 6, -99, 99, 4, 3, 0];

function qtest(qf::DataType; order=false, f=Void)
	if (!(qf <: PQueue))
		q = qf();
		extend!(q, na);
		for num in na
			@test num in q;
		end
		@test !(42 in q);
		return [pop!(q) for i in range(0, length(q))];
	else
		if (order == false)
			q = qf();
		else
			q = qf(order=order);
		end
		if (f != Void)
			extend!(q, na, f);
		else
			extend!(q, na, (function(item) return item; end));
		end
		for num in na
			@test num in [getindex(x, 2) for x in collect(q)];
		end
		@test !(42 in [getindex(x, 2) for x in collect(q)]);
		return [getindex(pop!(q), 2) for i in range(0, length(q))];
	end
end

@test qtest(Stack) == [0, 3, 4, 99, -99, 6, 5, 7, 2, 8, 1];

@test qtest(FIFOQueue) == [1, 8, 2, 7, 5, 6, -99, 99, 4, 3, 0];

@test qtest(PQueue) == [-99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 99];

@test qtest(PQueue, order=Base.Order.Reverse) == [99, 8, 7, 6, 5, 4, 3, 2, 1, 0, -99];

@test qtest(PQueue, f=abs) == [0, 1, 2, 3, 4, 5, 6, 7, 8, -99, 99];

@test qtest(PQueue, order=Base.Order.Reverse, f=abs) == [99, -99, 8, 7, 6, 5, 4, 3, 2, 1, 0];