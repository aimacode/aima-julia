
import Base: hash, ==, show,
            <, <=, >, >=,
            -, +, *, ^, /, \, %,
            ~, &, |, $, >>, <<;

export hash, ==, show,
        <, <=, >, >=,
        -, +, *, ^, /, \, %,
        ~, &, |, $, >>, <<,
        Expression, expr,
        variables, subexpressions;

immutable Expression
	operator::String
	arguments::Tuple
end

Expression(op::String, args::Vararg{Any}) = Expression(op, map(string, args));

function (e::Expression)(args::Vararg{Any})
    if ((length(e.arguments) == 0) && is_logic_symbol(e.operator))
        return Expression(e.operator, map(string, args));
    else
        error("ExpressionError: ", e, " is not a Symbol (Nullary Expression)!");
    end
end

# Hashing Expressions and n-Tuple of Expression(s).

hash(e::Expression, h::UInt) = ((hash(e.operator) $ hash(e.arguments)) $ h);

hash(t_e::Tuple{Vararg{Expression}}, h::UInt) = reduce($, vcat(map(hash, collect(t_e)), h));

<(e1::Expression, e2::Expression) = Expression("<", e1, e2);

<=(e1::Expression, e2::Expression) = Expression("<=", e1, e2);

>(e1::Expression, e2::Expression) = Expression(">", e1, e2);

>=(e1::Expression, e2::Expression) = Expression(">=", e1, e2);

-(e1::Expression) = Expression("-", e1);

+(e1::Expression) = Expression("+", e1);

-(e1::Expression, e2::Expression) = Expression("-", e1, e2);

+(e1::Expression, e2::Expression) = Expression("+", e1, e2);

*(e1::Expression, e2::Expression) = Expression("*", e1, e2);

^(e1::Expression, e2::Expression) = Expression("^", e1, e2);

/(e1::Expression, e2::Expression) = Expression("/", e1, e2);

\(e1::Expression, e2::Expression) = Expression("\\", e1, e2);

%(e1::Expression, e2::Expression) = Expression("<=>", e1, e2);

~(e::Expression) = Expression("~", e);

(&)(e1::Expression, e2::Expression) = Expression("&", e1, e2);

|(e1::Expression, e2::Expression) = Expression("|", e1, e2);

($)(e1::Expression, e2::Expression) = Expression("=/=", e1, e2);

>>(e1::Expression, e2::Expression) = Expression("==>", e1, e2);

<<(e1::Expression, e2::Expression) = Expression("<==", e1, e2);

==(e1::Expression, e2::Expression) = ((e1.operator == e2.operator) && (e1.arguments == e2.arguments));

function show(e::Expression)
    if (length(e.arguments) == 0)
        return e.operator;
    elseif (is_logic_symbol(e.operator))
        return @sprintf("%s(%s)", e.operator, join(map(show, map(Expression, e.arguments)), ", "));
    elseif (length(e.arguments) == 1)
        return @sprintf("%s(%s)", e.operator, show(Expression(e.arguments[1])));
    else
        return @sprintf("(%s)", join(map(show, map(Expression, map(string, e.arguments))), @sprintf(" %s ", e.operator)));
    end
end

function show(io::IO, e::Expression)
    print(io, show(e));
    nothing;
end

function is_logic_symbol(s::String)
    if (length(s) == 0)
        return false;
    else
        return isalpha(s);
    end
end

function is_logic_variable_symbol(s::String)
    return (is_logic_symbol(s) && islower(s[1]));
end

function is_logic_variable(e::Expression)
    return ((length(e.arguments) == 0) && islower(e.operator))
end

"""
    is_logic_proposition_symbol(s)

Return if the given 's' is an initial uppercase String that is not 'TRUE' or 'FALSE'.
"""
function is_logic_proposition_symbol(s::String)
    return (is_logic_symbol(s) && isupper(s[1]) && (s != "TRUE") && (s != "FALSE"));
end

function expr(s::String)
    for (op, new_op) in (("==>", ">>"), ("<==", "<<"), ("<=>", "%"), ("=/=", "\$"))
        s = replace(s, op, new_op);
    end
    return eval(parse(s));
end

function expr(e::Expression)
    return e;
end

function subexpressions(e::Expression)
    local answer::AbstractVector = [e];
    for arg in e.arguments
        answer = vcat(answer, subexpressions(expr(string(arg))));
    end
    return answer;
end

function subexpressions(e::Int)
    local answer::AbstractVector = [Expression(string(e))];
    return answer;
end


function variables(e::Expression)
    return Set(x for x in subexpressions(e) if is_logic_variable(x));
end



