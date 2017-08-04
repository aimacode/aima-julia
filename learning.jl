
export euclidean_distance, mean_square_error, root_mean_square_error,
        mean_error, manhattan_distance, mean_boolean_error,
        hamming_distance;

function euclidean_distance(X::AbstractVector, Y::AbstractVector)
    return sqrt(sum(((x - y)^2) for (x, y) in zip(X, Y)));
end

function mean_square_error(X::AbstractVector, Y::AbstractVector)
    return mean(((x - y)^2) for (x, y) in zip(X, Y));
end

function root_mean_square_error(X::AbstractVector, Y::AbstractVector)
    return sqrt(mean_square_error(X, Y));
end

function mean_error(X::AbstractVector, Y::AbstractVector)
    return mean(abs(x - y) for (x, y) in zip(X, Y));
end

function manhattan_distance(X::AbstractVector, Y::AbstractVector)
    return sum(abs(x - y) for (x, y) in zip(X, Y));
end

function mean_boolean_error(X::AbstractVector, Y::AbstractVector)
    return mean((x != y) for (x, y) in zip(X, Y));
end

function hamming_distance(X::AbstractVector, Y::AbstractVector)
    return sum((x != y) for (x, y) in zip(X, Y));
end

