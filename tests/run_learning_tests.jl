include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following learning tests are from the aima-python doctests

@test (repr(euclidean_distance([1, 2], [3, 4])) == "2.8284271247461903");

@test (repr(euclidean_distance([1, 2, 3], [4, 5, 6])) == "5.196152422706632");

@test (repr(euclidean_distance([0, 0, 0], [0, 0, 0])) == "0.0");

@test (root_mean_square_error([2, 2], [2, 2]) == 0);

@test (root_mean_square_error([0, 0], [0, 1]) == sqrt(0.5));

@test (root_mean_square_error([1, 0], [0, 1]) == 1);

@test (root_mean_square_error([0, 0], [0, -1]) == sqrt(0.5));

@test (root_mean_square_error([0, 0.5], [0, -0.5]) == sqrt(0.5));

@test (manhattan_distance([2, 2], [2, 2]) == 0);

@test (manhattan_distance([0, 0], [0, 1]) == 1);

@test (manhattan_distance([1, 0], [0, 1]) == 2);

@test (manhattan_distance([0, 0], [0, -1]) == 1);

@test (manhattan_distance([0, 0.5], [0, -0.5]) == 1);

@test (mean_boolean_error([1, 1], [0, 0]) == 1)

@test (mean_boolean_error([0, 1], [1, 0]) == 1)

@test (mean_boolean_error([1, 1], [0, 1]) == 0.5)

@test (mean_boolean_error([0, 0], [0, 0]) == 0)

@test (mean_boolean_error([1, 1], [1, 1]) == 0)

@test (mean_error([2, 2], [2, 2]) == 0);

@test (mean_error([0, 0], [0, 1]) == 0.5);

@test (mean_error([1, 0], [0, 1]) ==  1);

@test (mean_error([0, 0], [0, -1]) ==  0.5);

@test (mean_error([0, 0.5], [0, -0.5]) == 0.5);

iris_dataset = DataSet(name="iris", examples="aima-data/iris.csv", exclude=[4]);

@test (iris_dataset.inputs == [1, 2, 3]);

