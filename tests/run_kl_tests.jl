include("../aimajulia.jl");

using Base.Test;

using aimajulia;

using aimajulia.utils;

#The following learning with knowledge tests are from the aima-python doctests

restaurant_attribute_names = ("Alternate", "Bar", "Fri/Sat", "Hungry", "Patrons", "Price", "Rain", "Reservation", "Type", "WaitEstimate", "GOAL")

restaurant = [Dict(collect(zip(restaurant_attribute_names, ("Yes", "No", "No", "Yes", "Some", "\$\$\$", "No", "Yes", "French", "0-10", true)))),
			Dict(collect(zip(restaurant_attribute_names, ("Yes", "No", "No", "Yes", "Full", "\$", "No", "No", "Thai", "30-60", false)))),
			Dict(collect(zip(restaurant_attribute_names, ("No", "Yes", "No", "No", "Some", "\$", "No", "No", "Burger", "0-10", true)))),
			Dict(collect(zip(restaurant_attribute_names, ("Yes", "No", "Yes", "Yes", "Full", "\$", "Yes", "No", "Thai", "10-30", true)))),
			Dict(collect(zip(restaurant_attribute_names, ("Yes", "No", "Yes", "No",  "Full", "\$\$\$", "No", "Yes", "French", ">60", false)))),
			Dict(collect(zip(restaurant_attribute_names, ("No", "Yes", "No",  "Yes", "Some", "\$\$", "Yes", "Yes", "Italian", "0-10", true)))),
			Dict(collect(zip(restaurant_attribute_names, ("No", "Yes", "No",  "No",  "None", "\$", "Yes", "No", "Burger", "0-10", false)))),
			Dict(collect(zip(restaurant_attribute_names, ("No", "No", "No",  "Yes", "Some", "\$\$", "Yes", "Yes", "Thai", "0-10", true)))),
			Dict(collect(zip(restaurant_attribute_names, ("No", "Yes", "Yes", "No",  "Full", "\$", "Yes", "No", "Burger", ">60", false)))),
			Dict(collect(zip(restaurant_attribute_names, ("Yes", "Yes", "Yes", "Yes", "Full", "\$\$\$", "No", "Yes", "Italian", "10-30",false)))),
			Dict(collect(zip(restaurant_attribute_names, ("No",  "No", "No",  "No",  "None", "\$", "No", "No", "Thai", "0-10", false)))),
			Dict(collect(zip(restaurant_attribute_names, ("Yes", "Yes", "Yes", "Yes", "Full", "\$", "No", "No", "Burger", "30-60", true))))];

initial_h = [Dict([Pair("Alternate", "Yes")])];

h = current_best_learning(restaurant, initial_h);

@test (map(guess_example_value, restaurant, repeated(h)) == [true, false, true, true, false, true, false, true, false, false, false, true]);

animal_umbrellas = [Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", true)]),
                    Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Dog"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "Yes"), ("GOAL", true)])];

initial_h = [Dict([Pair("Species", "Cat")])];

h = current_best_learning(animal_umbrellas, initial_h);

@test (map(guess_example_value, animal_umbrellas, repeated(h)) == [true, true, true, false, false, false, true]);

party = [Dict([("Pizza", "Yes"), ("Soda", "No"), ("GOAL", true)]),
        Dict([("Pizza", "Yes"), ("Soda", "Yes"), ("GOAL", true)]),
        Dict([("Pizza", "No"), ("Soda", "No"), ("GOAL", false)])];

initial_h = [Dict([Pair("Pizza", "Yes")])];

h = current_best_learning(party, initial_h);

@test (map(guess_example_value, party, repeated(h)) == [true, true, false]);

party = [Dict([("Pizza", "Yes"), ("Soda", "No"), ("GOAL", true)]),
        Dict([("Pizza", "Yes"), ("Soda", "Yes"), ("GOAL", true)]),
        Dict([("Pizza", "No"), ("Soda", "No"), ("GOAL", false)])];

version_space = version_space_learning(party);

@test (map((function(e::Dict, V::AbstractVector)
                for h in V
                    if (guess_example_value(e, h))
                        return true;
                    end
                end
                return false;
            end), party, repeated(version_space)) == [true, true, false]);

@test ([Dict([Pair("Pizza", "Yes")])] in version_space);

party = [Dict([("Pizza", "Yes"), ("Soda", "No"), ("GOAL", true)]),
        Dict([("Pizza", "Yes"), ("Soda", "Yes"), ("GOAL", true)]),
        Dict([("Pizza", "No"), ("Soda", "No"), ("GOAL", false)])];

animal_umbrellas = [Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", true)]),
                    Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Dog"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "Yes"), ("GOAL", true)])];

conductance_attribute_names = ("Sample", "Mass", "Temperature", "Material", "Size", "GOAL");
conductance = [Dict(collect(zip(conductance_attribute_names, ("S1", 12, 26, "Cu", 3, 0.59)))),
                Dict(collect(zip(conductance_attribute_names, ("S1", 12, 100, "Cu", 3, 0.57)))),
                Dict(collect(zip(conductance_attribute_names, ("S2", 24, 26, "Cu", 6, 0.59)))),
                Dict(collect(zip(conductance_attribute_names, ("S3", 12, 26, "Pb", 2, 0.05)))),
                Dict(collect(zip(conductance_attribute_names, ("S3", 12, 100, "Pb", 2, 0.04)))),
                Dict(collect(zip(conductance_attribute_names, ("S4", 18, 100, "Pb", 3, 0.04)))),
                Dict(collect(zip(conductance_attribute_names, ("S4", 18, 100, "Pb", 3, 0.04)))),
                Dict(collect(zip(conductance_attribute_names, ("S5", 24, 100, "Pb", 4, 0.04)))),
                Dict(collect(zip(conductance_attribute_names, ("S6", 36, 26, "Pb", 6, 0.05))))];

@test (minimal_consistent_determination(party, Set(["Pizza", "Soda"])) == Set(["Pizza"]));

@test (minimal_consistent_determination(party[1:2], Set(["Pizza", "Soda"])) == Set());

@test (minimal_consistent_determination(animal_umbrellas, Set(["Species", "Rain", "Coat"])) == Set(["Species", "Rain", "Coat"]));

@test (minimal_consistent_determination(conductance, Set(["Mass", "Temperature", "Material", "Size"])) == Set(["Temperature", "Material"]));

@test (minimal_consistent_determination(conductance, Set(["Mass", "Temperature", "Size"])) == Set(["Mass", "Temperature", "Size"]));

