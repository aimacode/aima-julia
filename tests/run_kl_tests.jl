include("../aimajulia.jl");

using Test;

using Main.aimajulia;

using Main.aimajulia.utils;

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

@test (map(guess_example_value, restaurant, Base.Iterators.repeated(h)) == [true, false, true, true, false, true, false, true, false, false, false, true]);

animal_umbrellas = [Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", true)]),
                    Dict([("Species", "Cat"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "Yes"), ("GOAL", true)]),
                    Dict([("Species", "Dog"), ("Rain", "Yes"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Dog"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "No"), ("GOAL", false)]),
                    Dict([("Species", "Cat"), ("Rain", "No"), ("Coat", "Yes"), ("GOAL", true)])];

initial_h = [Dict([Pair("Species", "Cat")])];

h = current_best_learning(animal_umbrellas, initial_h);

@test (map(guess_example_value, animal_umbrellas, Base.Iterators.repeated(h)) == [true, true, true, false, false, false, true]);

party = [Dict([("Pizza", "Yes"), ("Soda", "No"), ("GOAL", true)]),
        Dict([("Pizza", "Yes"), ("Soda", "Yes"), ("GOAL", true)]),
        Dict([("Pizza", "No"), ("Soda", "No"), ("GOAL", false)])];

initial_h = [Dict([Pair("Pizza", "Yes")])];

h = current_best_learning(party, initial_h);

@test (map(guess_example_value, party, Base.Iterators.repeated(h)) == [true, true, false]);

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
            end), party, Base.Iterators.repeated(version_space)) == [true, true, false]);

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

# Initialize FOIL knowledge bases for extend_example(), choose_literal(), new_clause(),
# new_literals(), and foil().

test_network = FOILKnowledgeBase([expr("Conn(A, B)"),
                                expr("Conn(A ,D)"),
                                expr("Conn(B, C)"),
                                expr("Conn(D, C)"),
                                expr("Conn(D, E)"),
                                expr("Conn(E ,F)"),
                                expr("Conn(E, G)"),
                                expr("Conn(G, I)"),
                                expr("Conn(H, G)"),
                                expr("Conn(H, I)")]);

small_family = FOILKnowledgeBase([expr("Mother(Anne, Peter)"),
                                expr("Mother(Anne, Zara)"),
                                expr("Mother(Sarah, Beatrice)"),
                                expr("Mother(Sarah, Eugenie)"),
                                expr("Father(Mark, Peter)"),
                                expr("Father(Mark, Zara)"),
                                expr("Father(Andrew, Beatrice)"),
                                expr("Father(Andrew, Eugenie)"),
                                expr("Father(Philip, Anne)"),
                                expr("Father(Philip, Andrew)"),
                                expr("Mother(Elizabeth, Anne)"),
                                expr("Mother(Elizabeth, Andrew)"),
                                expr("Male(Philip)"),
                                expr("Male(Mark)"),
                                expr("Male(Andrew)"),
                                expr("Male(Peter)"),
                                expr("Female(Elizabeth)"),
                                expr("Female(Anne)"),
                                expr("Female(Sarah)"),
                                expr("Female(Zara)"),
                                expr("Female(Beatrice)"),
                                expr("Female(Eugenie)")]);

@test (extend_example(test_network, Dict([(expr("x"), expr("A")), (expr("y"), expr("B"))]), expr("Conn(x, z)"))
        == [Dict([(expr("x"), expr("A")),
                (expr("y"), expr("B")),
                (expr("z"), expr("B"))]),
            Dict([(expr("x"), expr("A")),
                (expr("y"), expr("B")),
                (expr("z"), expr("D"))])]);

@test (extend_example(test_network, Dict([(expr("x"), expr("G"))]), expr("Conn(x, y)"))
        == [Dict([(expr("x"), expr("G")),
                    (expr("y"), expr("I"))])]);


@test (extend_example(test_network, Dict([(expr("x"), expr("C"))]), expr("Conn(x, y)")) == []);

@test (length(extend_example(test_network, Dict(), expr("Conn(x, y)"))) == 10);

@test (length(extend_example(small_family, Dict([(expr("x"), expr("Andrew"))]), expr("Father(x, y)"))) == 2);

@test (length(extend_example(small_family, Dict([(expr("x"), expr("Andrew"))]), expr("Mother(x, y)"))) == 0);

@test (length(extend_example(small_family, Dict([(expr("x"), expr("Andrew"))]), expr("Female(y)"))) == 6);

# Initialize Tuple of literals and examples for choose_literal().

literals = map(expr, ("Conn(p, q)", "Conn(x, z)", "Conn(r, s)", "Conn(t, y)"));

examples_positive = [Dict([map(expr, ("x", "A")), map(expr, ("y", "B"))]),
                    Dict([map(expr, ("x", "A")), map(expr, ("y", "D"))])];

examples_negative = [Dict([map(expr, ("x", "A")), map(expr, ("y", "C"))]),
                    Dict([map(expr, ("x", "C")), map(expr, ("y", "A"))]),
                    Dict([map(expr, ("x", "C")), map(expr, ("y", "B"))]),
                    Dict([map(expr, ("x", "A")), map(expr, ("y", "I"))])];

@test (choose_literal(test_network, literals, (examples_positive, examples_negative)) == expr("Conn(x, z)"));

literals = map(expr, ("Conn(x, p)", "Conn(p, x)", "Conn(p, q)"));

examples_positive = [Dict([map(expr, ("x", "C"))]),
                    Dict([map(expr, ("x", "F"))]),
                    Dict([map(expr, ("x", "I"))])];

examples_negative = [Dict([map(expr, ("x", "D"))]),
                    Dict([map(expr, ("x", "A"))]),
                    Dict([map(expr, ("x", "B"))]),
                    Dict([map(expr, ("x", "G"))])];

@test (choose_literal(test_network, literals, (examples_positive, examples_negative)) == expr("Conn(p, x)"));

literals = map(expr, ("Father(x, y)", "Father(y, x)", "Mother(x, y)", "Mother(x, y)"));

examples_positive = [Dict([map(expr, ("x", "Philip"))]),
                    Dict([map(expr, ("x", "Mark"))]),
                    Dict([map(expr, ("x", "Peter"))])];

examples_negative = [Dict([map(expr, ("x", "Elizabeth"))]),
                    Dict([map(expr, ("x", "Sarah"))])];

@test (choose_literal(small_family, literals, (examples_positive, examples_negative)) == expr("Father(x, y)"));

literals = map(expr, ("Father(x, y)", "Father(y, x)", "Male(x)"));

examples_positive = [Dict([map(expr, ("x", "Philip"))]),
                    Dict([map(expr, ("x", "Mark"))]),
                    Dict([map(expr, ("x", "Andrew"))])];

examples_negative = [Dict([map(expr, ("x", "Elizabeth"))]),
                    Dict([map(expr, ("x", "Sarah"))])];

@test (choose_literal(small_family, literals, (examples_positive, examples_negative)) == expr("Male(x)"));

# Initialize target literal and examples for new_clause().

target = expr("Open(x, y)");

examples_positive = [Dict([map(expr, ("x", "B"))]),
                    Dict([map(expr, ("x", "A"))]),
                    Dict([map(expr, ("x", "G"))])];

examples_negative = [Dict([map(expr, ("x", "C"))]),
                    Dict([map(expr, ("x", "F"))]),
                    Dict([map(expr, ("x", "I"))])];

clause = new_clause(test_network, (examples_positive, examples_negative), target)[1][2];

@test ((length(clause) == 1)
        && (clause[1].operator == "Conn")
        && (clause[1].arguments[1] == expr("x")));

target = expr("Flow(x, y)");

examples_positive = [Dict([map(expr, ("x", "B"))]),
                    Dict([map(expr, ("x", "D"))]),
                    Dict([map(expr, ("x", "E"))]),
                    Dict([map(expr, ("x", "G"))])];

examples_negative = [Dict([map(expr, ("x", "A"))]),
                    Dict([map(expr, ("x", "C"))]),
                    Dict([map(expr, ("x", "F"))]),
                    Dict([map(expr, ("x", "I"))]),
                    Dict([map(expr, ("x", "H"))])];

clause = new_clause(test_network, (examples_positive, examples_negative), target)[1][2];

@test ((length(clause) == 2) &&
        (((clause[1].arguments[1] == expr("x")) && (clause[2].arguments[2] == expr("x")))
        || ((clause[1].arguments[2] == expr("x")) && (clause[2].arguments[1] == expr("x")))));

# Check length of returned Tuple for new_literals().

@test (length(new_literals(test_network, (expr("p | q"),  [expr("p")]))) == 8);

@test (length(new_literals(test_network, (expr("p"), [expr("q"), expr("p | r")]))) == 15);

@test (length(new_literals(small_family, (expr("p"), []))) == 8);

@test (length(new_literals(small_family, (expr("p & q"), []))) == 20);

# Initialize examples Tuple and target literal for foil().

target = expr("Parent(x, y)");

examples_positive = [Dict([map(expr, ("x", "Elizabeth")), map(expr, ("y", "Anne"))]),
                    Dict([map(expr, ("x", "Elizabeth")), map(expr, ("y", "Andrew"))]),
                    Dict([map(expr, ("x", "Philip")), map(expr, ("y", "Anne"))]),
                    Dict([map(expr, ("x", "Philip")), map(expr, ("y", "Andrew"))]),
                    Dict([map(expr, ("x", "Anne")), map(expr, ("y", "Peter"))]),
                    Dict([map(expr, ("x", "Anne")), map(expr, ("y", "Zara"))]),
                    Dict([map(expr, ("x", "Mark")), map(expr, ("y", "Peter"))]),
                    Dict([map(expr, ("x", "Mark")), map(expr, ("y", "Zara"))]),
                    Dict([map(expr, ("x", "Andrew")), map(expr, ("y", "Beatrice"))]),
                    Dict([map(expr, ("x", "Andrew")), map(expr, ("y", "Eugenie"))]),
                    Dict([map(expr, ("x", "Sarah")), map(expr, ("y", "Beatrice"))]),
                    Dict([map(expr, ("x", "Sarah")), map(expr, ("y", "Eugenie"))])];

examples_negative = [Dict([map(expr, ("x", "Anne")), map(expr, ("y", "Eugenie"))]),
                    Dict([map(expr, ("x", "Beatrice")), map(expr, ("y", "Eugenie"))]),
                    Dict([map(expr, ("x", "Mark")), map(expr, ("y", "Elizabeth"))]),
                    Dict([map(expr, ("x", "Beatrice")), map(expr, ("y", "Philip"))])];

clauses = foil(small_family, (examples_positive, examples_negative), target);

@test ((length(clauses) == 2) &&
        (((clauses[1][2][1] == expr("Father(x, y)")) && (clauses[2][2][1] == expr("Mother(x, y)")))
        || ((clauses[2][2][1] == expr("Father(x, y)")) && (clauses[1][2][1] == expr("Mother(x, y)")))));

