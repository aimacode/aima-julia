# aima-julia

[![Build Status](https://travis-ci.org/aimacode/aima-julia.svg?branch=master)](https://travis-ci.org/aimacode/aima-julia)

Julia (v0.5+) implementation of the algorithms found in "Artificial Intelligence: A Modern Approach".

Using aima-julia for portable purposes
--------------------------------------

Include the following lines in all files within the same directory.

~~~
include("aimajulia.jl");
using aimajulia;
~~~

Running tests
-------------

All Base.Test tests for the aima-julia project can be found in the [tests](https://github.com/aimacode/aima-julia/tree/master/tests) directory.

Conventions
-----------

* 4 spaces, not tabs

* Please try to follow the style conventions of the file your are modifying.

We like this [style guide](https://docs.julialang.org/en/release-0.5/manual/style-guide/).

## Acknowledgements

The algorithms implemented in this project are found from both Russell And Norvig's "Artificial Intelligence - A Modern Approach" and [aima-pseudocode](https://github.com/aimacode/aima-pseudocode).