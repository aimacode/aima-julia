How to Contribute to aima-julia
===============================

Thanks for considering contributing to aima-julia!

Here is a guide (similar to the [aima-python contributing guide](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md)) on how you can help.

In general, the main ways you can contribute to the repository are the following:

1. Implement algorithms from the [list of algorithms](https://github.com/aimacode/aima-julia/blob/master/README.md).
2. Add tests for algorithms that are missing them (you can also add more tests to algorithms that already have some).
3. Take care of [issues](https://github.com/aimacode/aima-julia/issues).
4. Write on the notebooks (`.ipynb` files).
5. Add and edit documentation (the docstrings in `.jl` files).

In more detail:

## Read the Code and Start on an Issue

- First, read and understand the code to get a feel for the extent and the style (see Style Guide below).
- Look at the issues and pick one to work on.
- One of the issues is that some algorithms are missing from the list of algorithms and that some don't have tests.

## Writing and Running Tests

For aima-julia, Julia tests should avoid the `@testset` macro for greater modularity. For example, in Machine Learning, the trained neural network might fail during a `@test` for accuracy. The training should still be evaluated, so its corresponding test will be a custom Julia function that those not assert a threshold on accuracy.

## RandomDevice()

Avoid using `RandomDevice()`. Try using `aimajulia.RandomDeviceInstance` (can be referenced as `RandomDeviceInstance` within the `aimajulia` module) instead. Multiple `RandomDevice()` calls cause errors on some operating systems when opening multiple concurrent file descriptors to  `/dev/urandom` or `/dev/random`.

## Haskell-like type assertion, Haskell/Lisp Functional Programming

- When writing functions, the arguments should be type asserted (with exception to functions like getindex() for the Dict DataType).
- The use of `collect()`, `map()`, `reduce()`, `mapreduce()`, and anonymous functions are recommended.
- When declaring type definitions, try to assert the type of the fields.

## Porting to Julia from Python

- Use comprehensions when possible. In addition, use `Iterators` when dealing with Python generator expressions (collecting the items if required).
- String formatting can be accomplished with `sprintf()` and string concatenation (using the `*` operator or passing the `*` operator to `reduce()`).
- Division between 2 real numbers results in a float.
- Julia has native matrices, avoid using arrays of arrays unless required.
- Add more tests in `test_*.jl` files. Strive for terseness; it is ok to group multiple asserts into one function. Move most tests to `test_*.jl`, but it is fine to have a single doctest example in the docstring of a function in the `.jl` file, if the purpose of the doctest is to explain how to use the function, rather than test the implementation.

## New and Improved Algorithms

- Implement functions that were in the third edition of the book but were not yet implemented in the code. Check the [list of pseudocode algorithms (pdf)](https://github.com/aimacode/pseudocode/blob/master/aima3e-algorithms.pdf) to see what's missing.
- As we finish chapters for the new fourth edition, we will share the new pseudocode in the [`aima-pseudocode`](https://github.com/aimacode/aima-pseudocode) repository, and describe what changes are necessary. We hope to have an `algorithm-name.md` file for each algorithm, eventually; it would be great if contributors could add some for the existing algorithms.
- Give examples of how to use the code in the `.ipynb` files.

## Jupyter Notebooks

In this project we use Jupyter/IJulia Notebooks to showcase the algorithms in the book. They serve as short tutorials on what the algorithms do, how they are implemented and how one can use them. To install Jupyter, you can follow the instructions here. These are some ways you can contribute to the notebooks:

- Proofread the notebooks for grammar mistakes, typos, or general errors.
- Move visualization and unrelated to the algorithm code from notebooks to `notebook.jl` (a file used to store code for the notebooks, like visualization and other miscellaneous stuff). Make sure the notebooks still work and have their outputs showing!
- Replace the `%psource` magic notebook command with the function `psource` from `notebook.jl` where needed. Examples where this is useful are a) when we want to show code for algorithm implementation and b) when we have consecutive cells with the magic keyword (in this case, if the code is large, it's best to leave the output hidden).
- Add the function pseudocode(algorithm_name) in algorithm sections. The function prints the pseudocode of the algorithm. You can see some example usage in `knowledge.ipynb`.
- Edit existing sections for algorithms to add more information and/or examples.
- Add visualizations for algorithms. The visualization code should go in notebook.jl to keep things clean.
- Add new sections for algorithms not yet covered. The general format we use in the notebooks is the following: First start with an overview of the algorithm, printing the pseudocode and explaining how it works. Then, add some implementation details, including showing the code (using psource). Finally, add examples for the implementations, showing how the algorithms work. Don't fret with adding complex, real-world examples; the project is meant for educational purposes. You can of course choose another format if something better suits an algorithm.

Apart from the notebooks explaining how the algorithms work, we also have notebooks showcasing some indicative applications of the algorithms. These notebooks are in the `*_apps.ipynb` format. We aim to have an apps notebook for each module, so if you don't see one for the module you would like to contribute to, feel free to create it from scratch! In these notebooks we are looking for applications showing what the algorithms can do. The general format of these sections is this: Add a description of the problem you are trying to solve, then explain how you are going to solve it and finally provide your solution with examples. Note that any code you write should not require any external libraries apart from the ones already provided (like matplotlib).

# Style Guide

There are a few style rules that are unique to this project:

- The first rule is that the code should correspond directly to the pseudocode in the book. When possible this will be almost one-to-one, just allowing for the syntactic differences between Julia and pseudocode, and for different library functions.
- Don't make a function more complicated than the pseudocode in the book, even if the complication would add a nice feature, or give an efficiency gain. Instead, remain faithful to the pseudocode, and if you must, add a new function (not in the book) with the added feature.
- I use functional programming (functions with no side effects) in many cases, but not exclusively (sometimes type declarations and/or functions with side effects are used). Let the book's pseudocode be the guide.

Beyond the above rules, we use the official Julia Style Guide ([0.5](https://docs.julialang.org/en/release-0.5/manual/style-guide/)/[0.6](https://docs.julialang.org/en/release-0.5/manual/style-guide/)/[1.1](https://docs.julialang.org/en/v1/manual/style-guide/)), with a few minor exceptions:

- One line comments start with a space after the # sign.
- Use 4 spaces instead of tabs
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings. In most cases, a one-line docstring does not suffice. It is necessary to list what each argument does; the name of the argument usually is enough.
- Not all constants have to be uppercase.
- Parenthesize expressions consisting of multiple subexpressions to avoid confusion.

Updating existing code to newer Julia versions
==============================================

The Julia language frequently changes their latest stable version. For example, Julia 0.5 was announced October 11, 2016 and Julia 0.6 was announced June 27, 2017. As a result, we should have a separate branch for each supported Julia version. Pull requests should be made specifically to those branches (make an issue if the branch does not exist).

Contributing a Patch
====================

- Submit an issue describing your proposed change to the repo in question (or work on an existing issue).

- The repo owner will respond to your issue promptly.

- Fork the desired repo, develop and test your code changes.

- Submit a pull request.

Reporting Issues
================

- Under which versions of Julia does this happen?

- Provide an example of the issue occurring.

- Is anybody working on this?

Patch Rules
===========

- Ensure that the patch is Julia 1.1 compliant.

- Include tests if your patch is supposed to solve a bug, and explain clearly under which circumstances the bug happens. Make sure the test fails without your patch.

- Follow the style guidelines described above.

