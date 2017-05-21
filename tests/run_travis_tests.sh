#!/bin/sh

set -e

echo "Changing directory to: $(dirname $0)"

cd $(dirname $0)

echo "TRAVIS_PULL_REQUEST: $TRAVIS_PULL_REQUEST"
echo "TRAVIS_PULL_REQUEST_SHA: $TRAVIS_PULL_REQUEST_SHA"

echo "$" "ulimit -a"

ulimit -a

echo

git clone https://github.com/aimacode/aima-data

echo

julia -e "versioninfo();"

echo

#Some of the testv() doctests in agents.py can sometimes fail when the 
#scores are out of expected bounds.

julia --color=yes run_agent_tests.jl

julia --color=yes run_search_tests.jl

julia --color=yes run_util_tests.jl

julia --color=yes run_game_tests.jl

julia --color=yes run_csp_tests.jl

julia --color=yes run_logic_tests.jl

julia --color=yes run_planning_tests.jl

