#!/bin/sh

set -e

echo "TRAVIS_PULL_REQUEST: $TRAVIS_PULL_REQUEST"
echo "TRAVIS_PULL_REQUEST_SHA: $TRAVIS_PULL_REQUEST_SHA"

git clone https://github.com/aimacode/aima-data

julia -e "versioninfo();"

#Some of the testv() doctests in agents.py can sometimes fail when the 
#scores are out of expected bounds.

julia --color=yes run_agent_tests.jl

julia --color=yes run_search_tests.jl

julia --color=yes run_util_tests.jl

