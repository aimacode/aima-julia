#!/bin/sh

set -e

echo "TRAVIS_PULL_REQUEST: $TRAVIS_PULL_REQUEST"
echo "TRAVIS_PULL_REQUEST_SHA: $TRAVIS_PULL_REQUEST_SHA"

julia -e "versioninfo();"

git clone https://github.com/aimacode/aima-data

julia --color=yes -e "include(\"run_search_tests.jl\");"

rm -rf aima-data

julia --color=yes -e "include(\"run_util_tests.jl\");"


#Some of the testv() doctests can sometimes fail when the scores are
#out of expected bounds.

julia --color=yes -e "include(\"run_agent_tests.jl\");" 


