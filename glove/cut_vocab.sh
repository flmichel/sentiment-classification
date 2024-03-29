#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:
cat ../data/vocab_full.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > data/vocab_cut.txt
