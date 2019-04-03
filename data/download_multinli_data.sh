#!/bin/sh

curl -# https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip  > multinli.zip && \
unzip multinli.zip                                  && \
rm multinli.zip
echo "Downloaded data"
