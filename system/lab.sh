#!/bin/bash

TEMP="./temp/"
INPUT=$1
./splitter $INPUT $TEMP
./train.py $TEMP
