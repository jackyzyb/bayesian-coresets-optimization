#!/bin/bash

for ID in {1..10}
do
    for alg in "SVI"  "GIGAOE" "RAND" 'IHT' 'IHT-2' # "GIGAO"  "GIGAR" "GIGARE"
    do
	python3 main.py $alg $ID
    done
done
