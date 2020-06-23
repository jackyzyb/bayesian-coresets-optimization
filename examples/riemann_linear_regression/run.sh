#!/bin/bash

for ID in {1..10}
do
    for alg in "IHT" "SVI" "IHT-2" "GIGAOE" "RAND" #"GIGAO" "GIGAR" "GIGARE"
    do
	python3 main.py $alg $ID
    done
done
