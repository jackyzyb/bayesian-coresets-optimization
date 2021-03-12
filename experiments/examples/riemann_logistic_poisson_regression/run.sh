#!/bin/bash

for ID in {1..20}
do
    for alg in "PRIOR" "IHT" "IHT-2" "SVI" "GIGAR" "RAND" # "IHT-stoc" "GIGAO"
    do
        for dnm in "synth_lr" "phishing" "ds1" "synth_poiss" "biketrips" "airportdelays" #"synth_lr_large" "phishing_old"
	do
		python3 main.py $dnm $alg $ID
	done
    done
done
