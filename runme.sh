#!/bin/sh

base=~/Repositories/benchmarks/
max_nreps=8

for s in 1 2 3; do
    for t in 0a 0b 0c; do
	    for r in 2 4 8; do
		    for g in 2; do
			    python script.py ${base}/set${t} ${max_nreps} ${r} ${g} ${s} &
		    done
		done
		wait
	done
#	wait
done

#python script.py ~/Repositories/benchmarks/set1 8 2 2 3 &
