#!/bin/bash

sh call_experiments.sh -m lda_experiment -c 10 -s 10 -t 100 -r GCNPredAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 42 -t 100 -r GCNPredAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 35 -t 100 -r GCNPredAvg

sh call_experiments.sh -m lda_experiment -c 10 -s 10 -t 100 -r GCNAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 42 -t 100 -r GCNAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 35 -t 100 -r GCNAvg

sh call_experiments.sh -m lda_experiment -c 10 -s 10 -t 100 -r FedAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 42 -t 100 -r FedAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 35 -t 100 -r FedAvg

sh call_experiments.sh -m lda_experiment -c 10 -s 10 -t 100 -r GCNAngleAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 42 -t 100 -r GCNAngleAvg
sh call_experiments.sh -m lda_experiment -c 10 -s 35 -t 100 -r GCNAngleAvg