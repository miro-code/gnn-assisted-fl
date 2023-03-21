#!/bin/bash

sh call_experiments.sh -m single_experiment -c 10 -s 10 -t 3229 -r GCNPredAvg
sh call_experiments.sh -m single_experiment -c 10 -s 42 -t 3229 -r GCNPredAvg
sh call_experiments.sh -m single_experiment -c 10 -s 35 -t 3229 -r GCNPredAvg

sh call_experiments.sh -m single_experiment -c 10 -s 10 -t 3229 -r GCNAvg
sh call_experiments.sh -m single_experiment -c 10 -s 42 -t 3229 -r GCNAvg
sh call_experiments.sh -m single_experiment -c 10 -s 35 -t 3229 -r GCNAvg

sh call_experiments.sh -m single_experiment -c 10 -s 10 -t 3229 -r FedAvg
sh call_experiments.sh -m single_experiment -c 10 -s 42 -t 3229 -r FedAvg
sh call_experiments.sh -m single_experiment -c 10 -s 35 -t 3229 -r FedAvg
