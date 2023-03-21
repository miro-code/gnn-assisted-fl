#!/bin/bash

# Parse the command line arguments
while [ $# -gt 0 ]
do
    key="$1"

    case $key in
        -m|--module)
        MODULE="$2"
        shift # past argument
        shift # past value
        ;;
        -c|--num_clients_per_round)
        NUM_CLIENTS="$2"
        shift # past argument
        shift # past value
        ;;
        -s|--seed)
        SEED="$2"
        shift # past argument
        shift # past value
        ;;
        -t|--num_total_clients)
        NUM_TOTAL="$2"
        shift # past argument
        shift # past value
        ;;
        -r|--strategy)
        STRATEGY="$2"
        shift # past argument
        shift # past value
        ;;
        *) # unknown option
        shift # past argument
        ;;
    esac
done


# Call the function using Python with the parsed arguments
python -c "from gflower.examples.${MODULE} import run_fixed_fl; run_fixed_fl(num_clients_per_round=${NUM_CLIENTS}, seed=${SEED}, num_total_clients=${NUM_TOTAL}, strategy='${STRATEGY}')"
