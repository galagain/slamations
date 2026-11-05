#!/bin/bash


if [ $# -ne 1 ]; then
    echo "Usage: $0 <version> <num_iterations>"
    exit 1
fi

version=$1
num_itr=$2

# datasets=( "MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3")
datasets=("MH01" "MH02" "V101" "V102" "room1" "room2" "corridor1" "corridor2")

for i in $(seq 0 $(expr $num_itr - 1)); do
    for dataset in "${datasets[@]}"; do
        echo -e "[bash:] -> $dataset 0 0 0 0 pose_optimization=on $version.$i"
        ./run_script.sh $dataset 0 1
    done
done

# for dataset in "${datasets[@]}"; do
#     log_file="Results/gpu_usage/gpu_usage_${dataset}.csv"
#     echo "timestamp, gpu_utilization(%), memory_utilization(%)" > "$log_file"  # Add CSV header

#     echo -e "[bash:] -> $dataset"

#     # Start logging GPU utilization in the background
#     (while true; do
#         echo "$(date +%Y-%m-%d\ %H:%M:%S), $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader | tr -d ' %')" >> "$log_file"
#         sleep 1
#     done) &
#     gpu_logger_pid=$!  # Store the PID of the background process

#     # Run your script
#     ./run_script.sh "$dataset" 1

#     # Stop GPU logging once the script finishes
#     kill "$gpu_logger_pid"
# done