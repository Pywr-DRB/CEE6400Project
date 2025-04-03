#!/bin/bash
#SBATCH --output=logs/tmp.out  # Standard output log file with job ID
#SBATCH --error=logs/tmp.err   # Standard error log file with job ID

# Define the path to the logs directory
LOGS_DIR="./logs"

# Check if the directory exists
if [ -d "$LOGS_DIR" ]; then
    # Remove all files in the logs directory
    rm -rf "$LOGS_DIR"/*
fi