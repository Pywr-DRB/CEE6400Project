#!/bin/bash

# Define the path to the outputs directory
OUTPUTS_DIR="./outputs"

# Function to delete files
delete_files() {
    rm -rf "$1"/*
    echo "All files in the $2 have been deleted."
}

# Check if the directory exists
if [ -d "$OUTPUTS_DIR" ]; then
    # Check if a subfolder name is provided as an argument
    if [ -z "$1" ]; then
        echo "No subfolder specified. Cleaning the entire outputs directory."
        echo "Are you sure you want to delete all files in the outputs directory? (yes/no)"
        read confirmation
        if [ "$confirmation" = "yes" ]; then
            delete_files "$OUTPUTS_DIR" "outputs directory"
        else
            echo "Deletion cancelled."
        fi
    else
        SUBFOLDER="$OUTPUTS_DIR/$1"
        # Check if the specified subfolder exists
        if [ -d "$SUBFOLDER" ]; then
            echo "You are about to delete all files in the $1 subfolder."
            echo "Are you sure you want to proceed? (yes/no)"
            read confirmation
            if [ "$confirmation" = "yes" ]; then
                delete_files "$SUBFOLDER" "$1 subfolder"
            else
                echo "Deletion cancelled."
            fi
        else
            echo "Error: Specified subfolder does not exist."
        fi
    fi
else
    echo "Error: Outputs directory does not exist."
fi