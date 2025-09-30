#!/bin/bash

# set number of seeds and problem name
NUM_SEEDS=10
PROBLEM=PolInf

# Set epsilon values for the objectives, data, runtime, and output directories
epsilon="0.01,0.01,0.01,0.01"
setDir="./refsets"
runtimeDir="./runtime"
metricDir="./metrics"
all_seeds=$(seq 1 $NUM_SEEDS)
refFile_name="STARFIT_fewalter_refset"

# Set the path to the CLI executable
MOEAFramework5Path="MOEAFramework-5.0"
cliPath="$MOEAFramework5Path/cli"

# Check the permission is given
if [ ! -x "$cliPath" ]; then
    echo "Error: CLI at $cliPath is not executable. Run:"
    echo "chmod +x $cliPath"
    exit 1
fi

refFile="$setDir/$refFile_name.ref"  ## Your approx reference set file 

# Merge all .set files across the seeds in the reference set directory
fileList=()
for f in "$setDir"/*.set; do
    fileList+=("$f")

    # Step 1: Check if file contains '# Version=5'
    if ! grep -q "# Version=5" "$f"; then
        echo "Adding header to $f"
        temp_file=$(mktemp)  # create a temporary file safely
        {
            cat 1-header-file.txt
            echo ""            # add a newline
            cat "$f"
        } > "$temp_file"      # group the commands inside braces
        mv "$temp_file" "$f"  # replace original file with the new one
    fi
done

echo "Merging all seeds' reference sets to create the approximate reference set"

if [ -f "$refFile" ]; then
    "$cliPath" ResultFileMerger --problem $PROBLEM --epsilon "$epsilon" --output "$refFile" "${fileList[@]}"  --overwrite
else
    "$cliPath" ResultFileMerger --problem $PROBLEM --epsilon "$epsilon" --output "$refFile" "${fileList[@]}" 
fi

echo "Approximate reference set created at $refFile"