#!/bin/bash

# set number of seeds and problem name
NUM_SEEDS=10
NUM_MASTERS=4
PROBLEM=PolInf

# Set epsilon values for the objectives, data, runtime, and output directories
epsilon="0.01,0.01,0.01,0.01"
setDir="./refsets"
runtimeDir="./runtime"
metricDir="./metrics"
all_seeds=$(seq 1 $NUM_SEEDS)
all_masters=$(seq 0 $((NUM_MASTERS-1)))
refFile_name="PiecewiseLinear_fewalter_refset"


refFile="$setDir/$refFile_name.ref"

# Set the path to the CLI executable
MOEAFramework5Path="MOEAFramework-5.0"
cliPath="$MOEAFramework5Path/cli"

# Check the permission is given
if [ ! -x "$cliPath" ]; then
    echo "Error: CLI at $cliPath is not executable. Run:"
    echo "chmod 775 $cliPath"
    exit 1
fi

# Check if the metrics directory exists. If not, create it
if [ ! -d "$metricDir" ]; then
    echo "$metricDir not found. Creating it..."
    mkdir -p "$metricDir"
fi

echo "Calculating runtime metrics across all seeds and masters..."
# Step 2: Evaluate metrics for runtime files
for s in $all_seeds; do
    for master in $all_masters; do

        # change the filename here to match your output convention
        filename="MMBorg_4M_PiecewiseLinear_fewalter_nfe30000_seed${s}_${master}"
        infile="$runtimeDir/$filename.runtime"
        outfile="$metricDir/${filename}.metric"

        # check if the output file already exists
        # overwrite it if it does
        if [ -f "$outfile" ]; then
            "$cliPath" MetricsEvaluator --problem $PROBLEM --epsilon "$epsilon" --input "$infile" --output "$outfile" --reference "$refFile" --overwrite
        else

        if ! grep -q "# Version=5" "$infile"; then
            echo "Adding header to $infile"
            temp_file=$(mktemp)  # create a temporary file safely
            {
                cat 1-header-file.txt
                echo ""            # add a newline
                cat "$infile"
            } > "$temp_file"      # group the commands inside braces
            mv "$temp_file" "$infile"  # replace original file with the new one
        fi
        "$cliPath" MetricsEvaluator --problem $PROBLEM --epsilon "$epsilon" --input "$infile" --output "$outfile" --reference "$refFile" 
        fi

        # Remove leading '#' from the first line of the .metric file if present
        if [ -f "$outfile" ]; then
            sed -i '1s/^#\s*//' "$outfile"
        fi
    done
done
