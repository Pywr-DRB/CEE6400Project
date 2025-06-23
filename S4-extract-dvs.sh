#!/bin/bash
# Usage: ./extract_dvs.sh merged_refset_file dvs_file

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <merged_refset_file> <dvs_file>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
NUM_DVS=15

# OPTIONAL: Define the new header row with the specified labels
# HEADER="DV1, DV2, DV3, DV4, DV5, DV6, DV7, DV8"

# OPTIONAL: Write the header to the output file
# echo "$HEADER" > "$OUTPUT_FILE"

# Process each line:
# - Print up to the first 30 fields (or all if fewer), comma-separated into the output file.

# skip all lines that begin with a #
awk -v num_dvs="$NUM_DVS" '{
    if ($0 ~ /^#/) next;
    max = (NF < num_dvs) ? NF : num_dvs;
    for(i = 1; i < max; i++){
        printf "%s,", $i;
    }
    print $max;
}' "$INPUT_FILE" >> "$OUTPUT_FILE"


echo "Header and first $NUM_DVS columns have been saved as a CSV file in $OUTPUT_FILE."