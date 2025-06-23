#!/bin/bash
# Usage: ./process_file.sh merged_refset_file objs_file

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <merged_refset_file> <objs_file>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
NUM_OBJS=4

# The output CSV file will be created or overwritten by the awk command below.

# OPTIONAL: Define the new header row with the specified labels
# HEADER="OBJ1, OBJ2, OBJ3"

# OPTIONAL: Write the header to the output file
# echo "$HEADER" > "$OUTPUT_FILE"

# Process each line:
# - Determine the starting field (if the line has fewer than five fields, start at field 1)
# - Print the fields as comma-separated values into the user-defined output file.
awk -v num_objs="$((NUM_OBJS))" '{
    if ($0 ~ /^#/) next;
    start = (NF - num_objs + 1 > 0) ? NF - num_objs + 1 : 1;
    for(i = start; i < NF; i++){
        printf "%s,", $i;
    }
    print $NF;
}' "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Extracted last $NUM_OBJS columns have been saved as a CSV file in $OUTPUT_FILE."

# Check if the last line of the input file contains a '#' character.
last_line=$(tail -n 1 "$INPUT_FILE")
if [[ "$last_line" != *"#"* ]]; then
    # Ensure the input file ends with a newline and append a line with '#'
    echo "#" >> "$INPUT_FILE"
    echo "A '#' has been appended to the input file."
else
    echo "The last line of the input file already contains a '#'."
fi