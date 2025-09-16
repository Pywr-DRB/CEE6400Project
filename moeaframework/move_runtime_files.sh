#!/bin/bash

# Set output root
OUTPUT_DIR="outputs"

# Define known policies and reservoirs
POLICIES=("STARFIT" "RBF" "PWL")
RESERVOIRS=("beltzvilleCombined" "fewalter" "prompton")

# Create output folders
for POLICY in "${POLICIES[@]}"; do
    for RES in "${RESERVOIRS[@]}"; do
        mkdir -p "$OUTPUT_DIR/Policy_${POLICY}/runtime/${RES}"
    done
done

# Move files to their folders
for FILE in "$OUTPUT_DIR"/MMBorg_4M_*; do
    for POLICY in "${POLICIES[@]}"; do
        for RES in "${RESERVOIRS[@]}"; do
            if [[ "$FILE" == *"MMBorg_4M_${POLICY}_${RES}"* ]]; then
                TARGET_DIR="$OUTPUT_DIR/Policy_${POLICY}/runtime/${RES}"
                echo "Moving $(basename "$FILE") → $TARGET_DIR"
                mv "$FILE" "$TARGET_DIR/"
                break 2  # move to next file once matched
            fi
        done
    done
done

echo "All output files organized by policy and reservoir."
