#!/bin/bash

set -e  # Exit on any error

echo "Executing multi-step MOEAFramework pipeline"

# The only thing you need to change in this file
epsilon="0.01,0.01,0.01"

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "[ERROR] Java is not installed or not found in PATH."
    echo "Please install Java Development Kit (JDK)."
    exit 1
fi

# Check for the expected JAR file
jarFile=""
jarURL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v4.5/MOEAFramework-4.5-Demo.jar"
jarName="MOEAFramework-4.5-Demo.jar"

for file in *Demo.jar; do
    if [ -f "$file" ]; then
        jarFile="$file"
        break
    fi
done

if [ -z "$jarFile" ]; then
    echo "\n[ERROR] MOEAFramework Demo JAR file not found in the current directory."
    echo "Downloading $jarName from: $jarURL"
    
    wget "$jarURL" -O "$jarName"
    
    if [ -f "$jarName" ]; then
        echo "Download complete."
        jarFile="$jarName"
    else
        echo "[ERROR] Failed to download the file."
        exit 1
    fi
fi

# Java and JAR file found
echo "Java is installed and MOEAFramework JAR file found: $jarFile"
echo "Proceeding with execution..."

# Run step 1: Merging result files
echo "Running step 1: Merging result files..."
# Count the number of elements in the epsilon array to set dimension
dimension=$(echo "$epsilon" | awk -F, '{print NF}')

# Loop over all .runtime files in the current directory
for input_file in *.runtime; do
    if [ -f "$input_file" ]; then
        output_file="${input_file%.runtime}.set"
        echo "Processing $input_file"

        java -cp "$jarFile" \
            org.moeaframework.analysis.tools.ResultFileMerger \
            --dimension "$dimension" \
            --output "$output_file" \
            --epsilon "$epsilon" \
            "$input_file"
    fi
done

echo "Merging complete."

# Run step 2: Merging reference set
echo "Running step 2: Merging reference set..."
# Loop over all .set files in the current directory
for input_file in *.set; do
    if [ -f "$input_file" ]; then
        echo "Processing $input_file"

        java -cp "$jarFile" \
            org.moeaframework.analysis.tools.ReferenceSetMerger \
            --output borg.ref \
            --epsilon "$epsilon" \
            "$input_file"
    fi
done

# Run step 3: Evaluating result files
echo "Running step 3: Evaluating result files..."
# Loop over all .runtime files in the current directory
for input_file in *.runtime; do
    if [ -f "$input_file" ]; then
        output_file="${input_file%.runtime}.metrics"
        echo "Evaluating $input_file"

        java -cp "$jarFile" \
            org.moeaframework.analysis.tools.ResultFileEvaluator \
            --dimension "$dimension" \
            --epsilon "$epsilon" \
            --input "$input_file" \
            --reference borg.ref \
            --output "$output_file"
    fi
done

echo "All tasks completed successfully."