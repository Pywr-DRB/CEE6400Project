#!/bin/bash

set -e  # Exit on any error

echo "Compute metrics"

# Default epsilon value if no argument is given
if [ -z "$1" ]; then
    epsilon="0.01,0.01"
    pause_at_end=true
else
    epsilon="$1"
    pause_at_end=false
fi

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

# Count the number of elements in the epsilon array to set dimension
dimension=$(echo "$epsilon" | awk -F, '{print NF}')

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

echo "Metrics calculation complete."

#if [ "$pause_at_end" = true ]; then
#    read -p "Press Enter to exit..."
#fi