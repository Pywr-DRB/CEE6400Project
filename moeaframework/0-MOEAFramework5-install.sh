#!/bin/bash

# Set the path to the CLI executable
MOEAFramework5Path="MOEAFramework-5.0"
cliPath="$MOEAFramework5Path/cli"

# Set MOEAFramework download info
MOEAFrameworkURL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v5.0/MOEAFramework-5.0.tar.gz"
MOEAFrameworkTar="MOEAFramework-5.0.tar.gz"

# Check if MOEAFramework directory exists
if [ ! -d "$MOEAFramework5Path" ]; then
    echo "MOEAFramework-5.0 not found. Downloading..."

    # Download using curl or wget
    curl -L -o "$MOEAFrameworkTar" "$MOEAFrameworkURL"

    # Extract using tar
    tar -xzf "$MOEAFrameworkTar" -C ./

    # Clean up
    rm "$MOEAFrameworkTar"
fi

# Check the permission is given
if [ ! -x "$cliPath" ]; then
    echo "Error: CLI at $cliPath is not executable. Run:"
    echo "chmod 775 $cliPath"
    exit 1
fi
