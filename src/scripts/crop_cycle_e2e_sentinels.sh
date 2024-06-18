#!/usr/bin/env bash

# PATHS
ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$ROOT" ]; then
  echo "ROOT repository not found. Are you in the right directory?"
  exit 1
fi
CONFIG_PATH="$ROOT/config/"

# Get the path to the active Python interpreter
PYTHON_INTERPRETER=$(which python)

# Function to display usage information
display_help() {
    echo "Usage: $0 [--load <data_path>]"
    echo "  --load <data_path>: Load villages .shp/.kml files to database"
    echo "  --help: Display this help message"
}

# Function to run load.py with the provided data path if --load flag is provided
run_load() {
    if [[ "$1" == "--load" ]]; then
        if [ -z "$2" ]; then
            echo "Error: Data path missing for --load option"
            exit 1
        else
            chmod +x $ROOT/src/data_loading/load_villages.py
            $PYTHON_INTERPRETER $ROOT/src/data_loading/load_villages.py -p "$2" -s pilot
        fi
    fi
}

# Function to run other Python scripts in sequence
run_scripts() {
    chmod +x $ROOT/src/sentinel_hub/sentinel.py
    chmod +x $ROOT/src/crop_presence_interval/clip_automatic.py
    chmod +x $ROOT/src/crop_presence_interval/crop_presence_inference.py
    chmod +x $ROOT/src/crop_cycle_monthly/crop_cycle_inference_interval.py
    
    $PYTHON_INTERPRETER $ROOT/src/sentinel_hub/sentinel.py
    $PYTHON_INTERPRETER $ROOT/src/crop_presence_interval/clip_automatic.py
    $PYTHON_INTERPRETER $ROOT/src/crop_presence_interval/crop_presence_inference.py
    $PYTHON_INTERPRETER $ROOT/src/crop_cycle_monthly/crop_cycle_inference_interval.py
}

# Main function
main() {
    if [[ "$1" == "--help" ]]; then
        display_help
        exit 0
    fi

    # Check if --load flag is provided with data path
    if [[ "$1" == "--load" ]]; then
        run_load "$1" "$2"
        shift 2
    fi

    # Run other scripts
    run_scripts
}

# Call the main function with the provided arguments
main "$@"
