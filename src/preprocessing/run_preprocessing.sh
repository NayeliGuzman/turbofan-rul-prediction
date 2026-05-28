#!/bin/bash
set -e

INPUT_DIR=$1
OUTPUT_DIR=$2

# Check if required arguments are provided
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: ./run_preprocessing.sh <input_dir> <output_dir>"
  exit 1
fi

# Define threshold if not provided
THRESHOLD=${THRESHOLD:-0.95}

# Create output directory if one doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting preprocessing"
echo "Reading from: $INPUT_DIR"
echo "Writing to:   $OUTPUT_DIR"
echo "----------------------------------------"

# Loop over each dataset ID (FD001 through FD004 only)
for DATASET_ID in FD001 FD002 FD003 FD004; do

  # Verify the train file exists before attempting to process
  TRAIN_FILE="$INPUT_DIR/train_${DATASET_ID}.txt"
  if [ ! -f "$TRAIN_FILE" ]; then
    echo "Skipping $DATASET_ID — train file not found: $TRAIN_FILE"
    continue
  fi

  echo "Processing $DATASET_ID..."
  python3 src/preprocessing/preprocess_nasa.py \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --dataset "$DATASET_ID" 
    
  echo "Finished $DATASET_ID"
  echo "----------------------------------------"

done

echo "All datasets processed."
