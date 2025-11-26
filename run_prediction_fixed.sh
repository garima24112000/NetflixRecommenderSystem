#!/bin/bash

CHECKPOINT="final_model_v2/checkpoint.pth"
PAIRS="data/processed/probe_pairs.parquet"
RATINGS="data/processed/probe_ratings.parquet"
OUTPUT="results/test_predictions.csv"

# Use the FIXED version with mapping
python predict_test_fixed.py \
    --checkpoint $CHECKPOINT \
    --pairs $PAIRS \
    --ratings $RATINGS \
    --output $OUTPUT \
    --device cuda \
    --batch-size 16384

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
if [ -f results/test_predictions_summary.txt ]; then
    cat results/test_predictions_summary.txt
fi
