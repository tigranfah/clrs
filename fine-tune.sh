#!/bin/bash

SEEDS=(1 2 3 4 5)
OUTPUT_FILE="logs/lora/bfs_samples=2.log"

mkdir -p logs/lora

echo "Running multi-seed fine-tuning" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for seed in "${SEEDS[@]}"; do
  echo "Seed: $seed" >> "$OUTPUT_FILE"

  python run.py \
    --seed="$seed" \
    --algorithms=bfs \
    --checkpoint_path=checkpoints \
    --shared_encoders_decoders \
    --encoder_decoder_rank=2 \
    --freeze_processor \
    --freeze_encoders_decoders_base \
    --train_multiplier=0.002 \
    --batch_size=2 \
    --train_steps=2000 \
    >> "$OUTPUT_FILE" 2>&1

  echo "" >> "$OUTPUT_FILE"
  echo "Seed: $seed done!" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
done

echo "Logges saved to $OUTPUT_FILE"