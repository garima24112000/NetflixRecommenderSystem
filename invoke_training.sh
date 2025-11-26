# Start the job
python train_dl_v3.py --base ./data/processed --out-dir ./final_model_v2 \
    --epochs 100 --batch-size 1024 --embedding-dim 256 \
    --mlp-layers 256 128 64 32 --accumulation-steps 2 \
    --device cuda --seed 42 \
    --lr 0.0005 \
    --early-stopping --patience 10 \
    --use-wandb --wandb-log-freq 1000 \
    > training.log 2>&1 &

# Save PID
PID=$!
echo $PID > training.pid
echo "Training started with PID: $PID"

# Disown the process (survives terminal closure)
disown

# Monitor
tail -f training.log
