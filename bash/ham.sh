MODELS=("CLIP" "BLIP2" "PubMedCLIP" "BiomedCLIP")

for MODEL in "${MODELS[@]}"; do
    echo "Running PAPILA for model: $MODEL"
    python main.py --task cls --usage unsup-clip-zs --dataset HAM10000 --sensitive_name Sex --model "$MODEL" --method unsup
done

# python main.py --task cls --usage unsup-clip-zs --dataset HAM10000 --sensitive_name Sex --model "$MODEL" --method unsup
