MODELS=("CLIP" "BLIP2" "PubMedCLIP" "BiomedCLIP")

for MODEL in "${MODELS[@]}"; do
    echo "Running GF3300 for model: $MODEL"
    python main.py --task cls --usage unsup-clip-zs --dataset GF3300 --sensitive_name Sex --model "$MODEL" --method unsup
done

# python main.py --task cls --usage unsup-clip-zs --dataset GF3300 --sensitive_name Sex --model "$MODEL" --method unsup
