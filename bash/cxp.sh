MODELS=("CLIP" "BLIP2" "PubMedCLIP" "BiomedCLIP") #"CLIP" "BLIP2"

for MODEL in "${MODELS[@]}"; do
    echo "Running CXP for model: $MODEL"
    python main.py --task cls --usage unsup-clip-zs --dataset CXP --sensitive_name Sex --model "$MODEL" --method unsup
done

# python main.py --task cls --usage unsup-clip-zs --dataset CXP --sensitive_name Sex --model "$MODEL" --method unsup
