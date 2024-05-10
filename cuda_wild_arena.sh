#!/bin/bash

# Define arrays of parameter values
declare -a hidden_vals=(100 200 1000)
declare -a style_vals=("ts" "ucb")
declare -a nu_vals=("1e-4" "1e-5" "1e-6" "1e-7")
declare -a lambda_vals=("1e-1" "1e-2" "1e-3")
declare -a lr_vals=("1e-2" "1e-3" "1e-4")
declare -a epoch_vals=(100 300)

# Path variables
X_path="/home/bowen/dataset/wild_arena/gemini_judge_full.npz"
y_path="/home/bowen/dataset/wild_arena/5k_embeddings_fresh_new.npz"

# Loop through each combination of parameters
for hidden in "${hidden_vals[@]}"; do
  for style in "${style_vals[@]}"; do
    for nu in "${nu_vals[@]}"; do
      for lambda in "${lambda_vals[@]}"; do
        for lr in "${lr_vals[@]}"; do
          for epochs in "${epoch_vals[@]}"; do
            # Execute the python command
            python train.py --encoding multi --learner neural --seed 1 --custom True --dataset wild_arena --inv diag \
            --X_path $X_path \
            --y_path $y_path \
            --hidden $hidden \
            --style $style \
            --nu $nu \
            --lamdba $lambda \
            --lr $lr --epochs $epochs
          done
        done
      done
    done
  done
done
    

