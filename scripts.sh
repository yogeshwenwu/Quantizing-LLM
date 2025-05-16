#!/bin/bash

source /mnt/mydisk/yogesh/anaconda3/etc/profile.d/conda.sh

echo "<--------Activated-------->"
conda activate env_py_3.10_torch_2.4
echo "Current conda environment: $CONDA_DEFAULT_ENV"

cd /mnt/mydisk/yogesh/Quantizing-LLM # project location
cd /GGUF

python hfQuant.py &>> output.txt 

# conda deactivate
# echo "<--------Deactivated-------->"
