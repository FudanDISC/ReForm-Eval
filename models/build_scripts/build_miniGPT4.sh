currentPath=$(dirname "$0")
cd $currentPath/../MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4