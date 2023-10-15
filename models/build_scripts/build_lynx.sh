currentPath=$(dirname "$0")
cd $currentPath/../interfaces/lynx
conda env create -f environment.yml