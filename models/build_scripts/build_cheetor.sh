currentPath=$(dirname "$0")
cd $currentPath/../Cheetah/Cheetah
conda create -n cheetah python=3.8
conda activate cheetah
pip install -r requirement.txt