currentPath=$(dirname "$0")
cd $currentPath/../LLaMA-Adapter/imagebind_LLM
# create conda env
conda create -n imagebind_LLM python=3.9 -y
conda activate imagebind_LLM
# install ImageBind
cd ImageBind
pip install -r requirements.txt
# install other dependencies
cd ../
pip install -r requirements.txt
pip install ninja
pip install scikit-learn