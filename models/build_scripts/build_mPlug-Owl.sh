currentPath=$(dirname "$0")
cd $currentPath/../mPLUG-Owl
conda create -n mplugowl python=3.9
conda activate mplugowl
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip uninstall protobuf
pip install protobuf==3.19.6
pip install scikit-learn