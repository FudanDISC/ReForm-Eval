currentPath=$(dirname "$0")
cd $currentPath/../LLaMA-Adapter/llama_adapter_v2_multimodal7b
conda create -n llama_adapter_v2 python=3.8 
conda activate llama_adapter_v2
pip install -r requirements.txt
pip install sentencepiece
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
pip install scikit-learn