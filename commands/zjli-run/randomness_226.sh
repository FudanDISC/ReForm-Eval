# conda activate imagebind_LLM
# python utils/randomess_runner.py --models imagebind_llm  --devices 0,1,2,3,4,5,6,7
conda activate blip2
python utils/randomess_runner.py --models blip2_t5  --devices 0,1,2,3,4,5,6,7
conda activate cheetah
python utils/randomess_runner.py --models cheetor_vicuna,cheetor_llama2  --devices 0,1,2,3,4,5,6,7
conda activate shikra
python utils/randomess_runner.py --models shikra  --devices 0,1,2,3,4,5,6,7
conda activate bliva
python utils/randomess_runner.py --models bliva  --devices 0,1,2,3,4,5,6,7
conda activate multimodal-gpt
python utils/randomess_runner.py --models mmgpt  --devices 0,1,2,3,4,5,6,7




