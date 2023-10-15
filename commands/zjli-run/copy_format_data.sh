for dc in "imageBind_llm" "minigpt4" "imagebindLLM" "pandagpt" "lynx" "cheetor" "shikra" "bliva" "llama_adapterv2"
do
cp -r output/$dc/VQA_MR_naive/ /root/tmp_data/format_hit_test/$dc
done