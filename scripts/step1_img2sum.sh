export CUDA_VISIBLE_DEVICES=0


python -m test.step1_img2sum \
    --sample_file ../../datasets/OMGM_data/E-VQA/test.csv\
    --knowledge_base ../../datasets/OMGM_data/E-VQA/wiki_corpus.json\
    --faiss_index ../../datasets/OMGM_data/E-VQA/summary_llama3.index\
    --save_result_path ./step1_results/E-VQA/ \
    --retriever_vit eva-clip \
    --top_ks 1,5,10,20 \
    --retrieval_top_k 20\
