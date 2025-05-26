export CUDA_VISIBLE_DEVICES=0

python -m test.step2b_IT2IT_rerank \
    --sample_file ../../datasets/OMGM_data/E-VQA/test.csv\
    --step1_result ./step1_results/E-VQA/result.json \
    --reranker_ckpt_path ./reranker_model/qformer_IT2IT_reranker/E-VQA/lr1e-05_bs8_neg_num15/model_23912.pth \
    --knowledge_base_path ../../datasets/OMGM_data/E-VQA/wiki_corpus.json \
    --top_ks 1,5,10,20 \
    --step1_alpha  0.9 \
    --wiki_img_csv_dir ../../datasets/OMGM_data/wiki_img/output/ \
    --wiki_img_path_prefix ../../datasets/OMGM_data/wiki_img/ \
    --save_result_path ./step2_results/E-VQA/ \



