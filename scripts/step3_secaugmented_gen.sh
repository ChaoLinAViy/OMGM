export CUDA_VISIBLE_DEVICES=0

python -m test.step3_secaugmented_gen \
    --test_file ../../datasets/OMGM_data/E-VQA/test.csv \
    --retrieval_results_file ./step2_results/E-VQA/result.json \
    --step2_beta 0.2 \
    --answer_generator internvl2_5 \
    --knowledge_base_path ../../datasets/OMGM_data/E-VQA/wiki_corpus.json \
    --output_file ./step3_results/E-VQA/ \
    --llm_checkpoint None \

