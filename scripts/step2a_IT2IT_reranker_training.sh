export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY='YOUR_WANDB_API_KEY'

python -m test.step2a_IT2IT_reranker_training \
   --reranker-model-name qformer_IT2IT_reranker \
   --dataset_name E-VQA \
   --num-epochs 1 \
   --num-workers 4 \
   --learning-rate 1e-5 \
   --batch-size 8 \
   --neg-num 15 \
   --save_frequency 0.5 \
   --train_file ../../datasets/OMGM_data/E-VQA/train.csv \
   --knowledge_base_file ../../datasets/OMGM_data/E-VQA/wiki_corpus.json \
   --negative_db_file ../../datasets/OMGM_data/E-VQA/step1_HardNeg.json \
   --wiki_img_csv_dir ../../datasets/OMGM_data/wiki_img/output/ \
   --wiki_img_path_prefix ../../datasets/OMGM_data/wiki_img/ \
   --save-training \
   --wandb 1\
