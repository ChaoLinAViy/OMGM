# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval

<p align="center">
<img src=assets/main.png width=700/>
</p>

This is the official implementation of [OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](https://arxiv.org/abs/2505.07879) (ACL 2025 Main Conference).


## 🛠️ Requirements
1. Create conda environment

```bash
conda create -n OMGM python=3.10
conda activate OMGM
```

2. Install the required packages
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## 📑 VQA Data and Knowledge Base
We provide the vqa data and knowledge bases used in OMGM. To facilitate comparison with related baselines, we use the VQA data processed by EchoSight, which involves cleaning a small number of samples missing entity images. Additionally, the knowledge base for InfoSeek is the same 100k-sample subset sampled from the E-VQA 2M knowledge base as used by EchoSight. The download links are provided below:

### Enclyclopedic-VQA

To download the VQA samples in E-VQA:

*   [train.csv](https://drive.google.com/file/d/13BZZAserLlqKT_RHq4sX5NAIu7Ii77eE/view?usp=drive_link)
*   [val.csv](https://storage.googleapis.com/encyclopedic-vqa/val.csv)
*   [test.csv](https://storage.googleapis.com/encyclopedic-vqa/test.csv)

To download the images in E-VQA:

- [iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021) ( You should change the image name to image_id with [train_id2name](https://drive.google.com/file/d/1cUP0sWtI4z7whH9V5FOvqfJ0LTxZLOd9/view?usp=drive_link) and [val_id2name](https://drive.google.com/file/d/1cYzo4qewPABFuoMhpME4j2DWAA_Y-l2L/view?usp=drive_link) for the direct connect to the dataset_image_id in vqa sample )

- [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark)

Remember to set the `GLD_image_path` and `iNat_image_path` in `utils/utils.py` to the path of each of the images dir.

To download the 2M Knowledge Base in E-VQA:

- [Encylopedic-VQA's 2M Knowledge Base](https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip)



### Infoseek

To download the VQA samples in InfoSeek:

* [train.csv](https://drive.google.com/file/d/1cQiQmdFq8_8gsaZPsmzKzIjZhdcd_kxP/view?usp=drive_link)
* [test.csv](https://drive.google.com/file/d/1cSG_dVuao9lKZy8vaUDWEo7mIHowjUeE/view?usp=drive_link)

To download the images in InfoSeek:

- [Oven](https://github.com/edchengg/oven_eval/tree/main/image_downloads)

Remember to set the `infoseek_test_path` and `infoseek_train_path` in `utils/utils.py` to the path of test and train images dir.

To download the 100k Knowledge Base in E-VQA:

- [InfoSeek's 100K Knowledge Base ](https://drive.google.com/file/d/1cIbKtYryD7XBAw0tjrrCvMCJC2rIzLM5/view?usp=drive_link)


### Wiki Images

Regarding the image data of entity webpages in the knowledge base, we re-crawled the images from Wikipedia to ensure completeness. The entity information, image URLs, and the relative paths for storing the images are recorded in 13 CSV files. These CSV files are provided to facilitate the downloading of knowledge base images:


- [Wiki Images CSVs](https://drive.google.com/file/d/1__laja2XMKA-J3oBT7EFdLxLCEGfpw4r/view?usp=drive_link)

For each CSV file, a corresponding folder needs to be created to download all associated images. The image URLs and their storage paths can be found in the image_URL and img_path columns of the CSV files. After downloading, the directory structure of the wiki images should be as follows:

<pre>.
├── full
│   └── wiki_image_split
│       ├── wiki_entity_image_1
│       ├── wiki_entity_image_2
│       ├── wiki_entity_image_3
│       ├── wiki_entity_image_4
│         ...
│       └── wiki_entity_image_13
└── output
    ├── wiki_image_url_part_1_processed.csv
    ├── wiki_image_url_part_2_processed.csv
    ├── wiki_image_url_part_3_processed.csv
      ... 
    └── wiki_image_url_part_13_processed.csv</pre>





## 🔍 Step 1 : Coarse-Grained Cross-Modal Entity Searching

In the initial entity search stage, OMGM utilizes the summaries of entity documents as candidates. We provide the FAISS indexes of the summaries generated for E-VQA and InfoSeek to facilitate direct reproduction of the Step 1 process:

- [E-VQA KB Summary Faiss Index](https://drive.google.com/file/d/1FPx1EOTjcx8zXPoPD9fBQ6cbucF0fL_D/view?usp=drive_link)

- [InfoSeek KB Summary Faiss Index](https://drive.google.com/file/d/1CE-SyPNKEx6fT09LXoMh4ckmh7jfzG_E/view?usp=drive_link)


To perform the step 1 of OMGM, run the following bash script after changing the necessary configurations.
```bash
bash scripts/step1_img2sum.sh
```

### Script Details
The step1_img2sum.sh script is used to complete OMGM’s preliminary coarse-grained cross-modal entity searching with specific parameters:

--`sample_file`: Path to the test file with CSV format.

--`knowledge_base`: Path to the knowledge base JSON file.

--`faiss_index`: Path to the FAISS index file of entities' summary.

--`save_result_path`: Path to the step1 result json file would be saved.

--`retriever_vit`: Name of the visual transformer model used for initial retrieval. In the example script, eva-clip is used.

--`top_ks`: Comma-separated list of top-k recall results for retrieval (e.g., 1,5,10,20).

--`retrieval_top_k`: The top-k value used for retrieval.


## 🔥 Step 2a : IT2IT Multimodal Fusion Reranker Training
The multimodal fusion reranker of OMGM is trained using E-VQA datasets. The top 20 entities in the training samples are obtained through Step 1 processing. Based on these top 20 entities, we further select 15 negative pairs for each sample following the method described in the paper. We provide our Hard_Neg result file as follows:

- [Hard_Neg Tranning Samples For Reranker](https://drive.google.com/file/d/1X9sOZV5jSielgszfIcncvcoCtDumHkBD/view?usp=drive_link)


To train the multimodal fusion reranker, run the bash script after changing the necessary configurations.

```bash
bash scripts/IT2IT_reranker_training.sh
```

### Script Details
The IT2IT_reranker_training.sh script is used to fine-tune the multimodal fusion reranker module with specific parameters:

--`reranker-model-name`: Name of the Reranker model. The default value for this item is `qformer_IT2IT_reranker`, which corresponds to the multimodal fusion reranker designed in Step 2 of OMGM.

--`dataset_name`: The name of dataset used for training. In our work, the dataset is `E-VQA`.

--`num-epochs`: Number of epochs for training. In our work, the reranker is trained for 1 epochs.

--`num-workers`: Number of worker threads for data loading.

--`learning-rate`: Learning rate for the optimizer. The default value is `1e-5`.

--`batch-size`: Number of samples per batch during training. The default value is `6`.

--`neg-num`: The number of negative pairs included in the each of the training samples. In our work, each sample has 15 corresponding negative pairs.

--`save_frequency`: How many epochs between each model checkpoint save. The default value is `0.25`.

--`train_file`: Path to the training data file. The training file should be the same format as provided by E-VQA.

--`knowledge_base_file`: Path to the knowledge base file in JSON format. The format should be the same with that of the E-VQA.

--`negative_db_file`: Path to the hard negative sampled database file used for training.

--`wiki_img_csv_dir`: The folder path containing the CSV file that records information related to Wiki images.

--`wiki_img_path_prefix`: The path to the parent folder containing the `full/wiki_image_split` folder, which holds all the wiki images from different subsets.

--`save-training`: Flag to save the training progress.

--`wandb`: Whether to use wandb for training process logging; set the value to 1 if used, otherwise 0. Remember to set your wandb API key in the shell using `WANDB_API_KEY` if you want to use wandb.

## 🔬 Step 2b : Hybrid-Grained Multimodal-Fused Reranking
After the training in Step 2a described above, we obtain a multimodal fusion reranker capable of performing image-text matching between the query and candidate sides. Here, we provide the trained reranker weights primarily used in our experiments:

- [Multimodal Fusion Reranker Weights](https://drive.google.com/file/d/1K1Mxn_ePj1cCHdbgESK8Wz_Xrw1bhBEM/view?usp=drive_link)


Using this reranker, we can perform hybrid-grained multimodal-fused reranking by running the following command after making the necessary parameter adjustments.

```bash
bash scripts/step2_IT2IT_rerank.sh
```

### Script Details
The step2_IT2IT_rerank.sh script is used to perform entities reranking with specific parameters:

--`sample_file`: Path to the test file with CSV format.

--`step1_result`: Path to the result json file of step 1.

--`reranker_ckpt_path`: Path to the trained multimodal fusion reranker checkpoint file. 

--`knowledge_base_path`: Path to the knowledge base file in JSON format. The format should be the same with that of the E-VQA.

--`top_ks`: Comma-separated list of top-k recall results for retrieval (e.g., 1,5,10,20).

--`step1_alpha`: The weight of the entity similarity from Step 1 in the calculation of the weighted similarity when performing cross-stage similarity fusion between Step 1 and Step 2. The default value is `0.9`.

--`wiki_img_csv_dir`: The folder path containing the CSV file that records information related to Wiki images.

--`wiki_img_path_prefix`: The path to the parent folder containing the `full/wiki_image_split` folder, which holds all the wiki images from different subsets.

--`save_result_path`: Path to the result json file would be saved.

## 💡 Step 3 : Fine-Grained Section-Augmented Generation
In this stage, the generator model we use can be either a pretrained LLM/MLLM or a fine-tuned model. The fine-tuned LLaVA-1.5-7B mentioned in the main results of the paper is detailed in the Appendix C.3. Here, we provide the corresponding fine-tuned model weights:

- [Finetuned LLaVA-1.5-7B Generator](https://drive.google.com/file/d/17gaVltG2acdWKyG8oBKsrvUURNnqVZGD/view?usp=drive_link)

By running the following command, we can perform top-1 section selection within the top-1 entity using the text similarity between the section and the question, along with the multimodal similarity from Step 2, to assist the generator in answering questions.

```bash
bash scripts/step3_secaugmented_gen.sh
```

### Script Details
The step3_secaugmented_gen.sh script is used to perform section-augmented generation with specific parameters:

--`test_file`: Path to the test file with CSV format.

--`retrieval_results_file`: Path to the result json file of step 2.

--`step2_beta`: The weight of the section similarity from Step 2 in the calculation of the weighted similarity when performing cross-stage similarity fusion between Step 2 and Step 3. The default value is `0.2`. 

--`answer_generator`: Name of the answer generator model to be used. Choose from [llama3, mistral, llava1_5, internvl2_5].

--`knowledge_base_path`: Path to the knowledge base file in JSON format. The format should be the same with that of the E-VQA.

--`output_file`: Path to the result json file would be saved.

--`llm_checkpoint`: Path to the local LLM checkpoint file. If not using a checkpoint, this parameter can be omitted or set to `None`.


## 📖 Citation
```
@article{yang2025omgm,
  title={OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval},
  author={Yang, Wei and Fu, Jingjing and Wang, Rui and Wang, Jinyu and Song, Lei and Bian, Jiang},
  journal={arXiv preprint arXiv:2505.07879},
  year={2025}
}
```
## 💖 Acknowledgements
Thanks to the code of [LAVIS](https://github.com/salesforce/LAVIS/tree/main) and [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main/) and data of [Encyclopedic-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa), [EchoSight](https://github.com/Go2Heart/EchoSight) and [InfoSeek](https://github.com/open-vision-language/infoseek).