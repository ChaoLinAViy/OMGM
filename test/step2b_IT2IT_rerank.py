from argparse import ArgumentParser
import json, tqdm
from utils import load_csv_data, get_test_question, get_image, get_title2wikiimg
import PIL
from lavis.models import load_model_and_preprocess
import torch
from data_utils import  targetpad_transform, process_images_in_parallel
import os
from model import WikipediaKnowledgeBase, reconstruct_wiki_sections



def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    sample_file_path: str,
    wiki_img_csv_dir: str,
    wiki_img_path_prefix: str,
    step1_result: str,
    knowledge_base_path: str,
    top_ks: list,
    **kwargs
):
    sample_list, sample_header = load_csv_data(sample_file_path)
    wiki_img_csv_path_format = wiki_img_csv_dir + "wiki_image_url_part_{split_num}_processed.csv"
    title2wikiimg = get_title2wikiimg(wiki_img_csv_path_format,wiki_img_path_prefix)
        
    step1_result_dict = json.load(open(step1_result, "r"))
    print("Knowledge Base Loading")
    knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
    knowledge_base_list = knowledge_base.load_knowledge_base()
    knowledge_base_dict = {entry_info.url: entry_info for entry_info in knowledge_base_list}
    del knowledge_base, knowledge_base_list
    print("Knowledge Base Loaded")

    reranker_model, vis_processors, txt_processors = load_model_and_preprocess(
            name="qformer_IT2IT_reranker", model_type="pretrain", is_eval=True, device="cuda"
        )

    if kwargs["reranker_ckpt_path"] != "None":
        print("Load reranker model checkpoint from: ", kwargs["reranker_ckpt_path"])
        checkpoint_path = kwargs["reranker_ckpt_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        msg = reranker_model.load_state_dict(checkpoint, strict=False)
        print("Missing keys {}".format(msg.missing_keys))
    else:
        print("No reranker model checkpoint loaded")

    reranker_model = reranker_model.half()

    if kwargs["vis_process"] == 1:
        print("Use vis_process")
        preprocess = vis_processors["eval"]
    else:
        print("Use targetpad")
        preprocess = targetpad_transform(1.25, 224)

    recalls = {k: 0 for k in top_ks}
    retrieval_result = {}
    sec_top1_recall = 0

    for it, sample_example in tqdm.tqdm(enumerate(sample_list), desc="Step2 IT2IT Rerank"):
        example = get_test_question(it, sample_list, sample_header)
        
        ground_truth = example["wikipedia_url"]

        image = PIL.Image.open(
            get_image(
                example["dataset_image_ids"].split("|")[0],
                example["dataset_name"],  
            ) #iNat_id2name,
        ).convert("RGB")
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        
        step1_retrieval_result = step1_result_dict[data_id]
        reference_image = preprocess(image).to("cuda").unsqueeze(0)
        question = txt_processors["eval"](example["question"])
        step1_sim = step1_retrieval_result["retrieval_similarities"]
        step1_entities = step1_retrieval_result['retrieved_entities']

        IT2IT_null = []
        candidate_images = []
        candidate_sections = []
        entry2ITindex= {}
        for entry_id, title_url in enumerate(step1_entities):
            wiki_title = title_url['title']
            if wiki_title not in title2wikiimg or title2wikiimg[wiki_title] == []: # entity does not have image, skip this entity
                IT2IT_null.append(True)
                continue

          
            wiki_img_path = title2wikiimg[wiki_title][0]['img_path']
            
            if os.path.exists(wiki_img_path) == False: # The image path is not exist, skip this entity
                IT2IT_null.append(True)
                continue
            
            entry = knowledge_base_dict[title_url['url']]
            entry_sections = reconstruct_wiki_sections(entry)
            if len(entry_sections) == 0: # no section in the entity, skip this entity
                IT2IT_null.append(True)
                continue
            candidate_images.extend(
                [PIL.Image.open(wiki_img_path).convert("RGB")] * len(entry_sections)
            )
            entry_sections = [txt_processors["eval"](section) for section in entry_sections]
            candidate_sections.extend(entry_sections)
            assert len(candidate_images) == len(candidate_sections)
            entry2ITindex[entry_id] = [len(candidate_sections) - len(entry_sections) + i for i in range(len(entry_sections))]
            IT2IT_null.append(False)

        with torch.cuda.amp.autocast():
            
            query_fusion_embs = reranker_model.extract_features(
                    {"image": reference_image, "text_input": question},
                    mode="multimodal_query",
                )["multimodal_embeds"]
            
            rerank_step = 256
            for candidate_spilit in range(0, len(candidate_images), rerank_step):

                processed_candidate_images = torch.stack(process_images_in_parallel(candidate_images[candidate_spilit : candidate_spilit + rerank_step], preprocess)).to('cuda')
                
                candidate_fusion_embs_split = reranker_model.extract_features(
                    {"image": processed_candidate_images,
                     "text_input": candidate_sections[candidate_spilit : candidate_spilit + rerank_step]
                     },
                    mode="multimodal_candidate",
                )["multimodal_embeds"]
                
                if candidate_spilit == 0:
                    candidate_fusion_embs = candidate_fusion_embs_split
                else:
                    candidate_fusion_embs = torch.cat(
                        (candidate_fusion_embs, candidate_fusion_embs_split), dim=0
                    )
            
            # query token 2 candidate token -> max sum average
            #[1, 32, 256] [93, 32, 256] 
            sim_q2c_token2token = torch.matmul(
                query_fusion_embs, candidate_fusion_embs.permute(0, 2, 1)
            ).squeeze() #[candidate_IT_num, 32, 32]
    
            sim_q2c_querytoken, _ = sim_q2c_token2token.max(-1) #[candidate_IT_num, 32]
            query_token_length = sim_q2c_querytoken.size(-1)
            sim_q2c = sim_q2c_querytoken.sum(-1) / query_token_length
        
            #print(f'IT2IT_null: {IT2IT_null}')
            step2_sim = [0] * len(step1_sim)
            for entry_id, null in enumerate(IT2IT_null):
                if null:
                    continue
                step2_sim[entry_id] = max([sim_q2c[IT_index].item() for IT_index in entry2ITindex[entry_id]])
            alpha_1 = kwargs["step1_alpha"]
            alpha_2 = 1 - alpha_1
            scores = (
                alpha_1 * torch.tensor(step1_sim).to("cuda")
                + alpha_2 * torch.tensor(step2_sim).to("cuda")
            )
            scores, reranked_index = torch.sort(scores, descending=True)
      
        top_k_wiki = [step1_entities[i]['url'] for i in reranked_index]

        assert len(top_k_wiki) == len(reranked_index)

        if kwargs["save_result_path"] != "None":
            top_20_sec_sim = []
            for i in reranked_index:
                if i.item() not in entry2ITindex:
                    top_20_sec_sim.append([])
                    continue
                top_20_sec_sim.append([sim_q2c[IT_index].item() for IT_index in entry2ITindex[i.item()]])
            retrieval_result[data_id] = {
                "retrieved_entities": [step1_entities[i] for i in reranked_index],
                "retrieval_similarities": [
                    sim.item() for sim in scores
                ],
                "sec_sim": top_20_sec_sim
            }
        
        recall = eval_recall(top_k_wiki, ground_truth, top_ks)
        for k in top_ks:
            recalls[k] += recall[k]
        if example["dataset_name"] != "infoseek":
            evidence_sec, _ = reconstruct_wiki_sections(knowledge_base_dict[ground_truth], example["evidence_section_id"])
            evidence_sec = txt_processors["eval"](evidence_sec)
            top1_sec_sim = [sim_q2c[IT_index].item() for IT_index in entry2ITindex[reranked_index[0].item()]]
            max_sec_sim_index = top1_sec_sim.index(max(top1_sec_sim))
            max_sim_sec = candidate_sections[entry2ITindex[reranked_index[0].item()][max_sec_sim_index]]
            if evidence_sec == max_sim_sec:
                sec_top1_recall += 1
            


    for k in top_ks:
        print("Avg Recall@{}: ".format(k), recalls[k] / (it+1))

    if example["dataset_name"] != "infoseek":
        print("Top1 Evidence Section Recall: ", sec_top1_recall / (it+1))

    if kwargs["save_result_path"] != "None":
        os.makedirs(os.path.dirname(kwargs["save_result_path"]), exist_ok=True)
        print("Save retrieval result to: ", kwargs["save_result_path"])
        with open(os.path.join(kwargs["save_result_path"],'result.json'), "w") as f:
            json.dump(retrieval_result, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument(
        "--step1_result", type=str, required=True, help="Path to the step1 retrieval result"
    )
    parser.add_argument("--reranker_ckpt_path", type=str, required=True, help="Path to the reranker model checkpoint")
    parser.add_argument("--knowledge_base_path", type=str, required=True, help="Path to the knowledge base")
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20,100",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--step1_alpha", type=float, default=0.9, help="alpha for step1 retrieval sim in step2 mix sim")
    parser.add_argument(
        "--wiki_img_csv_dir", default = '../../datasets/wiki_img/full/output/', type=str, help="Path to the wiki image csv directory"
    )
    parser.add_argument(
        "--wiki_img_path_prefix", default = '../../datasets/wiki_img/', type=str, help="Path to the wiki image csv directory"
    )
    
    parser.add_argument("--save_result_path", type=str, default="None", help="path to save retrieval result")
    parser.add_argument("--vis_process", type=int, default=0, help="0 for using target pad, 1 for using vis_process")
    
    args = parser.parse_args()

    test_config = {
        "sample_file_path": args.sample_file,
        "step1_result": args.step1_result,
        "reranker_ckpt_path": args.reranker_ckpt_path,
        "knowledge_base_path": args.knowledge_base_path,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "step1_alpha": args.step1_alpha,
        "wiki_img_csv_dir": args.wiki_img_csv_dir,
        "wiki_img_path_prefix": args.wiki_img_path_prefix,
        "save_result_path": args.save_result_path,
        "vis_process": args.vis_process
    }
    print("step2 test_config: ", test_config)
    run_test(**test_config)