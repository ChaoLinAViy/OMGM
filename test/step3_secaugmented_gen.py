from argparse import ArgumentParser
import json
from model import (
    MistralAnswerGenerator,
    LLaMA3AnswerGenerator,
    LlaVA1_5AnswerGenerator,
    InterVL2_5AnswerGenerator,
    BGESectionReranker,
    WikipediaKnowledgeBase,
    reconstruct_wiki_sections
)
from utils import evaluate_example, evaluate, load_csv_data, get_test_question
import tqdm
import os
import torch

from utils import load_csv_data, get_test_question, get_image


MODEL_NAME2ID = {
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llava1_5": "llava-hf/llava-1.5-7b-hf",
        "internvl2_5": "OpenGVLab/InternVL2_5-8B",
    }

def run_vqa(
    answer_generator, test_file, model_type, retrieval_results_file, output_file, knowledge_base_path, step2_beta
):
    
    q2c_ranker = BGESectionReranker(model_path="BAAI/bge-reranker-v2-m3",device="cuda")


    print("Loading Knowledge Base")
    knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
    knowledge_base_list = knowledge_base.load_knowledge_base()
    knowledge_base_dict = {entry_info.url: entry_info for entry_info in knowledge_base_list}
    del knowledge_base, knowledge_base_list
    print("Knowledge Base Loaded")

    test_list, test_header = load_csv_data(test_file)
    if retrieval_results_file:
        retrieval_results = json.load(open(retrieval_results_file, "r"))
    else:
        raise ValueError("Please provide retrieval results file")
    
    result_list = []
    eval_score = 0
    sec_top1_recall = 0
    for it, example in tqdm.tqdm(enumerate(test_list)):
        sample = get_test_question(it, test_list, test_header)
        ground_truth = sample["wikipedia_url"]
        if sample["dataset_name"] == "infoseek":
            data_id = sample["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)

        if retrieval_results:
            top1_entry = knowledge_base_dict[retrieval_results[data_id]["retrieved_entities"][0]['url']]

        question = sample["question"]

        image_dataset_name = sample["dataset_name"]

        vanilla_input = {"question": question}
        if model_type == 'MLLM':
            image_path = get_image(
                    sample["dataset_image_ids"].split("|")[0],
                    image_dataset_name,  
                ) 
            vanilla_input["image_path"] = image_path

        
        section_list = reconstruct_wiki_sections(top1_entry)

    
        q2s_scores, rank_idx = q2c_ranker.rank_entry_sections(question, section_list)
        q2s_scores = q2s_scores.tolist()
        
        step2_top1_entry_sec_sim = retrieval_results[data_id]['sec_sim'][0]
        assert len(step2_top1_entry_sec_sim) == len(q2s_scores)
        step3_scores = (
            step2_beta * torch.tensor(step2_top1_entry_sec_sim).to("cuda")
            + ( 1 - step2_beta ) * torch.tensor(q2s_scores).to("cuda")
        )
        _, index = torch.sort(step3_scores, descending=True)
        top1_section = section_list[index[0]]
        step3_scores = step3_scores.tolist()
        
        answer = answer_generator.llm_answering(vanilla_input, image_dataset_name, entry_section=top1_section)
           
    
        result_dict = {
                "data_id": data_id,
                "entity_url": sample["wikipedia_url"],
                "question": sample["question"],
                "prediction": answer,
                "gt_answer": sample["answer"],
            }
       
        result_dict["top1_entity_url"] = retrieval_results[data_id]["retrieved_entities"][0]['url']
    
        result_dict["step3_sec_sim"] = step3_scores
        result_dict["top1_sec"] = top1_section
        result_dict["step2_top1_entry_sec_sim"] = step2_top1_entry_sec_sim
        result_dict["q2s_scores"] = q2s_scores

        if image_dataset_name != "infoseek":
            evidence_sec, _ = reconstruct_wiki_sections(knowledge_base_dict[ground_truth], sample["evidence_section_id"])
            if evidence_sec == top1_section:
                sec_top1_recall += 1
            result_dict["evidence_sec"] = evidence_sec
            if evidence_sec in section_list and int(sample["evidence_section_id"]) < len(section_list):
                result_dict["gt_sec_sim"] = q2s_scores[int(sample["evidence_section_id"])]

        result_list.append(result_dict)

        torch.cuda.empty_cache()
   
    if output_file != "None":
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print("Save answer result to: ", output_file)
        with open(os.path.join(output_file,'result.json'), "w") as f:
            json.dump(result_list, f, indent=4)
            
    del q2c_ranker
    del answer_generator

    if image_dataset_name == 'infoseek':
        result = evaluate(None, "../../datasets/OMGM_data/InfoSeek/infoseek_val.jsonl", "../../datasets/OMGM_data/InfoSeek/infoseek_val_qtype.jsonl", result_list)
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]
        print(f"val final score: {final_score}")
        print(f"val unseen question score: {unseen_question_score}")
        print(f"val unseen entity score: {unseen_entity_score}")
        print(f"val unseen question score_time : {result['unseen_question_score']['score_time']}")
        print(f"val unseen question score_num : {result['unseen_question_score']['score_num']}")
        print(f"val unseen question score_string : {result['unseen_question_score']['score_string']}")
        print(f"val unseen entity score_time : {result['unseen_entity_score']['score_time']}")
        print(f"val unseen entity score_num : {result['unseen_entity_score']['score_num']}")
        print(f"val unseen entity score_string : {result['unseen_entity_score']['score_string']}")
    else:
        print("Top1 Evidence Section Recall: ", sec_top1_recall / (it+1))
        for it, example in enumerate(test_list):
            sample = get_test_question(it, test_list, test_header)
            if sample["dataset_name"] == "infoseek":
                data_id = sample["data_id"]
            else:
                data_id = "E-VQA_{}".format(it)
            if result_list[it]['data_id'] != data_id:
                raise ValueError("Data ID mismatch")

            score = evaluate_example(
                        sample["question"],
                        reference_list=sample["answer"].split("|"),
                        candidate=result_list[it]['prediction'],
                        question_type=sample["question_type"],
                    )
            eval_score += score
        print("eval score: ", eval_score / (it + 1))
        



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--retrieval_results_file", type=str, default=None)
    parser.add_argument("--step2_beta", type=float, default=0, help="beta for step2 retrieval it2it sim in step3 fusion sim")
    parser.add_argument("--answer_generator", type=str)
    parser.add_argument("--llm_checkpoint", type=str, default="None")
    parser.add_argument("--knowledge_base_path", type=str)
    parser.add_argument("--output_file", type=str, default="None")
    
    args = parser.parse_args()
    test_file = args.test_file
    step2_beta = args.step2_beta
    generator_name = args.answer_generator.lower()
    llm_checkpoint = args.llm_checkpoint
    output_file = args.output_file
    knowledge_base_path = args.knowledge_base_path
    retrieval_results = args.retrieval_results_file
    

    
    print("retrieval_results:", retrieval_results)
    print("generator_name: ", generator_name)

    if llm_checkpoint == "None":
        llm_checkpoint = MODEL_NAME2ID[generator_name]
    
    print("llm_checkpoint: ", llm_checkpoint)
    if generator_name == "mistral":
        answer_generator = MistralAnswerGenerator(model_path=llm_checkpoint,device="cuda")
        model_type = 'LLM'
    elif generator_name == "llama3":
        answer_generator = LLaMA3AnswerGenerator(model_path=llm_checkpoint,device="cuda")
        model_type = 'LLM'
    elif generator_name == "llava1_5":
        answer_generator = LlaVA1_5AnswerGenerator(model_path=llm_checkpoint,device="cuda")
        model_type = 'MLLM'
    elif generator_name == "internvl2_5":
        answer_generator = InterVL2_5AnswerGenerator(model_path=llm_checkpoint,device="cuda")
        model_type = 'MLLM'
    else:
        raise ValueError("Invalid Answer Generator, Please choose from Mistral, LLaMA3, Internvl2")
    
    run_vqa(answer_generator, test_file, model_type, retrieval_results, output_file, knowledge_base_path, step2_beta)
