from argparse import ArgumentParser
import json, tqdm
from model import ClipRetriever
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
import PIL
import os


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    sample_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    **kwargs
):
    sample_list, sample_header = load_csv_data(sample_file_path)
    
    retriever = ClipRetriever(device="cuda:0", model=kwargs["retriever_vit"])
    print("Knowledge Base Loading")
    retriever.load_knowledge_base(knowledge_base_path) 
    retriever.load_entity_faiss_index(faiss_index_path)
    print("Knowledge Base Loaded")

    recalls = {k: 0 for k in top_ks}
    retrieval_result = {}
    for it, sample_example in tqdm.tqdm(enumerate(sample_list), desc="Step1 Img2Sum"):
        example = get_test_question(it, sample_list, sample_header)
        
        ground_truth = example["wikipedia_url"]

        image = PIL.Image.open(
            get_image(
                example["dataset_image_ids"].split("|")[0],
                example["dataset_name"],  
            ) 
        )
        
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)

        top_k = retriever.I2T_faiss(image, top_k=retrieval_top_k)
        top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
        top_k_wiki = remove_list_duplicates(top_k_wiki)
        assert len(top_k_wiki) == len(top_k)

        if kwargs["save_result_path"] != "None":
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            entries = remove_list_duplicates(entries)
            seen = set()
            retrieval_simlarities = [
                top_k[i]["similarity"]
                for i in range(retrieval_top_k)
                if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
            ]
            retrieval_result[data_id] = {
                "retrieved_entities": [{'url': entry.url, 'title': entry.title} for entry in entries[:20]],
                "retrieval_similarities": [
                    sim.item() for sim in retrieval_simlarities[:20]
                ]
            }
        
        recall = eval_recall(top_k_wiki, ground_truth, top_ks)
        for k in top_ks:
            recalls[k] += recall[k]
    for k in top_ks:
        print("Avg Recall@{}: ".format(k), recalls[k] / (it+1))

    if kwargs["save_result_path"] != "None":
        os.makedirs(os.path.dirname(kwargs["save_result_path"]), exist_ok=True)
        print("Save retrieval result to: ", kwargs["save_result_path"])
        with open(os.path.join(kwargs["save_result_path"],'result.json'), "w") as f:
            json.dump(retrieval_result, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20,100",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument(
        "--retriever_vit", type=str, default="clip", help="clip or eva-clip"
    )
    parser.add_argument("--save_result_path", type=str, default="None", help="path to save retrieval result")
    
    args = parser.parse_args()

    test_config = {
        "sample_file_path": args.sample_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "retriever_vit": args.retriever_vit,
        "save_result_path": args.save_result_path,
    }
    print("step 1 test_config: ", test_config)
    run_test(**test_config)
