import csv
import json
import random
import PIL
import torch
import os
from tqdm import tqdm
from utils import get_title2wikiimg, get_train_image
from .dataset_utils import (
    reconstruct_wiki_sections_dict,
)


class IT2IT_section_RerankerDataset(torch.utils.data.Dataset):
    """
    Dataset for training to retrieve semantically similar wiki_image-section with reference_image-question.
    """
    def __init__(
        self,
        knowledge_base_file,
        train_file,
        preprocess: callable,
        neg_parrallel_process: callable,
        get_image_function=get_train_image,
        negative_db_file=None,
        wiki_img_csv_dir="../../datasets/wiki_img/full/output/",
        wiki_img_path_prefix="../../datasets/wiki_img/",
        use_hard_negative=True,
        neg_num=15,
    ):
        """Initialize the dataset.

        Args:
            knowledge_base_file (str): The path to the knowledge base file.
            train_file (str): The path to the train file.
            preprocess (callable): A callable function for preprocessing the data.
            get_image_function (callable, optional): A callable function for getting the image. Defaults to get_image.
            negative_db_file (str, optional): The path to the negative database file. Defaults to None.
            wiki_img_csv_dir (str, optional): The path to the wiki_img_csv. Defaults to None.
            wiki_img_path_prefix (str, optional): The prefix for the wiki image path. Defaults to "../../datasets/wiki_img/".
            use_hard_negative (bool, optional): Whether to use hard negative examples. Defaults to True.
            neg_num (int, optional): The number of negative examples to use. Defaults to 15.
        """
        # load the knowledge base
        with open(knowledge_base_file, "r") as f:
            self.knowledge_base = json.load(f)
        self.kb_keys = list(self.knowledge_base.keys())

        self.train_list = []
        self.dataset_name = 'E-VQA'
        with open(train_file, "r") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            for row in tqdm(reader,desc="Loading train data"):
                if (
                    row[self.header.index("question_type")] == "automatic"
                    or row[self.header.index("question_type")] == "templated"
                    or row[self.header.index("question_type")] == "multi_answer"
                    or row[self.header.index("question_type")] == "infoseek"
                ):
                    if self.dataset_name == "E-VQA" and row[self.header.index("question_type")] == "infoseek":
                        self.dataset_name = "InfoSeek"
                    if len(row[self.header.index("dataset_image_ids")].split("|")) > 1:
                        for image_id in [row[self.header.index("dataset_image_ids")].split("|")[0]]:
                            cur_row = row.copy()
                            cur_row[self.header.index("dataset_image_ids")] = image_id
                            assert cur_row[self.header.index("dataset_image_ids")] != row[self.header.index("dataset_image_ids")]
                            self.train_list.append(cur_row)
                    else:
                        self.train_list.append(row)
            print(f"Loaded {len(self.train_list)} training IQA examples.")

        self.preprocess = preprocess
        self.neg_parrallel_process = neg_parrallel_process
        self.get_image = get_image_function

        if negative_db_file is not None:
            with open(negative_db_file, "r") as f:
                self.negative_db = json.load(f)
        else:
            self.negative_db = None
            print("No negative database file provided.")

        wiki_img_csv_path_format = wiki_img_csv_dir + "wiki_image_url_part_{split_num}_processed.csv"

        self.title2wikiimg = get_title2wikiimg(wiki_img_csv_path_format, wiki_img_path_prefix)
        
        self.wiki_img_path_prefix = wiki_img_path_prefix

        self.use_hard_negative = use_hard_negative
        self.neg_num = neg_num
        
    def __len__(self):
        return len(self.train_list)


    def __getitem__(self, idx):
        example = self.train_list[idx]
        question = example[self.header.index("question")]
        positive_url = example[self.header.index("wikipedia_url")]

        #get question image
        assert len(example[self.header.index("dataset_image_ids")].split("|")) == 1
        question_image_path = self.get_image(
                example[self.header.index("dataset_image_ids")], self.dataset_name)
        question_image = self.preprocess(PIL.Image.open(question_image_path).convert("RGB"))

        #get positive image
        neg_info = self.negative_db[question_image_path.split("/")[-1].split(".")[0]]
        if neg_info['gt_entity_pos_img'] is None:
            return None
        postive_image_path = os.path.join(self.wiki_img_path_prefix, neg_info['gt_entity_pos_img'])
        
        if not os.path.exists(postive_image_path):
            print(f'positive image not found: {postive_image_path}')
            return None
        positive_image = self.preprocess(PIL.Image.open(postive_image_path).convert("RGB"))


        #get positive section and negative sections in the positive entry
        positive_entry = self.knowledge_base[positive_url]
        evidence_section_id = example[self.header.index("evidence_section_id")]
        positive_section, negative_sections = reconstruct_wiki_sections_dict(
            positive_entry, evidence_section_id
        )

        #get negative entries
        if self.use_hard_negative:
            # choose from the hard negative samples withou the positive entry
            neg_list = [entry["entry"] for entry in neg_info["top_20"]]
            if positive_url in neg_list:
                neg_list.remove(positive_url)
            negative_entry_keys = neg_list
        else:
            # choose from knowledge base without the positive entry
            negative_entry_keys = random.choices(self.kb_keys, k=20)
            if positive_url in negative_entry_keys:
                negative_entry_keys.remove(positive_url)

        negative_entries = [self.knowledge_base[key] for key in negative_entry_keys]

        #get negative images and sections

        if len(negative_sections) <= 3: 
            negative_image_paths = [postive_image_path] * len(negative_sections)
        else: # limit the number of negative sections in the positive entry to 3
            negative_image_paths = [postive_image_path] * 3
            negative_sections = negative_sections[:3]
        for it, entry in enumerate(negative_entries):
            negative_title = entry['title']
            if negative_title not in self.title2wikiimg or self.title2wikiimg[negative_title] == []:# neg entity has no wiki image, skip this example
                continue
            main_img_path = self.title2wikiimg[negative_title][0]['img_path']
  
            if not os.path.exists(main_img_path): # wiki image not found, skip it
                continue
            negative_section = reconstruct_wiki_sections_dict(entry)
            negative_image_paths.extend([main_img_path] * len(negative_section))
            negative_sections.extend(negative_section)
        assert len(negative_image_paths) == len(negative_sections)

        if len(negative_sections) < self.neg_num:
            return None
        select_index = random.sample(
            range(len(negative_sections)), self.neg_num
        )
        negative_sections = [negative_sections[i] for i in select_index]
        selected_negative_image_paths = [negative_image_paths[i] for i in select_index]
        negative_images = [PIL.Image.open(image_path).convert("RGB") for image_path in selected_negative_image_paths]
        negative_images = self.neg_parrallel_process(negative_images, self.preprocess)
        
        return (
            question_image,
            question,
            positive_image,
            positive_section,
            negative_images,
            negative_sections,
        )