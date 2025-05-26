"""  

Serves as the stage 1 initial-search retriever for OMGM.

"""

import os
import torch
import tqdm
import pickle
import json
from transformers import AutoModel, AutoProcessor, CLIPTokenizer, CLIPImageProcessor
import faiss
import numpy as np
from faiss import write_index
import faiss.contrib.torch_utils
import torch


INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

class KnowledgeBase:
    """Knowledge base for OMGM system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None

    def load_knowledge_base(self):
        """Load the knowledge base."""
        raise NotImplementedError


class WikipediaKnowledgeBase(KnowledgeBase):
    """Knowledge base for OMGM."""

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []

    def load_knowledge_base_full(
        self, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base from multiple score files.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The parent folder path to the vision similarity scores to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None

        if visual_attr is not None:
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob

            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError(
                        "Image scores not found, which should be a url or path to a pickle file."
                    )
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base

    def load_knowledge_base(self, image_dict=None, scores_path=None, visual_attr=None):
        """Load the knowledge base.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if (
            scores_path is not None
        ):  # TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError(
                    "Image scores not found, which should be a url or path to a pickle file."
                )
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base


class WikipediaKnowledgeBaseEntry:
    """Knowledge base entry for OMGM.

    Returns:
    """

    def __init__(self, entry_dict, visual_attr=None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict: The dictionary containing the knowledge base entry.
            visual_attr: The visual attribute. Deprecated in the current version.

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr


class Retriever:
    """Retriever parent class for OMGM."""

    def __init__(self, model=None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
        """
        self.model = model

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.
        """
        raise NotImplementedError


class ClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT."""

    def __init__(self, model="clip", device="cpu"):
        """Initialize the ClipRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
        """
        super().__init__(model)
        self.model_type = model
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to("cuda").eval()
            self.processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            self.model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to("cuda").eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip-full":
            self.model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model.to("cuda").eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device
        self.model.to(device)
        self.knowledge_base = None
        

    def load_knowledge_base(
        self, knowledge_base_path, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        knowledge_base_list = self.knowledge_base.load_knowledge_base(
            image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
        )
        return knowledge_base_list
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file

    def save_knowledge_base_faiss(
        self,
        knowledge_base_path,
        image_dict=None,
        scores_path=None,
        visual_attr=None,
        save_path=None,
    ):
        """Save the knowledge base with faiss index.

        Args:
            knowledge_base_path: The knowledge base to load.
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
            save_path: The path to save the faiss index.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)


    def save_faiss_index(self, save_index_path):
        """Save the faiss index.
        
        Args:
            save_index_path: The path to save the faiss index.
        """
        if save_index_path is not None:
            write_index(self.faiss_index, save_index_path + "kb_index.faiss")
            with open(os.path.join(save_index_path, "kb_index_ids.pkl"), "wb") as f:
                pickle.dump(self.faiss_index_ids, f)




    def load_entity_faiss_index(self, load_index_path):
        """Load the summary faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            print('Loading index...')
            self.entity_faiss_index = faiss.read_index(load_index_path)
            res = faiss.StandardGpuResources()
            self.entity_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.entity_faiss_index)
            print("Faiss index loaded with {} entries.".format(self.entity_faiss_index.ntotal))
        return

    def prepare_faiss_index(self):
        """Prepare the faiss index from scores in the knowledge base."""
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [
            score for entry in self.knowledge_base for score in entry.score.values()
        ]
        score_ids = [
            i
            for i in range(len(self.knowledge_base))
            for j in range(len(self.knowledge_base[i].score))
        ]
        
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print("Faiss index built with {} entries.".format(index.ntotal))

        return

    
    @torch.no_grad()
    def I2T_faiss(
        self, image, top_k=100, return_entry_list=False
    ):
        """Retrieve the top K similar summary from the knowledge base using faiss with query image.

        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        if self.model_type == "clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip" or self.model_type == "eva-clip-full":
            # EVA-CLIP Process the input image
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.encode_image(inputs)
        assert self.entity_faiss_index is not None
        query = image_score.float()
        query = torch.nn.functional.normalize(query)

        D, I = self.entity_faiss_index.search(query, top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            index_id = I[0][i]
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[index_id])
            else:
                # find the knowledge base entry index
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[index_id].url,
                        "knowledge_base_index": int(index_id),
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[index_id],
                    }
                )
        return top_k_entries

