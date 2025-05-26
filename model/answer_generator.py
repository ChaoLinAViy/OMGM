""" 

Serves as the stage 3 generator for OMGM. 

"""

import torch
from .retriever import WikipediaKnowledgeBaseEntry
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# from huggingface_hub import login
# login(token=os.getenv("HF_TOKEN"))


VLM_INFOSEEK_INSTRUCTION = ("Answer the encyclopedic question about the given image." 
                    +"Don't mention the visuall content of image in your output. "
                    +"Directly output the answer of the question according to the Context.\n"
                    +"If you need to answer questions about numbers or time, please output the corresponding numerical format directly.\n"
                    +"If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.\n "
                    +"There is an example:\n- Context: # Wiki Article: Dolomites\n## Section Title: Dolomites\nThe Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno, Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone.\n"
                    +"- Question: Which city or region does this mountain locate in?\n"
                    +"Just answer the questions , no explanations needed. Short answer is: Province of Belluno\n\n")

VLM_INFOSEEK_VANILLA_INSTRUCTION = ("Answer the encyclopedic question about the given image." 
                    +"Don't mention the visuall content of image in your output. "
                    +"Directly output the answer of the question.\n"
                    +"If you need to answer questions about numbers or time, please output the corresponding numerical format directly.\n")

VLM_EVQA_INSTRUCTION = ("Answer the encyclopedic question about the given image." 
                    +"Don't mention the visuall content of image in your output. "
                    +"Directly output the answer of the question according to the Context."
                    +"If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.\n ")


VLM_EVQA_VANILLA_INSTRUCTION = ("Answer the encyclopedic question about the given image." 
                    +"Don't mention the visuall content of image in your output. ")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def reconstruct_wiki_article(knowledge_entry: WikipediaKnowledgeBaseEntry):
    """Reconstruct the wiki article from the knowledge entry class."""
    title = knowledge_entry.title
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if (
            "external link" in section_title.lower()
            or "reference" in section_title.lower()
        ):
            continue
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return article


def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry class."""
    title = knowledge_entry.title
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if it == int(section_index):
            evidence_section = (
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
        elif (
            "external links" in section_title.lower()
            or "references" in section_title.lower()
        ):
            continue
        else:
            sections.append(
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
    if section_index != -1:
        return evidence_section, sections
    return sections


def get_all_sections(knowledge_entry):
    """Get all sections in list format."""
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        sections.append(
            "* Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return sections


pseudo_tokenizer = None


def _adjust_prompt_length(prompt, desired_token_length):
    """Adjust the prompt length to the desired token length."""
    global pseudo_tokenizer

    if pseudo_tokenizer is None:
        pseudo_tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b"
        )

    # Tokenize the prompt
    tokens = pseudo_tokenizer.encode(prompt)

    if len(tokens) > desired_token_length:
        # If the prompt is too long, trim it
        trimmed_tokens = tokens[:desired_token_length]
        # Convert tokens back to text
        trimmed_text = pseudo_tokenizer.decode(
            trimmed_tokens, clean_up_tokenization_spaces=True
        )[4:]
        return trimmed_text
    else:
        # If the length is already as desired
        return prompt


class AnswerGenerator:
    """Question generator for OMGM."""

    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        """Load the model.

        Args:
            model_name: The model to load.
        """
        raise NotImplementedError

def prompt_constructor(question, dataset_name, entry = None, entry_section = None):

    if entry is not None:
        # setting for article-based VQA (entity level)
        context = reconstruct_wiki_article(entry)
        context = _adjust_prompt_length(context, 3072)
        if dataset_name == "infoseek":
            prompt = (
                "- Context: "
                + context
                + "\n- Question: "
                + question
                + "\nJust answer the questions , no explanations needed. Short answer is:"
            )
        else:
            prompt = (
                "- Context: "
                + context
                + "\n- Question: "
                + question
                + "\nThe answer is:"
            ) #e-vqa setting
    elif entry_section is not None:
        #setting for section-based VQA
        entry_section = _adjust_prompt_length(entry_section, 3072)
        if dataset_name == "infoseek":
            prompt = (
                "- Context: "
                + entry_section
                + "\n- Question: "
                + question
                + "\nJust answer the questions , no explanations needed. Short answer is:"
            )
        else:
            prompt = (
                "- Context: "
                + entry_section
                + "\n- Question: "
                + question
                + "\nThe answer is:"
            ) #e-vqa setting
    else:
        #setting for vanilla VQA
        prompt = "- Question: " + question + "\nThe answer is:"

    return prompt

def Get_dataset_based_messages(dataset_name, prompt):
    
    if dataset_name == 'infoseek':
        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for answering encyclopedic questions. Do not answer anything else."
                        +"If you need to answer questions about numbers or time, please output the corresponding numerical format directly."
                        +"If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge."
                    },
                {"role": "user", "content":  "- Context: # Wiki Article: Dolomites\n## Section Title: Dolomites\nThe Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno, Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone."
                        +"\n- Question: Which city or region does this mountain locate in?"
                        +"\nJust answer the questions , no explanations needed. Short answer is: Province of Belluno\n\n"
                        +prompt}
        ]        
    else:
        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for answering encyclopedic questions."
                    +"If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge."
                },
                {"role": "user", "content": prompt},
        ]
    return messages
        


class MistralAnswerGenerator(AnswerGenerator):
    """Mistral Question generator for OMGM."""

    def __init__(self, device, model_path, use_embedding_model=False):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
            use_embedding_model: Whether to use the embedding model. Deprecated in the current version.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()
        

    def _load_model(self):
        """Load the model."""
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).eval()
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        vanilla_input,
        dataset_name,
        entry=None,
        entry_section=None
    ):
        """Mistral Answering for a given entry

        Args:
            vanilla_input: vanilla input to the model.
            dataset_name: The image dataset name.
            entry: The entry to answer the question for.
            entry_section: The entry section to answer the question for.
            
        """
        question = vanilla_input["question"]
        prompt = prompt_constructor(question, dataset_name, entry, entry_section)
            
        messages = Get_dataset_based_messages(dataset_name, prompt)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        encodeds = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", max_length=8000, truncation=True
        )
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=2,
        )
        response = self.tokenizer.decode(generated_ids[0][model_inputs.shape[1] :], skip_special_tokens=True)

        return response
    
   


class LLaMA3AnswerGenerator(AnswerGenerator):
    def __init__(self, device, model_path):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()


    def _load_model(self):
        """Load the model.

        Args:
            model_path: The model to load.
        """
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)#, padding_side='left'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).eval()

        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        vanilla_input,
        dataset_name,
        entry=None,
        entry_section=None
    ):
        """LLaMA3 Answering for a given entry

        Args:
            vanilla_input: vanilla input to the model.
            dataset_name: The image dataset name.
            entry: The entry to answer the question for.
            entry_section: The entry section to answer the question for.
            
        """
        question = vanilla_input["question"]
        prompt = prompt_constructor(question, dataset_name, entry, entry_section)
            
        messages = Get_dataset_based_messages(dataset_name, prompt)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        encodeds = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", max_length=8000, truncation=True
        )
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            # temperature=None,
            # top_p=None
        )
        response = self.tokenizer.decode(generated_ids[0][model_inputs.shape[1] :], skip_special_tokens=True)
        return response
    

def Get_VLM_dataset_based_messages(dataset_name, prompt):

    
    if dataset_name == "infoseek":
        messages = VLM_INFOSEEK_INSTRUCTION + prompt
    else:
        messages = VLM_EVQA_INSTRUCTION + prompt
    return messages

class InterVL2_5AnswerGenerator(AnswerGenerator):
    def __init__(self, device, model_path):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model.

        Args:
            model_path: The model to load.
        """
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        vanilla_input,
        dataset_name,
        entry=None,
        entry_section=None
    ):
        """InterVL2_5 Answering for a given entry

        Args:
            vanilla_input: vanilla input to the model.
            dataset_name: The image dataset name.
            entry: The entry to answer the question for.
            entry_section: The entry section to answer the question for.
            
        """
        question = vanilla_input["question"]
        image_path = vanilla_input["image_path"]
        prompt = prompt_constructor(question, dataset_name, entry, entry_section)
            
        messages = Get_VLM_dataset_based_messages(dataset_name, prompt)

        messages = "<image>\n" + messages
        
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
        response = self.model.chat(self.tokenizer, pixel_values, messages, dict(max_new_tokens=128, do_sample=False))
        return response

class LlaVA1_5AnswerGenerator(AnswerGenerator):
    def __init__(self, device, model_path):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model.

        Args:
            model_path: The model to load.
        """
        disable_torch_init()
        if self.model_path == 'llava-hf/llava-1.5-7b-hf':
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True).eval()
        else:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import (
                get_model_name_from_path,
            )
            model_name = get_model_name_from_path(self.model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                self.model_path, None, model_name
            )
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        vanilla_input,
        dataset_name,
        entry=None,
        entry_section=None
    ):
        """LlaVA1_5 Answering for a given entry

        Args:
            vanilla_input: vanilla input to the model.
            dataset_name: The image dataset name.
            entry: The entry to answer the question for.
            entry_section: The entry section to answer the question for.
            
        """
        question = vanilla_input["question"]
        image_path = vanilla_input["image_path"]
        raw_image = Image.open(image_path).convert("RGB")

        prompt = prompt_constructor(question, dataset_name, entry, entry_section)

        messages = Get_VLM_dataset_based_messages(dataset_name, prompt)
        if self.model_path == 'llava-hf/llava-1.5-7b-hf':
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": messages},
                    {"type": "image"},
                    ],
                },
            ]
            input_format = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs =self.processor(images=raw_image, text=input_format, return_tensors='pt').to(self.device, torch.float16)
                
            
            
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            response = self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        else:
            from llava.constants import  IMAGE_TOKEN_INDEX
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
            )
            from llava.conversation import conv_templates

            # conv = conv_templates["llava_v0"].copy()
            # conv.append_message(conv.roles[0], "qs")
            # conv.append_message(conv.roles[1], None)
            # messages = conv.get_prompt()
            
            
            image_sizes = [raw_image.size]
            images_tensor = process_images(
                [raw_image],
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)
            input_ids = (
                tokenizer_image_token(messages, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample= False,
                    max_new_tokens=200,
                    use_cache=True,
                )
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        return response


class BGESectionReranker:
    """BGE Section Reranker"""

    def __init__(self, model_path, device):
        """Initialize the Text Reranker"""
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model."""
        print("Loading BGE Section Reranker")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print("BGE Section Reranker Loaded")

    @torch.no_grad()
    def rank_entry_sections(self, question, sections):
        pairs = [[question, section] for section in sections]
        pair_step = 4
        for pair_spilit in range(0, len(pairs), pair_step):
            inputs = self.tokenizer(
                pairs[pair_spilit : pair_spilit + pair_step], padding=True, truncation=True, return_tensors="pt", max_length=6000
            ).to(self.device)
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            if pair_spilit == 0:
                all_scores = scores
            else:
                all_scores = torch.cat([all_scores, scores], dim=0)
        _, index = torch.sort(all_scores, descending=True)

        return torch.sigmoid(all_scores), index
