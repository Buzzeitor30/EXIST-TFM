from typing import OrderedDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import os
from .keywords import ES_KEYWORDS, EN_KEYWORDS, ES_SPACY_MODEL, EN_SPACY_MODEL
import spacy
import random
import re
from tqdm import tqdm

class EXISTMemeDataset(Dataset):
    def __init__(self, df, task_id, approach, **kwargs):
        self.df = df
        self.approach = approach
        self.task_id = task_id
    
    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        label = self.get_label_from_task_and_approach(
            sample[f"task{self.task_id}_{self.approach}"]
        )
        id = sample["id_EXIST"]
        return {"label": label, "id": id}

    def get_label_from_task_and_approach(self, sample_value):
        if self.task_id == "4":
            return self.get_label_from_task4(sample_value)
        if self.task_id == "5":
            return self.get_label_from_task5(sample_value)
        return self.get_label_from_task6(sample_value)

    def get_label_from_task4(self, sample_value):
        if self.approach == "hard":
            return 1.0 if sample_value == "YES" else 0.0
        return sample_value.get("YES")

    def get_label_from_task5(self, sample_value):
        if self.approach == "hard":
            return TASK5_MAP_LABEL_TO_INT[sample_value]
        return [sample_value[x] for x in TASK5_MAP_LABEL_TO_INT.keys()]

    def get_label_from_task6(self, sample_value):
        labels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.approach == "hard":
            for x in sample_value:
                labels[TASK6_MAP_LABEL_TO_INT[x]] = 1.0
        else:
            for x in sample_value:
                labels[TASK6_MAP_LABEL_TO_INT[x]] = sample_value[x]
        return labels
    
class EXISTMemeTextDataset(EXISTMemeDataset):
    def __init__(self, df, hf_text_model_name, lang, task_id, approach, text_aug_p=0.2, max_length=256, **kwargs):
        super().__init__(df, task_id, approach)
        self.df = df
        self.text_tokenizer = AutoTokenizer.from_pretrained(hf_text_model_name)
        self.max_length = min(max_length, self.text_tokenizer.model_max_length)
        self.text_aug_p = text_aug_p
        self.nlp = spacy.load(ES_SPACY_MODEL) if lang == "es" else spacy.load(EN_SPACY_MODEL)
        self.keywords = ES_KEYWORDS if lang == "es" else EN_KEYWORDS
        self.pattern = "|".join(self.keywords)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        current_sample = self.df.iloc[index]

        text = current_sample["text"]
        image_description = current_sample.get("caption", None)

        if self.text_aug_p:
            text = self.apply_text_augmentation(text)

        tokenized_text = self.text_tokenizer(
            text=text,
            text_pair=image_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        labels = super().__getitem__(index)
        return {
            "text": tokenized_text,
            "original_text": self.text_tokenizer.decode(tokenized_text["input_ids"], skip_special_tokens=True),
            **labels,
        }
 
    def apply_text_augmentation(self, text):
        """
        Applies text augmentation to the given text.

        Args:
            text (str): The input text to be augmented.

        Returns:
            str: The augmented text.

        """
        # Tokenize the text using the spaCy model
        doc = self.nlp(text)
        # Create a list of tuples containing the original text and its lemma
        original_text_with_lemmas = [(tok.text, tok.lemma_.lower()) for tok in doc]
        # Compile the regular expression pattern from the list of identity terms
        resulting_string = []
        for word, lemma in original_text_with_lemmas:
            word_to_be_added = word
            # Check if the lemma matches the pattern
            if re.match(self.pattern, lemma):
                if random.random() < self.text_aug_p:
                    # Replace the word with the mask token
                    word_to_be_added = self.text_tokenizer.mask_token

            resulting_string.append(word_to_be_added)

        return " ".join(resulting_string)
    
class EXISTMemeImageDataset(EXISTMemeDataset):
    def __init__(self, df, hf_vision_model_name, task_id, approach, image_aug, base_image_path="data/training", **kwargs):
        super(EXISTMemeImageDataset, self).__init__(df, task_id, approach)
        self.df = df
        self.image_processor = AutoImageProcessor.from_pretrained(hf_vision_model_name, use_fast=True)
        self.image_aug = image_aug
        self.base_image_path = base_image_path
        self.load_images_in_memory()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_processed = self.image_processor(self.image_aug(self.images[index]), return_tensors="pt")
        return {**image_processed, **super().__getitem__(index)}

    def load_images_in_memory(self):
        self.images = []
        print("Loading images in memory...")
        for index in tqdm(range(len(self.df))):
            current_sample = self.df.iloc[index]
            image_path = current_sample["path_memes"]
            with Image.open(os.path.join(self.base_image_path, image_path)).convert("RGB") as img:
                self.images.append(img)

class EXISTMemeTextAndImageDataset(EXISTMemeDataset):
    def __init__(self, df, hf_vision_model_name, hf_text_model_name, lang, task_id, approach, image_aug, text_aug_p=0.2, max_length=256, base_image_path="data/training", **kwargs):
        super(EXISTMemeTextAndImageDataset, self).__init__(df, task_id, approach)
        self.df = df
        self.text_tokenizer = AutoTokenizer.from_pretrained(hf_text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(hf_vision_model_name, use_fast=True)
        self.max_length = min(max_length, self.text_tokenizer.model_max_length)
        self.text_aug_p = text_aug_p
        self.nlp = spacy.load(ES_SPACY_MODEL) if lang == "es" else spacy.load(EN_SPACY_MODEL)
        self.keywords = ES_KEYWORDS if lang == "es" else EN_KEYWORDS
        self.pattern = "|".join(self.keywords)
        self.image_aug = image_aug
        self.base_image_path = base_image_path
        self.load_images_in_memory()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        current_sample = self.df.iloc[index]

        text = current_sample["text"]
        image_description = current_sample.get("caption", None)

        if self.text_aug_p:
            text = self.apply_text_augmentation(text)

        tokenized_text = self.text_tokenizer(
            text=text,
            text_pair=image_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        labels = super().__getitem__(index)

        image_processed = self.image_processor(self.image_aug(self.images[index]), return_tensors="pt")

        return {
            "text": tokenized_text,
            "original_text": self.text_tokenizer.decode(tokenized_text["input_ids"], skip_special_tokens=True),
            **labels,
            **image_processed
        }
    
    def load_images_in_memory(self):
        self.images = []
        print("Loading images in memory...")
        for index in tqdm(range(len(self.df))):
            current_sample = self.df.iloc[index]
            image_path = current_sample["path_memes"]
            with Image.open(os.path.join(self.base_image_path, image_path)).convert("RGB") as img:
                self.images.append(img)
    def apply_text_augmentation(self, text):
        """
        Applies text augmentation to the given text.

        Args:
            text (str): The input text to be augmented.

        Returns:
            str: The augmented text.

        """
        # Tokenize the text using the spaCy model
        doc = self.nlp(text)
        # Create a list of tuples containing the original text and its lemma
        original_text_with_lemmas = [(tok.text, tok.lemma_.lower()) for tok in doc]
        # Compile the regular expression pattern from the list of identity terms
        resulting_string = []
        for word, lemma in original_text_with_lemmas:
            word_to_be_added = word
            # Check if the lemma matches the pattern
            if re.match(self.pattern, lemma):
                if random.random() < self.text_aug_p:
                    # Replace the word with the mask token
                    word_to_be_added = self.text_tokenizer.mask_token

            resulting_string.append(word_to_be_added)

        return " ".join(resulting_string)
    
TASK5_MAP_LABEL_TO_INT = OrderedDict({"NO": 0, "DIRECT": 1, "JUDGEMENTAL": 2})
TASK6_MAP_LABEL_TO_INT = OrderedDict(
    {
        "NO": 0,
        "IDEOLOGICAL-INEQUALITY": 1,
        "MISOGYNY-NON-SEXUAL-VIOLENCE": 2,
        "OBJECTIFICATION": 3,
        "SEXUAL-VIOLENCE": 4,
        "STEREOTYPING-DOMINANCE": 5,
    }
)

class EXISTMemeTestDataset(Dataset):
    def __init__(self, df, hf_text_model_name, hf_vision_model_name, max_length=256, base_image_path="data/test", **kwargs):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(hf_text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(hf_vision_model_name, use_fast=True)
        self.max_length = min(max_length, self.tokenizer.model_max_length)
        self.base_image_path = base_image_path

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        text = sample["text"]
        context = sample.get("captions", None)
        id = sample["id_EXIST"]
        text_tokenized = self.tokenizer(text=text, text_pair=context, max_length=self.max_length, padding="max_length", truncation=True)

        image_path = sample["path_memes"]
        with Image.open(os.path.join(self.base_image_path, image_path)).convert("RGB") as img:
            image_processed = self.image_processor(img, return_tensors="pt")
    

        return {
            "text": text_tokenized,
            **image_processed,
            "id": id

        }
