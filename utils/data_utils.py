import pandas as pd
import torch
from dataset.dataset import EXISTMemeTestDataset, EXISTMemeTextDataset, EXISTMemeImageDataset, EXISTMemeTextAndImageDataset
import demoji
import albumentations as A
import torchvision.transforms as transforms


def read_json_as_df(json_path: str) -> pd.DataFrame:
    """
    Read a JSON file and return its contents as a pandas DataFrame.

    Parameters:
        json_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: The JSON data as a pandas DataFrame.
    """
    return pd.read_json(json_path, orient="index")


def split_df_by_lang(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Splits a DataFrame based on the specified language.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        lang (str): The language to filter the DataFrame by. Use "all" to return the original DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame based on the specified language.
    """
    if lang == "all":
        return df
    return df[df["lang"] == lang]


def collate_fn_text(batch):
    """
    Collate function for text data.

    Args:
        batch (list): A list of dictionaries representing the batch of data.

    Returns:
        dict: A dictionary containing the collated batch of data.
    """
    return {
        k: torch.tensor([x["text"][k] for x in batch], dtype=torch.long)
        for k in batch[0]["text"].keys()
    }


def collate_fn(batch):
    """
    Collates a batch of data samples into a single batch.

    Args:
        batch (list): A list of data samples, where each sample is a dictionary.

    Returns:
        dict: A dictionary containing the collated batch, with the following keys:
            - "id": A tensor containing the IDs of the samples in the batch.
            - "text": The collated tokenized text data from the batch.
            - "pixel_values": The collated image data from the batch.
            - "task4": A tensor containing the task4 values from the batch.
            - "task5": A tensor containing the task5 values from the batch.
            - "task6": A tensor containing the task6 values from the batch.
            - "task6_mask" (optional): A tensor containing the task6 mask values from the batch, if present.
            - "original_text": A list containing the original text data from the batch.
    """

    res_batch = {}
    res_batch["id"] = torch.tensor([x["id"] for x in batch])
    if "label" in batch[0]:
        res_batch["label"] = torch.tensor([x["label"] for x in batch])
    if "text" in batch[0]:
        res_batch["text"] = collate_fn_text(batch)


    if "pixel_values" in batch[0]:
        aux = torch.cat([x["pixel_values"] for x in batch], dim=0)
        res_batch["image"] = {"pixel_values": aux}
    # res_batch["original_text"] = [x["original_text"] for x in batch]
    return res_batch


def get_vision_augmentation_pipeline(image_aug):
    """
    Returns a composition of image augmentation transformations.

    Parameters:
    image_aug (bool): Flag indicating whether data augmentation is needed for the train dataset.

    Returns:
    albumentations.Compose: Composition of image augmentation transformations.
    """
    transformations = []
    if image_aug is True:  # We need data augmentation for train dataset
        transformations = [transforms.RandAugment(5, 6),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.RandomVerticalFlip(p=0.5),
                           #transforms.RandomRotation(degrees=45)
                           ]

    # transformations = transformations + [
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]

    return transforms.Compose(transformations)


def setup_pytorch_dataset(
    df: pd.DataFrame,
    hf_text_model: str,
    hf_vision_model: str,
    text_aug: bool,
    image_aug_pipeline: bool,
    lang: str,
    approach: str,
    task_id: str,
    max_length: int,
    architecture: str,
):
    """
    Set up a PyTorch dataset for EXISTMeme.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        hf_text_model (str): The name of the Hugging Face text model.
        hf_vision_model (str): The name of the Hugging Face vision model.
        text_aug (bool): Flag indicating whether to apply text augmentation.
        image_aug_pipeline (bool): Flag indicating whether to apply image augmentation pipeline.
        lang (str): The language of the dataset.
        approach (str): The approach used for the dataset.

    Returns:
        EXISTMemeDataset: The PyTorch dataset for EXISTMeme.
    """

    if architecture == "text":
        dataset_cls = EXISTMemeTextDataset
    elif architecture == "image":
        dataset_cls = EXISTMemeImageDataset
    elif architecture == "early":
        dataset_cls = EXISTMemeTextAndImageDataset
    else:
        dataset_cls = EXISTMemeTestDataset 

    params = {"df":df, "hf_text_model_name":hf_text_model, "hf_vision_model_name":hf_vision_model, "lang":lang, "approach":approach, "task_id":task_id, "text_aug":text_aug, "image_aug":get_vision_augmentation_pipeline(image_aug_pipeline), "max_length":max_length}

    return dataset_cls(
        **params,
    )


def clean_text(text):
    """
    Cleans the given text by removing URLs, usernames, emojis, and hashtag symbols.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    text = text.lower()

    def remove_urls(text):
        """
        Removes URLs from the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with URLs removed.
        """
        # Regular expression to remove URLs
        url_pattern = re.compile(
            r"\b(?:(?:https?://|www\.)\S+|(?!https?://|www\.)\b\w+\.[a-zA-Z]{2,}\b(?:/[^\s]*)?)\b",
            flags=re.IGNORECASE,
        )
        return url_pattern.sub(r"", text)

    import re

    def remove_usernames(text):
        """
        Removes usernames from the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with usernames removed.
        """
        # Regular expression pattern to remove social media usernames
        username_pattern = re.compile(r"@\w+", flags=re.IGNORECASE)
        return username_pattern.sub(r"", text)

    def remove_emojis(text):
        """
        Removes emojis from the given text.

        Parameters:
        text (str): The text from which emojis need to be removed.

        Returns:
        str: The text with emojis removed.
        """
        return demoji.replace(text, "")

    import re

    def remove_hashtag_symbol(text):
        """
        Removes the hashtag symbol from the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with the hashtag symbol removed.
        """
        pattern = r"#"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = remove_urls(text)
    text = remove_usernames(text)
    text = remove_emojis(text)
    text = remove_hashtag_symbol(text)

    return text
