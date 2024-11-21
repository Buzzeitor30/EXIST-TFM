import argparse


def setup_argument_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Main file to launch experimentation from. Includes a bunch of options",
    )
    # REQUIRED ARGUMENTS

    parser.add_argument(
        "run_name",
        type=str,
        help="Specify the run name in order to store the information.",
    )
    parser.add_argument(
        "lang",
        type=str,
        choices=["es", "en", "all"],
        help="Select in which language to train the model on. You can also train on the whole corpus. Available options are 'es' for spanish, 'en' for english and 'all' for both languages",
    )

    parser.add_argument(
        "architecture",
        type=str,
        choices=["text", "image", "multimodal", "late"],
        help="Select the architecture to use for the run. Available options are text(unimodal), image(unimodal), multimodal(using cross attention) & late fusion(concat embeddings)",
    )

    parser.add_argument(
        "task",
        type=str,
        choices=["4", "5", "6"],
        help="Select in which tasks to train the model. Default values are for task *4*, *5* and *6*.",
    )

    parser.add_argument(
        "approach",
        type=str,
        choices=["hard", "soft", "annotators"],
        help="Select the approach to train the models on. Available options are on hard labels and soft labels. Annotator has yet to be implemented",
    )
    # MODEL SETTINGS ARGUMENTS
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Select the learning rate for the model",
    )

    parser.add_argument(
        "--hf-text",
        type=str,
        help="HuggingFace text model to use. Default options are a roBERTa-base architecture for spanish, english and both languages",
        default=None,
    )

    parser.add_argument(
        "--hf-vision",
        type=str,
        help="HuggingFace ViT to use. Default option is ViT base patch 16",
        default="google/vit-base-patch16-224",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Load a lightning module from a checkpoint",
    )
    # DATASET SETTINGS
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the JSON file containing the dataset",
        default="./data/training/EXIST2024_training_with_gold.json",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for the Python DataLoader",
        default=8,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate for the model",
        default=0.25,
    )

    parser.add_argument(
        "--clean-text",
        help="Clean the text data",
        action="store_true",
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        help="Projection dimension for the multimodal model",
        default=512,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length of the text data",
    )
    parser.add_argument(
        "--text-aug",
        action="store_true",
        help="Whether to apply text data augmentation by randomly masking a subset of words",
    )
    parser.add_argument(
        "--vision-aug", action="store_true", help="Apply Image data augmentation"
    )
    parser.add_argument(
        "--epochs", type=int, help="Maximum number of epochs", default=10
    )
    parser.add_argument(
        "--freeze-text",
        action="store_true",
        help="Freeze the weights of the Transformer Text Encoder",
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        help="Freeze the weights of the Transformer Vision Encoder",
    )

    parser.add_argument(
        "--device",
        type=int,
        help="Select the device to train the model on. Default is the first available GPU",
        default=0,
    )

    parser.add_argument(
        "--run-id",
        type=int,
        help="Select the run id to store the information. Default is 0",
        default=0,
    )
    return parser


def convert_parser_arguments_to_dict(arguments_from_parser):
    return vars(arguments_from_parser)
