from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from models import (
    TextEncoderL,
    VisionEncoderL,
    LateFusionModelL,
    MultiModalCrossAttentionL,
)
from torchmetrics.classification import F1Score, Precision, Recall, MatthewsCorrCoef
from torchmetrics import MetricCollection
from torch import tensor

def setup_torch_lighning_module(arguments_from_parser, HF_TEXT_MODEL, HF_VISION_MODEL, loss_weights=None):
    """
    Set up a Torch Lightning module based on the provided arguments.

    Args:
        arguments_from_parser (dict): A dictionary containing the arguments from the parser.
        HF_TEXT_MODEL: The text model for the Torch Lightning module.
        HF_VISION_MODEL: The vision model for the Torch Lightning module.

    Returns:
        torch.nn.Module: The initialized Torch Lightning module.

    Raises:
        None

    """
    lightning_module_type = {
        "text": TextEncoderL,
        "image": VisionEncoderL,
        "late": LateFusionModelL,
        "multimodal": MultiModalCrossAttentionL,
    }.get(arguments_from_parser["architecture"])

    checkpoint_path = arguments_from_parser.get("checkpoint", None)


    default_params = get_model_default_params(arguments_from_parser, loss_weights)
    specific_architecture_model_parameters = get_model_params(
        arguments_from_parser["architecture"], HF_TEXT_MODEL, HF_VISION_MODEL, arguments_from_parser["projection_dim"]
    )

    if checkpoint_path is not None:
        return lightning_module_type.load_from_checkpoint(checkpoint_path, map_location="cuda:" + str(arguments_from_parser["device"]), **{**default_params, **specific_architecture_model_parameters})


    return lightning_module_type(
        **{**default_params, **specific_architecture_model_parameters}
    )


def get_model_params(architecture, HF_TEXT_MODEL, HF_VISION_MODEL, projection_dim):
    """
    Get the model parameters based on the specified architecture.

    Args:
        architecture (str): The architecture approach ("text", "image", or "late").
        HF_TEXT_MODEL: The huggingface text model for the "text" or "late" approach.
        HF_VISION_MODEL: The huggingface vision model for the "image" or "late" approach.

    Returns:
        dict: A dictionary containing the model parameters grouped by approach.
              For the "text" approach, the dictionary will have the key "hf_text_model".
              For the "image" approach, the dictionary will have the key "hf_vision_model".
              For the "late" approach, the dictionary will have the keys "hf_text_model",
              "hf_vision_model", and "projection_dim" with a value of 256.

    """
    parameters_grouped_by_approach = {
        "text": {
            "hf_text_model": HF_TEXT_MODEL,
        },
        "image": {"hf_vision_model": HF_VISION_MODEL},
        "late": {
            "hf_text_model": HF_TEXT_MODEL,
            "hf_vision_model": HF_VISION_MODEL,
            "projection_dim": projection_dim,
        },
        "multimodal": {
            "hf_text_model": HF_TEXT_MODEL,
            "hf_vision_model": HF_VISION_MODEL,
            "hf_multimodal_model": "facebook/flava-full",
        },
    }
    return parameters_grouped_by_approach.get(architecture)


def get_model_default_params(arguments_from_parser, loss_weights=None):
    """
    Get the default parameters for the model based on the given arguments.

    Args:
        arguments_from_parser (dict): A dictionary containing the arguments from the parser.

    Returns:
        dict: A dictionary containing the default parameters for the model.
    """
    task_id = arguments_from_parser["task"]
    approach = arguments_from_parser["approach"]
    loss_fn = get_loss_fn_from_task_and_approach(task_id, approach, loss_weights)
    output_neurons_for_each_classifier = get_number_of_output_neurons_for_given_task(
        task_id
    )
    metrics_collection = get_metrics_collection_from_tasks_and_approach(
        task_id, arguments_from_parser["approach"]
    )

    default_parms = {
        "learning_rate": arguments_from_parser["lr"],
        "loss_fn": loss_fn,
        "output_neurons_for_each_classifier": output_neurons_for_each_classifier,
        "metrics_collection": metrics_collection,
        "dropout_rate": arguments_from_parser["dropout"],
    }
    return default_parms


def get_loss_fn_from_task_and_approach(task_id, approach, weight=None):
    """
    Returns the loss function based on the given task ID.

    Parameters:
    - task_id (str): The ID of the task.

    Returns:
    - loss_fn: The loss function corresponding to the task ID.
    """
    if weight is not None:
        weight = tensor(weight)
    task_to_loss_fn = {
        "4": {
            "hard": BCEWithLogitsLoss(),
            "soft": BCEWithLogitsLoss(),
        },
        "5": {
            "hard": CrossEntropyLoss(weight=weight),
            "soft": CrossEntropyLoss(weight=weight),
        },
        "6": {"hard": BCEWithLogitsLoss(), "soft": BCEWithLogitsLoss()},
    }

    return task_to_loss_fn.get(task_id).get(approach)


def get_metrics_collection_from_tasks_and_approach(task_id, approach):
    """
    Returns a MetricCollection object containing the metrics based on the given task ID and approach.

    Parameters:
    - task_id (str): The ID of the task.
    - approach (str): The approach used for the task.

    Returns:
    - MetricCollection: A collection of metrics based on the task ID and approach.
    """
    metrics_to_approach_and_task = {
        "hard": {
            "4": {
                "F1": F1Score(task="binary"),
                "P": Precision(task="binary"),
                "R": Recall(task="binary"),
                "MCC": MatthewsCorrCoef(task="binary"),
            },
            "5": {
                "F1": F1Score(task="multiclass", num_classes=3, average="macro"),
                "P": Precision(task="multiclass", num_classes=3, average="macro"),
                "R": Recall(task="multiclass", num_classes=3, average="macro"),
                "MCC": MatthewsCorrCoef(task="multiclass", num_classes=3),
            },
            "6": {
                "F1": F1Score(task="multilabel", num_labels=6, average="macro"),
                "P": Precision(task="multilabel", num_labels=6, average="macro"),
                "R": Recall(task="multilabel", num_labels=6, average="macro"),
            },
        },
        "soft": {
            "4": {},
            "5": {
            },
            "6": 
            {}
        },
    }

    metrics = metrics_to_approach_and_task[approach][task_id]
    return MetricCollection(metrics)


def get_number_of_output_neurons_for_given_task(task_id):
    """
    Returns the number of output neurons for a given task ID.

    Parameters:
        task_id (str): The ID of the task.

    Returns:
        int: The number of output neurons for the given task ID.
    """
    return {"4": 1, "5": 3, "6": 6}.get(task_id)
