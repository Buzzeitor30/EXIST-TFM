import lightning as L

L.seed_everything(42, workers=True)
from dataset.dataset import TASK5_MAP_LABEL_TO_INT, TASK6_MAP_LABEL_TO_INT
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import os
import utils
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from callbacks import EXISTT4Writer, EXISTT5Writer, EXISTT6Writer
import evaluation
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pandas import cut

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_k_fold_splitter(df, target_col, splits=5):
    """
    Returns a fold splitter object based on the type of target column.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - target_col (str): The name of the target column.
    - splits (int): The number of splits for the fold splitter. Default is 5.

    Returns:
    - fold_splitter: The fold splitter object based on the type of target column.
    """

    if target_col.endswith("hard"):
        return get_hard_k_fold_splitter(df, target_col, splits)
    return get_k_fold_soft_splitter(df, target_col, splits)


def get_hard_k_fold_splitter(df, target_col, splits=5):
    """
    Returns a hard k-fold splitter for stratified cross-validation.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the data.
    - target_col: str
        The name of the target column in the DataFrame.
    - splits: int, optional
        The number of splits for the cross-validation. Default is 5.

    Returns:
    - splitter: generator
        A generator that yields the indices for train and test sets for each fold.
    """
    if not target_col.startswith("task6"):
        values = np.array(df[target_col].values)
        return StratifiedKFold(splits, shuffle=True, random_state=42).split(
            [0] * len(values), values
        )
    values = np.zeros((len(df), len(TASK6_MAP_LABEL_TO_INT))).tolist() # len(DF) x 7
    
    for idx, tasks in enumerate(df["task6_hard"]):
        for task in tasks:
            values[idx][TASK6_MAP_LABEL_TO_INT[task]] = 1
    mskf = MultilabelStratifiedKFold(n_splits=splits, random_state=42, shuffle=True)
    return mskf.split([0] * len(values), values)

def get_k_fold_soft_splitter(df, target_col, splits=5):
    if target_col.startswith("task4"):
        values = [x["YES"] for x in df["task4_soft"].values.tolist()]
        le = LabelEncoder()
        values = le.fit_transform(values)
        return StratifiedKFold(splits, shuffle=True, random_state=42).split(
        [0] * len(values), values
        )
    elif target_col.startswith("task5"):
        values = cut(df["task5_soft"].apply(lambda x: x["JUDGEMENTAL"]), bins=5, labels=False)
        return StratifiedKFold(splits, shuffle=True, random_state=42).split(
        [0] * len(values), values
        )

    elif target_col.startswith("task6"):
        values = np.zeros((len(df), len(TASK6_MAP_LABEL_TO_INT))).tolist() # len(DF) x 7
        for idx, tasks in enumerate(df["task6_hard"]):
            if tasks is not None:
                for task in tasks:
                    values[idx][TASK6_MAP_LABEL_TO_INT[task]] = 1
        mskf = MultilabelStratifiedKFold(n_splits=splits, random_state=42, shuffle=True)
        return mskf.split([0] * len(values), values)

    return StratifiedKFold(splits, shuffle=True, random_state=42).split(
        [0] * len(values), values
    )


def remove_unwanted_columns(df, task_id, approach):
    """
    Removes unwanted columns from a DataFrame based on the specified task ID and approach.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to remove columns.
    - task_id (int): The ID of the task.
    - approach (str): The approach used.

    Returns:
    - pandas.DataFrame: The DataFrame with unwanted columns removed.
    """
    # Nothing to remove from soft approach
    if approach == "soft":
        return df
    df.dropna(subset=[f"task{task_id}_{approach}"], axis=0, inplace=True)
    return df


def read_data(args_as_dict):
    #    Read JSON as DF
    data_df = utils.read_json_as_df(args_as_dict["dataset"])
    # Clean text
    data_df["text"] = data_df["text"].apply(lambda x: x.lower())
    if args_as_dict["clean_text"] is True:
        data_df["text"] = data_df["text"].apply(utils.clean_text)
    # Split by language
    data_df = utils.split_df_by_lang(data_df, args_as_dict["lang"])
    return data_df


def setup_pytorch_dataloader(dataset, batch_size, shuffle=True, sampler=None):
    num_workers = 8
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )


def get_default_hf_text_model_from_given_lang(arguments_from_parser):
    if arguments_from_parser["hf_text"] is not None:
        return arguments_from_parser["hf_text"]

    return {
        "es": "PlanTL-GOB-ES/roberta-base-bne",
        "en": "FacebookAI/roberta-base",
        "all": "FacebookAI/xlm-roberta-base",
    }.get(arguments_from_parser["lang"], None)


def create_early_stopping_callback():
    return EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=4)


def create_checkpoint_callback(dir_path, filename):
    return ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=dir_path,
        filename=filename,
    )


def create_prediction_writer_callback(
    output_dir, write_interval, filename, approach, task_id
):
    return {
        "4": EXISTT4Writer(output_dir, write_interval, filename, approach),
        "5": EXISTT5Writer(output_dir, write_interval, filename, approach),
        "6": EXISTT6Writer(output_dir, write_interval, filename, approach),
    }.get(task_id)


def run_cross_validation_experiment():
    # Seed everything

    # Create argument parser
    parser = utils.setup_argument_parser()
    args = parser.parse_args()
    args = utils.convert_parser_arguments_to_dict(args)
    # Huggingface models
    HF_TEXT_MODEL = get_default_hf_text_model_from_given_lang(args)
    HF_VISION_MODEL = args["hf_vision"]
    # Read data into DF, after filtering by language and labels
    data_df = read_data(args)
    data_df = remove_unwanted_columns(data_df, args["task"], args["approach"])
    # Get kfold splitter
    kfold_splitter = get_k_fold_splitter(
        data_df, target_col=f"task{args['task']}_{args['approach']}", splits=5
    )
    directory = os.path.join("logs", "T" + args["task"], args["approach"])

    for fold_id, (train_idx, val_idx) in enumerate(kfold_splitter, start=1):
        print(f"Starting fold {fold_id}")
        train_df = data_df.iloc[train_idx]
        val_df = data_df.iloc[val_idx]
        
        class_weights = None
        sampler = None
        #Class weights for task5
        #class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["task5_hard"].values), y=train_df["task5_hard"].values)
        #Sampler for task5
        #class_sample_count = (1.0 / train_df["task5_hard"].value_counts()).to_dict()
        #samples_weight =  train_df["task5_hard"].map(class_sample_count).values
        #sampler = WeightedRandomSampler(samples_weight, 2 * len(train_df), replacement=True)

        # Create dataset
        print("Creating Torch Datasets")
        # Create Pytorch Datasets
        train_dataset = utils.setup_pytorch_dataset(
            df=train_df,
            hf_text_model=HF_TEXT_MODEL,
            hf_vision_model=HF_VISION_MODEL,
            text_aug=args["text_aug"],
            image_aug_pipeline=args["vision_aug"],
            lang=args["lang"],
            approach=args["approach"],
            task_id=args["task"],
            max_length=args["max_length"],
            architecture=args["architecture"]
        )
        val_dataset = utils.setup_pytorch_dataset(
            df=val_df,
            hf_text_model=HF_TEXT_MODEL,
            hf_vision_model=HF_VISION_MODEL,
            text_aug=False,  # Validation dataset does not require text augmentation
            image_aug_pipeline=False,  # ídem
            lang=args["lang"],
            approach=args["approach"],
            task_id=args["task"],
            max_length=args["max_length"],
            architecture=args["architecture"]
            
        )
        # Create dataloaders
        print("Creating Torch Dataloaders")
        train_dataloader = setup_pytorch_dataloader(
            train_dataset, args["batch_size"], shuffle=False, sampler=sampler
        )
        val_dataloader = setup_pytorch_dataloader(
            val_dataset,
            16,
            shuffle=False,
        )
        print("Set up model")
        L_module = utils.setup_torch_lighning_module(
            args, HF_TEXT_MODEL, HF_VISION_MODEL, class_weights
        )

        if args["freeze_text"] is True:
            L_module.freeze_text()
        if args["freeze_vision"] is True:
            L_module.freeze_vision()
        # Create logger
        print("Set up loggers")
        loggers = utils.setup_loggers(directory, args["run_name"], f"fold_{fold_id}")
        # Create Trainer
        callbacks = [
            create_checkpoint_callback(
                os.path.join(directory, args["run_name"], f"fold_{fold_id}"),
                "best_model_{epoch:02d}_{val_loss:.2f}",
            ),
            create_early_stopping_callback(),
            create_prediction_writer_callback(
                output_dir=os.path.join(directory, args["run_name"], f"fold_{fold_id}"),
                write_interval="epoch",
                filename=f"{args['run_name']}_fold{fold_id}",
                approach=args["approach"],
                task_id=args["task"],
            ),
        ]

        trainer = L.Trainer(
            devices=[args["device"]],
            max_epochs=args["epochs"],
            callbacks=callbacks,
            accumulate_grad_batches=32 // args["batch_size"],
            num_sanity_val_steps=0,
            logger=loggers,
            # Only show summary for first fold
            enable_model_summary=False if fold_id > 1 else True,
        )
        print("Start training!")
        # Fit through trainer
        L.seed_everything(42, workers=True)
        trainer.fit(
            L_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        print("VALIDATE!")
        trainer.validate(
            L_module,
            dataloaders=val_dataloader,
            ckpt_path="best",
        )
        print("PREDICT!")
        trainer.predict(
            L_module, val_dataloader, ckpt_path="best", return_predictions=False
        )

    print("EVALUATE!")
    directory = os.path.join(directory, args["run_name"])
    preds_files = sorted(
        [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if f.endswith(".json")
        ]
    )
    gold_file = os.path.join(
        "data",
        "training",
        "evaluation",
        "golds",
        f"EXIST2024_training_task{args['task']}_gold",
    )
    hard_df, soft_df = evaluation.evaluate_task(
        preds_files,
        gold_file,
        args["task"],
        os.path.join(directory, "current_gold.json"),
    )
    hard_df.to_csv(
        os.path.join(directory, f"results_{args['run_name']}_hard.csv"), index=False
    )
    soft_df.to_csv(
        os.path.join(directory, f"results_{args['run_name']}_soft.csv"), index=False
    )


def main():
    # Create argument parser
    parser = utils.setup_argument_parser()
    args = parser.parse_args()
    args = utils.convert_parser_arguments_to_dict(args)
    # Huggingface models
    HF_TEXT_MODEL = get_default_hf_text_model_from_given_lang(args)
    HF_VISION_MODEL = args["hf_vision"]
    # Read data into DF, after filtering by language and labels
    data_df = read_data(args)

    # Train test split
    train_df, test_df = train_test_split(
        data_df,
        test_size=0.2,
        stratify=data_df["task4_hard"].values,
        random_state=42,
        shuffle=True,
    )
    # Create Pytorch Datasets
    train_dataset = utils.setup_pytorch_dataset(
        df=train_df,
        hf_text_model=HF_TEXT_MODEL,
        hf_vision_model=HF_VISION_MODEL,
        text_aug=args["text_aug"],
        lang=args["lang"],
        approach=args["approach"],
        architecture=args["architecture"]
    )
    test_dataset = utils.setup_pytorch_dataset(
        df=test_df,
        hf_text_model=HF_TEXT_MODEL,
        hf_vision_model=HF_VISION_MODEL,
        text_aug=False,  # Validation dataset does not require text augmentation
        lang=args["lang"],
        approach=args["approach"],
        architecture=args["architecture"]
    )

    # Create dataloaders
    train_dataloader = setup_pytorch_dataloader(train_dataset, args, shuffle=True)
    test_dataloader = setup_pytorch_dataloader(test_dataset, args, shuffle=False)
    # Create Lightning module
    L_module = utils.setup_torch_lighning_module(args, HF_TEXT_MODEL, HF_VISION_MODEL)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args["epochs"],
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=8),
        ],
    )

    trainer.fit(
        L_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )


def test():
    # Create argument parser
    parser = utils.setup_argument_parser()
    args = parser.parse_args()
    args = utils.convert_parser_arguments_to_dict(args)
    HF_TEXT_MODEL = get_default_hf_text_model_from_given_lang(args)
    HF_VISION_MODEL = args["hf_vision"]

    #Setup dataset

    df = read_data(args)
    dataset = utils.setup_pytorch_dataset(
        df=df,
        hf_text_model=HF_TEXT_MODEL,
        hf_vision_model=HF_VISION_MODEL,
        text_aug=False,
        lang=args["lang"],
        approach=args["approach"],
        task_id=args["task"],
        max_length=args["max_length"],
        architecture="test",
        image_aug_pipeline=False
    )
    test_dataloader = setup_pytorch_dataloader(dataset, args["batch_size"], shuffle=False)
    #Setup model normal

    L_model = utils.setup_torch_lighning_module(args, HF_TEXT_MODEL, HF_VISION_MODEL)
    #Create Prediction Callback
    directory = os.path.join("logs", "T" + args["task"], args["approach"])
    prediction_writer_callback = create_prediction_writer_callback(output_dir=os.path.join(directory, args["run_name"]),
                write_interval="epoch",
                filename=f"task{args["task"]}_EUA_{args['run_id']}",
                approach=args["approach"],
                task_id=args["task"])
    callbacks = [prediction_writer_callback]
    #Create Trainer
    trainer = L.Trainer(callbacks=callbacks, devices=[args["device"]])
    trainer.predict(L_model, dataloaders=test_dataloader, return_predictions=False)


if __name__ == "__main__":
    # With train-test only
    #test()
    #main()
    run_cross_validation_experiment()
