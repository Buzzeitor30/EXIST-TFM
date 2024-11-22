# EXIST-TFM
The following repository contains the source code for the proposals of the sexism detection tasks in memes presented on [EXIST 2024](https://nlp.uned.es/exist2024/).

The aim of this project is to investigate how the Learning With Disagreements (LeWiDi) paradigm influences a classifier performance on detecting sexism hate speech as well as performing a fine-grained classification of sexist content.

A deeper analysis can be found on Elias Urios Alacreu final master thesis [*Identification and categorization of racial stereotypes in texts and of sexist memes: An approach based on learning by disagreement*](https://riunet.upv.es/handle/10251/209997).

## Project structure
```
EXIST-TFM/
├── data/                    # Data files and processing scripts
│   └── training/           # Training data and analysis notebooks
│   └── test/               # Test data
├── logos/                   # Project logos
├── logs/                   # Results and model outputs
├── loss_fn/                # Custom loss function implementation
├── dataset/                 # Dataset class and keywords file
├── notebooks/              # Jupyter notebooks for data preparation
├── models/                 # Python files with models
├── utils/                  # CLI, data, logger and model utils
├── main.py                 # Main training script
├── callbacks.py            # Callbacks for torch lightning
├── exist2024_format_val.py # Check correct submission format
├── evaluation.py           # Script for performing evaluation
└── environment.yml         # Conda environment file

```

## How to run
### Prepare data
#### Add gold labels and silver labels into a single JSON file
The original EXIST dataset only provides the individual annotations of each annotator for each sample. However, both gold and silver labels can be found in separate files. In order to handle more easily the dataset files, we have decided to include extra columns with  the format *task{4, 5, 6}_{hard, soft}*. 

This new dataset format can be obtained through the scripts on [this notebook](data/training/data_analysis.ipynb)
#### Add captions
The following techniques are highly recommended to enhance the text encoder 's performance.
##### Generate captions
In order to generate captions from the given memes, please run the following [Jupyter notebook](notebooks/llava2_captions_generator.ipynb). Note that the latter notebook was prepared to run on a Kaggle environment, in which the available GPUs are more powerful. Note that this notebook will generate a text file with the corresponding caption for each meme. 

##### Translate captions and add them into the dataset
Since some memes are provided on Spanish, we will translate the corresponding generated caption on English to Spanish using a Transformer and Huggingface pipelines. This process can be found inside the following [notebook](notebooks/data_analysis.ipynb)

### Launch script
As mentioned previously, the following project presents three different architectures (text-only, image-only, multimodal) approaches for each one of the tasks.

```python main.py RUN_NAME LANG ARCHITECTURE TASK APPROACH```

Where each one of the following mandatory parameters has the following meaning:
- `RUN_NAME` is the name for this experiment.
- `LANG` represents source language for the memes, with values 'es', 'en' or 'all' (Spanish + English). This parameter not only decides on which partition of the dataset to train, but also the default text encoder to be used (monolingual or multilingual).
- `ARCHITECTURE`, which decides on whether to employ an architecture for 'text' or 'image' only, or an early fusion architecture which combines both of them ('early').
- `TASK` which represents the task to tackle ('4', '5' or '6'). 
- `APPROACH`, which decides whether to train our models with 'hard' labels (classic approach) or train them using 'soft' labels (following the LeWiDi paradigm)

Where RUN_NAME is the name of the current execution, LANG is an option among 'es' (spanish), 'en' (english) or 'all' (english + spanish)


It also includes the following optional flags:
- `--lr ` allows to specify the learning rate (float).
- `--hf-text` allows to specify the HuggingFace Transformer text encoder to use.
- `--hf-vision` allows to specify the Huggingface Vision Transformer encoder to use.
- `--checkpoint` load a lightning module from a checkpoint.
- `--dataset` to specify the JSON file containing the dataset. This parameter is **very important** if you have various JSON files, where one of them does not contain captions and other does, for example.
- `--batch-size` to specify the batch size in which the dataloader delivers the data. NOTE: Batch size is always 16, if a smaller value is specified we will be accumulating gradient for the corresponding number of steps.
- `--dropout` to specify the dropout ratio.
- `--clean-text` to preprocess the text of the meme.
- `--projection-dim` dimension to project the \[CLS\] tokens of the Transformers.
- `--max-length`, which specifies the maximum length of the input text.
- `--text-aug` in order to randomly mask identity terms during training.
- `--vision-aug` to apply image data augmentation.
- `--epochs` is the maximum number of epochs to train the model for.
- `--freeze-text` freezes the text encoder weights during training.
- `--freeze-vision` freezes the vision encoder weights during training.
- `--device` specifies the CUDA device to train the model on.

### Check results

The project automatically manages results logging with the following folder structure:

* Creates a `logs` folder if it doesn't exist
* Inside `logs`, creates a subfolder for the specific task (e.g., Task4, Task5, Task6)
* Within the task subfolder, creates another subfolder for the given approach (hard/soft labels)
* Generates a new folder with the run name inside this approach subfolder
* The run folder contains:
  - Predictions for each fold
  - Summary CSV files with evaluation metrics

Example structure:
### Examples
#### Text only (ES) trained with hard label approach (TASK 4)
`python main.py TextESTask4Hard es text 4 hard` 
#### Image only (ES + EN) trained with soft label approach (TASK 5)
`python main.py ImageALLTask5Soft all image 5 soft`
#### Multimodal (EN) trained with soft label approach (TASK 6)
`python main.py EarlyFusionENTask6Soft en early 6 soft`

## Results on test set
### Task 4
#### Hard evaluation
| Architecture                                   | Approach | Ranking | ICM ↑      | ICM Norm ↑ | F1 - Sexist ↑ |
|-----------------------------------------------|-------|---------|------------|------------|---------------|
| Text + Caption                                   | Hard  | 14      | 0.0868     | 0.5441     | 0.7288        |
|                                               | Soft  | 13      | 0.0880     | 0.5448     | 0.6972        |
| Text + Caption + Tweets                          | Hard  | 29      | -0.0932    | 0.4526     | 0.6842        |
|                                               | Soft  | 8       | 0.1044     | 0.5531     | 0.7155        |
| Image                                         | Hard  | 43      | -0.3120    | 0.3414     | 0.6770        |
|                                               | Soft  | 45      | -0.3592    | 0.3173     | 0.6402        |
| Early Fusion + Caption                                        | **Hard**  | **4**       | **0.1657** | **0.5843** | **0.7358**    |
|                                               | Soft  | 34      | -0.1645    | 0.4164     | 0.6518        |
| **Gold Baseline**                             | -     | 0       | 0.9832     | 1.0000     | 1.0000        |
| **Winners**                                   | -     | 1       | 0.3182     | 0.6618     | 0.7642        |

#### Soft evaluation
| Architecture                                   | Approach | Ranking | ICM Soft ↑  | ICM Soft Norm ↑ | Cross Entropy ↓ |
|-----------------------------------------------|-------|---------|-------------|-----------------|-----------------|
| Text + Captions                                   | Hard  | 2       | -0.2006     | 0.4678          | 0.9693          |
|                                               | Soft  | 25      | -0.6790     | 0.3909          | 0.9254          |
| Text + Captions + Tweets                          | Hard  | 17      | -0.5461     | 0.4122          | 1.0767          |
|                                               | Soft  | 11      | -0.4301     | 0.4309          | **0.9180**      |
| Image                                         | Hard  | 27      | -0.9466     | 0.3478          | 1.0333          |
|                                               | Soft  | 34      | -1.1603     | 0.3135          | 1.0154          |
| Early Fusion + Captions                                        | **Hard**  | **1**      | **-0.1182** | **0.4810**      | 1.0810          |
|                                               | Soft  | 26      | -0.8690     | 0.3603          | 0.9795          |
| **Gold Baseline**                             | -     | 0       | 3.1107      | 1.0000          | 0.5852          |
| **Winners**                                 | -     | 1       | -0.2925     | 0.4530          | 1.1028          |


### Task 5
#### Hard evaluation
| Architecture     | Approach | Ranking | ICM ↑         | ICM Norm ↑     | F1 - Sexist ↑    |
|------------------|-------|---------|---------------|----------------|------------------|
| Text + Captions      | Hard  | 6       | -0.2716       | 0.4056         | 0.3820           |
|                  | **Soft**  | **1**       | **-0.2067**   | **0.4281**     | 0.3997           |
| Image            | Hard  | 16      | -0.6535       | 0.2728         | 0.2941           |
|                  | Soft  | 20      | -0.7523       | 0.2385         | 0.3145           |
| Early Fusion + Captions          | Hard  | 13      | -0.3598       | 0.3749         | 0.3772           |
|                  | Soft  | 2       | -0.2373       | 0.4175         | **0.4106**       |
| **Gold Baseline**| -     | 0       | 1.4383        | 1.0000         | 1.0000           |
| **Winners**    | -     | 1       | -0.2397       | 0.4167         | 0.3873           |

#### Soft evaluation
| Architecture     | Approach | Ranking | ICM Soft ↑    | ICM Soft Norm ↑ | Cross Entropy ↓  |
|------------------|-------|---------|---------------|------------------|------------------|
| Text + Captions      | **Hard**  | **3**       | **-1.3231**   | **0.3593**       | 1.6018           |
|                  | Soft  | 8       | -1.6195       | 0.3278           | 1.4493           |
| Image            | Hard  | 10      | -1.9688       | 0.2906           | 1.5648           |
|                  | Soft  | 13      | -2.0116       | 0.2861           | 1.5118           |
| Early Fusion + Captions         | Hard  | 9       | -1.6200       | 0.3277           | 1.5203           |
|                  | Soft  | 5       | -1.3766       | 0.3536           | **1.4342**       |
| **Gold Baseline**| -     | 0       | 4.7018        | 1.0000           | 0.9325           |
| **Winners**    | -     | 1       | -1.2453       | 0.3676           | 1.6235           |

### Task 6
#### Hard evaluation
| Architecture                  | Approach | Ranking | ICM ↑           | ICM Norm ↑       | F1 - Sexist ↑    |
|-------------------------------|-------|---------|------------------|------------------|------------------|
| Text + Captions                  | Hard  | **2**       | **-0.7826**      | **0.3376**       | 0.4018           |
|                               | Soft  | 5       | -0.8531          | 0.3230           | 0.3800           |
| Text + Captions + Tweets          | Hard  | 8       | -1.0572          | 0.2807           | 0.3866           |
|                               | Soft  | 3       | -0.8104          | 0.3319           | **0.4337**       |
| Image                         | Hard  | 20      | -1.6473          | 0.1582           | 0.2215           |
|                               | Soft  | 21      | -1.6516          | 0.1573           | 0.2015           |
| Early Fusion + Captions                       | Hard  | 11      | -1.2121          | 0.2485           | 0.2891           |
|                               | Soft  | 12      | -1.2697          | 0.2366           | 0.3158           |
| **Gold Baseline**             | -     | 0       | 2.4100           | 1.0000           | 1.0000           |
| **Winners**                 | -     | 1       | -0.6996          | 0.3549           | 0.4319           |

#### Soft evaluation
| Architecture                  | Approach | Ranking | ICM Soft ↑      | ICM Soft Norm ↑  |
|-------------------------------|-------|---------|-----------------|------------------|
| Text + Captions                  | Hard  | 9       | -5.7370         | 0.1960           |
|                               | Soft  | 2       | -4.6089         | 0.2557           |
| Text + Captions + Tweets          | Hard  | 20      | -8.0799         | 0.0718           |
|                               | Soft  | **1**       | **-4.3100**     | **0.2716**       |
| Image                         | Hard  | 11      | -6.4111         | 0.1602           |
|                               | Soft  | 14      | -6.5186         | 0.1545           |
| Early Fusion + Caption                       | Hard  | 7       | -5.4716         | 0.2100           |
|                               | Soft  | 8       | -5.5504         | 0.2058           |
| **Gold Baseline**             | -     | 0       | 9.4343          | 1.0000           |
| **Winners**                 | -     | 1       | -4.9040         | 0.2454           |

## Acknowledgments
Research Project: **FAKE news and HATE speech (FAKEnHATE-PdC)** 

Grant n. PDC2022-133118-I00 funded by MCIN/AEI/10.13039/501100011033 and by European Union NextGenerationEU/PRTR 

<img src="logos/ministerio.png" height="200" style="display: inline-block;"> <img src="logos/plan.png" height="200" style="display: inline-block;">