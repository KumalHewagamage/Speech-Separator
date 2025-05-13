# Speech Seperator

## Environment Setup

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

```bash
# 1. Create a new conda environment with Python 3.13.3
conda create -n vsep python=3.13.3

# 2. Activate the environment
conda activate vsep

# 3. Install required packages from requirements.txt
pip install -r requirements.txt

```

## Prepare Dataset

1. Download LibriSpeech dataset

    Get LibriSpeech dataset at http://www.openslr.org/12/.
    - For Training
        - `train-clear-100.tar.gz`(6.3G) contains speech of 252 speakers, and `train-clear-360.tar.gz`(23G) contains 922 speakers. You may use either, but the more speakers you have in dataset, the more better VoiceFilter will be.
    - For Testing
        - `test-clean.tar.gz` is sufficient.

    Unzip `tar.gz` file to desired folder:

    eg:

    ```bash
    tar -xvzf test-clean.tar.gz
    ```

## For Training

(skip all this and go to `Evaluate` section if you only want to test.)
1. Resample & Normalize wav files

    Copy `utils/normalize-resample.sh` to root directory of unzipped data folder. Then:
    ```bash
    vim normalize-resample.sh # set "N" as your CPU core number.
    chmod a+x normalize-resample.sh
    ./normalize-resample.sh # this may take long
    ```

1. Edit `config.yaml`

    ```bash
    cd config
    cp default.yaml config.yaml
    vim config.yaml
    ```

1. Preprocess wav files

    In order to boost training speed, perform STFT for each files before training by:
    ```bash
    python generator.py -c [config yaml] -d [data directory] -o [output directory] -p [processes to run]
    ```
    This will create 100,000(train) + 1000(test) data. (About 160G)


## Train VoiceFilter

1. Get pretrained model for speaker recognition system

    VoiceFilter utilizes speaker recognition system ([d-vector embeddings](https://google.github.io/speaker-id/publications/GE2E/)).
    Here, we provide pretrained model for obtaining d-vector embeddings.

    This model was trained with [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset,
    where utterances are randomly fit to time length [70, 90] frames.
    Tests are done with window 80 / hop 40 and have shown equal error rate about 1%.
    Data used for test were selected from first 8 speakers of [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test dataset, where 10 utterances per each speakers are randomly selected.
    
    **Update**: Evaluation on VoxCeleb1 selected pair showed 7.4% EER.
    
    The model can be downloaded at [this GDrive link](https://drive.google.com/file/d/1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL/view?usp=sharing).

1. Run

    After specifying `train_dir`, `test_dir` at `config.yaml`, run:
    ```bash
    python trainer.py -c [config yaml] -e [path of embedder pt file] -m [name]
    ```
    This will create `chkpt/name` and `logs/name` at base directory(`-b` option, `.` in default)

1. View tensorboardX

    ```bash
    tensorboard --logdir ./logs
    ```
    
    ![](./assets/tensorboard.png)

1. Resuming from checkpoint

    ```bash
    python trainer.py -c [config yaml] --checkpoint_path [chkpt/name/chkpt_{step}.pt] -e [path of embedder pt file] -m name
    ```

## Evaluate

For evaluvation download  pretrained weights and move them to `ckpt` folder. You need following:
-   embedder.pt (speech encoder)
-   seperator_best_checkpoint.pt (seperator weights)
-   whisper-small (entire archive from hugging face. )
    - run `git clone https://huggingface.co/openai/whisper-small` and move the folder to `ckpt` folder

Use pipeline_ver1.ipynb notebook for evaluvation.

1. Run `Mixed Audio Generator` to create 2 speaker conversation dataset for testing.
2. Run `Separator` to separate previously created mixed audio.
3. Run `Eval-Dvec` to run trasncription setup with d-vec confidence filtering.
4. Run `Visualizer` to inspect processed audio.