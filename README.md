# GrainCountingP2PNet: An RGB image-based phenotyping system for assessing spikelet fertility in rice panicles

\[link to paper \]

GrainCountingP2PNet is a modified P2PNet with multi level pyramid features ...

## Pretrained Models

Model name: `official_best_mae.pth`, download link: \[Google Drive links\]

## Evaluation

This project has been examined on Linux Manjaro x86 machine with Nvidia RTX 4090, Windows OS has not been tested but should be executable.

### Setup Environment

To ensure reproducibility, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to create and manage Python virtual environments. Please verify that the `uv` command is available in your command line:

```bash
> uv --version
uv 0.6.14
```

Using the following command to setup the virtual environment for code development:

```bash
> git clone https://github.com/UTokyo-FieldPhenomics-Lab/GrainCountingP2PNet.git
> cd ./GrainCountingP2PNet
> uv venv  # create virtual env
> uv sync  # install all dependencies
```

To use that virtual environment, you can run python scripts directly (refer [Running scripts | uv](https://docs.astral.sh/uv/guides/scripts/)):

```bash
> uv run xxxx.py
```

Which is almost equal to the traditional way that activate the virtual enviroment and execute the code:

```bash
> source .venv/bin/activate
(.venv) > python xxxx.py
```

### Download datasets

The organized demo dataset for training is available at [release/demo_dataset.zip](https://github.com/UTokyo-FieldPhenomics-Lab/GrainCountingP2PNet/releases/tag/v0.0.1)

Please download and unzip contents into `data/demo_dataset/` with the following structures:

```
data/demo_dataset/
|-- train/
|-- valid/
|-- classes.json
```

### Prepare datasets

Data preprocessing code under construction

### Inference 

To be continued

### Training 

```bash
> uv run -m gcp2pnet.train \
    --dataset_folder ./data/demo_dataset \
    --batch_size 1 \  # due to multi-class, currently only batch_size=1 is supported.
    --epochs 100 \ 
    --run_name demo_train \
    ...
```

After training, using the following command to check the results figure by tensorboard:

```bash
> uv run tensorboard \
    --logdir ./runs/<run_name>/tensorboard_logs \
    --port 8123

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.19.0 at http://localhost:8123/ (Press CTRL+C to quit)
```

Then press `ctrl` + left click to open the `localhost;8123` to check in browser.

**Train on your own data**

Label data on v7labs, with the following structure.

Then execute the parepare datasets code to prepare training dataset.

(to be continued)

## Publications

Please cite our paper if this project helps you:

```bib
Under review
```
