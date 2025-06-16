# GrainCountingP2PNet: An RGB image-based phenotyping system for assessing spikelet fertility in rice panicles

\[link to paper \]

GrainCountingNet is a modified P2PNet with multi level pyramid features ...

## Pretrained Models

Model name: `official_best_mae.pth`, download link: \[Google Drive links\]

## Evaluation

This project has been examined on Linux Manjaro x86 machine with Nvidia RTX 4090, Windows OS has not been tested but should be executable.

### Setup Environment

We recommend to using [uv](https://docs.astral.sh/uv/getting-started/installation/) to setup and manage your developing environments. Please ensure the command uv is accessable to your command line:

```bash
> uv --version
uv 0.6.14
```

Using the following command to setup the virtual environment for code development:

```bash
> cd "/path/to/source/code/of/this/project"
...GrainCountingNet > uv venv  # create virtual env
...GrainCountingNet > uv sync  # install all dependencies
```

To use that virtual environment, you can run python scripts directly (refer [Running scripts | uv](https://docs.astral.sh/uv/guides/scripts/)):

```bash
...GrainCountingNet > uv run xxxx.py
```

Or activate the virtual enviroment in traditional way:

```bash
...GrainCountingNet > source .venv/bin/activate
(GrainCountingNet) ...GrainCountingNet > python xxxx.py
```

### Download datasets

Dataset under preparation

### Prepare datasets

Data preprocessing code under construction

### Inference 

To be continued

### Training 

```bash
...GrainCountingNet > uv run -m gcp2pnet.train \
    --dataset_folder ./data/dataset \
    --batch_size 1 \  # due to multi-class, currently only batch_size=1 is supported.
    --epochs 100 \ 
    --run_name demo_train \
    ...
```

After training, using the following command to check the results figure by tensorboard:

```bash
...GrainCountingNet > uv run tensorboard --logdir ./runs/<run_name>/tensorboard_logs --port 8123

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.19.0 at http://localhost:8123/ (Press CTRL+C to quit)
```

Then press `ctrl` + left click to open the `localhost;8123` to check in browser.

**Train on your own data**

Label data on v7labs, with the following structure.

Then execute the parepare datasets code to prepare training dataset.

## Publications

Please cite our paper if this project helps you:

```bib
Under review
```
