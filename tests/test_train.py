import os
import shutil
from pathlib import Path

from gcp2pnet.train import main, get_train_arguments

def test_get_train_arguements():

    args = get_train_arguments()

    # assert default configs
    assert args.lr == 1e-3
    assert args.lr_fpn == 1e-4
    assert args.epochs == 100

    assert args.output_dir == './runs'


def test_main_one_epoch_run():

    args = get_train_arguments()
    args.dataset_folder = "./data/demo_dataset"
    args.run_name = "tests"
    args.epochs = 2
    args.eval_freq = 1

    tests_out = "./runs/tests"
    if os.path.exists(tests_out):
        shutil.rmtree(tests_out)

    # Test if train.main() runs without errors
    try:
        
        main(args)

    except Exception as e:
        assert False, f"main() raised an exception: {e}"

    # Test if output folder was created
    assert os.path.exists( tests_out ), "Output folder was not created"

    assert os.path.exists( f"{tests_out}/args.yaml")
    assert os.path.exists( f"{tests_out}/weights/latest.pth")
    assert os.path.exists( f"{tests_out}/weights/best_mae.pth")