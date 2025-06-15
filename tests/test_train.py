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


def test_main_prepare_output_folder():

    args = get_train_arguments()
    args.run_name = "tests"

    tests_out = "./runs/tests"

    if os.path.exists(tests_out):
        shutil.rmtree(tests_out)

    # Test if main runs without errors
    try:
        args.epoch = 1
        main(args)

        # Test if output folder was created
        assert os.path.exists(tests_out), "Output folder was not created"

        assert os.path.exists( Path(tests_out) / "args.yaml")

        
    except Exception as e:
        assert False, f"main() raised an exception: {e}"


    
