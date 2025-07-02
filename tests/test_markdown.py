
def test_inference_api():

    from pathlib import Path
    import gcp2pnet

    args = gcp2pnet.inference.get_inf_arguments()
    args.weight_path = Path("./demo_best_mae.pth")
    args.img_path = Path("./data/20220207_18_G_a.JPG")
    args.result_folder = Path("data/")
    args.sliding_window = True
    args.window_size = 256 * 3
    args.overlap_ratio = 0.2
    args.merge_distance = 25 * 3
    args.patch_result_folder = Path("data/patch/")
    args.threshold = 0.5

    result_df = gcp2pnet.inference.main(args)

    print(result_df)

def test_train_api():
    import os
    import shutil
    from pathlib import Path
    import gcp2pnet

    args = gcp2pnet.train.get_train_arguments()
    args.dataset_folder = Path("./data/demo_dataset")
    args.batch_size = 8
    args.epochs = 1
    args.run_name = "markdown_train"
    args.seed = 42

    tests_out = f"./runs/{args.run_name}"
    if os.path.exists(tests_out):
        shutil.rmtree(tests_out)

    gcp2pnet.train.main(args)