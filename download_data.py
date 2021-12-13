import argparse
import subprocess
import pandas as pd
import os

ARkitscense_url = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
TRAINING = 'Training'
VALIDATION = 'Validation'


def raw_files(video_id, assets):
    file_names = []
    for asset in assets:
        if asset in ['confidence','lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics',
                      'ultrawide', 'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics']:
            file_names.append(asset + '.zip')
        elif asset == 'mov':
            file_names.append(f'{video_id}.mov')
        elif asset == 'mesh':
            file_names.append(f'{video_id}_3dod_mesh.ply')
        elif asset == 'annotation':
            file_names.append(f'{video_id}_3dod_annotation.json')
        elif asset == 'lowres_wide.traj':
            file_names.append('lowres_wide.traj')
        else:
            raise Exception(f'No asset = {asset} in raw dataset')
    return file_names


def download_file(url, file_name, dst):
    command = f"curl {url} -o {file_name}"
    subprocess.check_call(command, shell=True, cwd=dst)


def download_data(dataset,
                  video_ids,
                  splits,
                  download_dir,
                  keep_zip,
                  raw_dataset_assets
                  ):
    download_dir = os.path.abspath(download_dir)
    for video_id in set(video_ids):
        split = splits[video_ids.index(video_id)]
        dst_dir = os.path.join(download_dir, dataset, split)
        if dataset == 'raw':
            dst_dir = os.path.join(dst_dir, str(video_id))
            url_prefix = f"{ARkitscense_url}/raw/{split}/{video_id}" + "/{}"
            file_names = raw_files(video_id, raw_dataset_assets)
        elif dataset == '3dod':
            url_prefix = f"{ARkitscense_url}/threedod/{split}" + "/{}"
            file_names = [f"{video_id}.zip", ]
        elif dataset == 'upsampling':
            url_prefix = f"{ARkitscense_url}/upsampling/{split}" + "/{}"
            file_names = [f"{video_id}.zip", ]
        else:
            raise Exception(f'No such dataset = {dataset}')
        os.makedirs(dst_dir, exist_ok=True)

        for file_name in file_names:
            dst_zip = os.path.join(dst_dir, file_name)
            url = url_prefix.format(file_name)
            download_file(url, file_name, dst_dir)

            # unzipping data
            if file_name.endswith('.zip'):
                command = f"unzip {dst_zip} -d {dst_dir}"
                subprocess.check_call(command, shell=True)
                if not keep_zip:
                    os.remove(dst_zip)

    if dataset == 'upsampling':
        meta_file = "metadata.csv"
        url = f"{ARkitscense_url}/upsampling/{meta_file}"
        dst_file = os.path.join(download_dir, dataset)
        download_file(url, meta_file, dst_file)

        if VALIDATION in splits:
            val_attributes_file = "val_attributes.csv"
            url = f"{ARkitscense_url}/upsampling/{VALIDATION}/{val_attributes_file}"
            dst_file = os.path.join(download_dir, dataset, VALIDATION)
            download_file(url, val_attributes_file, dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        choices=['3dod', 'upsampling', 'raw']
    )

    parser.add_argument(
        "--split",
        choices=["Training", "Validation"],
    )

    parser.add_argument(
        "--video_id",
        nargs='*'
    )

    parser.add_argument(
        "--video_id_csv",
    )

    parser.add_argument(
        "--download_dir",
        default="data",
    )

    parser.add_argument(
        "--keep_zip",
        action='store_true'
    )

    parser.add_argument(
        "--raw_dataset_assets",
        nargs='+',
        default=['mov', 'annotation', 'mesh', 'confidence', 'lowres_depth',
                 'lowres_wide.traj', 'lowres_wide', 'lowres_wide_intrinsics', 'ultrawide',
                  'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics']
    )

    args = parser.parse_args()
    if args.video_id is None and args.video_id_csv is None:
        raise argparse.ArgumentError('video_id or video_id_csv must be specified')
    elif args.video_id is not None and args.video_id_csv is not None:
        raise argparse.ArgumentError('only video_id or video_id_csv must be specified')
    if args.video_id is not None and args.split is None:
        raise argparse.ArgumentError('given video_id the split argument must be specified')

    if args.video_id is not None:
        video_ids_ = args.video_id
        splits_ = splits = [args.split, ] * len(video_ids_)
    elif args.video_id_csv is not None:
        df = pd.read_csv(args.video_id_csv)
        if args.split is not None:
            df = df[df["fold"] == args.split]
        video_ids_ = df["video_id"].to_list()
        splits_ = df["fold"].to_list()
    else:
        raise Exception('No video ids specified')

    download_data(args.dataset,
                  video_ids_,
                  splits_,
                  args.download_dir,
                  args.keep_zip,
                  args.raw_dataset_assets)
