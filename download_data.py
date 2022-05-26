import argparse
import subprocess
import pandas as pd
import math
import os

ARkitscense_url = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
TRAINING = 'Training'
VALIDATION = 'Validation'
HIGRES_DEPTH_ASSET_NAME = 'highres_depth'
POINT_CLOUDS_FOLDER = 'laser_scanner_point_clouds'

missing_3dod_assets_video_ids = ['47334522','47334523','42897421','45261582','47333152','47333155',
                                 '48458535','48018733','47429677','48458541','42897848','47895482',
                                 '47333960','47430089','42899148','42897612','42899153','42446164',
                                 '48018149','47332198','47334515','45663223','45663226','45663227']


def raw_files(video_id, assets):
    file_names = []
    for asset in assets:
        if asset in ['confidence', 'highres_depth', 'lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics',
                     'ultrawide', 'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics']:
            file_names.append(asset + '.zip')
        elif asset == 'mov':
            file_names.append(f'{video_id}.mov')
        elif asset == 'mesh':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f'{video_id}_3dod_mesh.ply')
        elif asset == 'annotation':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f'{video_id}_3dod_annotation.json')
        elif asset == 'lowres_wide.traj':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append('lowres_wide.traj')
        else:
            raise Exception(f'No asset = {asset} in raw dataset')
    return file_names


def download_file(url, file_name, dst):
    command = f"curl {url} -o {file_name} --fail"
    try:
        subprocess.check_call(command, shell=True, cwd=dst)
    except Exception as error:
        print(f'Error downloading {url}, error: {error}')
        return False

    return True


def download_laser_scanner_point_clouds_for_video(video_id, metadata, download_dir):
    visit_id = metadata.loc[metadata['video_id'] == float(video_id), ['visit_id']].iat[0, 0]
    has_laser_scanner_point_clouds = metadata.loc[metadata['video_id'] == float(video_id),
                                                  ['has_laser_scanner_point_clouds']].iat[0, 0]
    if math.isnan(visit_id) or not visit_id.is_integer():
        return

    if not has_laser_scanner_point_clouds:
        print(f"Laser scanner point clouds for video {video_id} are not available")
        return

    visit_id = int(visit_id)  # Expecting an 8 digit integer
    laser_scanner_point_clouds_ids = laser_scanner_point_clouds_for_visit_id(visit_id, download_dir)

    for point_cloud_id in laser_scanner_point_clouds_ids:
        download_laser_scanner_point_clouds(point_cloud_id, visit_id, download_dir)


def laser_scanner_point_clouds_for_visit_id(visit_id, download_dir):
    point_cloud_to_visit_id_mapping_filename = "laser_scanner_point_clouds_mapping.csv"
    point_cloud_to_visit_id_mapping_url = \
        f"{ARkitscense_url}/raw/laser_scanner_point_clouds/{point_cloud_to_visit_id_mapping_filename}"
    if not download_file(point_cloud_to_visit_id_mapping_url, point_cloud_to_visit_id_mapping_filename, download_dir):
        print(
            f"Error downloading point cloud for visit_id {visit_id} at location {point_cloud_to_visit_id_mapping_url}")
        return []

    point_cloud_to_visit_id_mapping_filepath = os.path.join(download_dir, point_cloud_to_visit_id_mapping_filename)
    point_cloud_to_visit_id_mapping = pd.read_csv(point_cloud_to_visit_id_mapping_filepath)
    point_cloud_ids = point_cloud_to_visit_id_mapping.loc[
        point_cloud_to_visit_id_mapping['visit_id'] == visit_id, ["laser_scanner_point_clouds_id"]
    ]
    point_cloud_ids_list = [scan_id[0] for scan_id in point_cloud_ids.values]

    return point_cloud_ids_list


def download_laser_scanner_point_clouds(laser_scanner_point_cloud_id, visit_id, download_dir):
    laser_scanner_point_clouds_folder_path = os.path.join(download_dir, POINT_CLOUDS_FOLDER, str(visit_id))
    point_cloud_filename = f"{laser_scanner_point_cloud_id}.ply"
    point_cloud_filepath = os.path.join(laser_scanner_point_clouds_folder_path, point_cloud_filename)
    if os.path.exists(point_cloud_filepath):
        return

    if not os.path.exists(laser_scanner_point_clouds_folder_path):
        os.makedirs(laser_scanner_point_clouds_folder_path)

    point_cloud_url = f"{ARkitscense_url}/raw/laser_scanner_point_clouds/{visit_id}/{point_cloud_filename}"
    download_file(point_cloud_url, point_cloud_filename, laser_scanner_point_clouds_folder_path)


def get_metadata(dataset, download_dir):
    filename = "metadata.csv"
    url = f"{ARkitscense_url}/threedod/{filename}" if '3dod' == dataset else f"{ARkitscense_url}/{dataset}/{filename}"
    dst_folder = os.path.join(download_dir, dataset)
    dst_file = os.path.join(dst_folder, filename)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not download_file(url, filename, dst_folder):
        return

    metadata = pd.read_csv(dst_file)
    return metadata


def download_data(dataset,
                  video_ids,
                  splits,
                  download_dir,
                  keep_zip,
                  raw_dataset_assets,
                  should_download_laser_scanner_point_cloud,
                  ):
    metadata = get_metadata(dataset, download_dir)
    if None is metadata:
        print(f"Error retrieving metadata for dataset {dataset}")
        return

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

        if should_download_laser_scanner_point_cloud and dataset == 'raw':
            # Point clouds only available for the raw dataset
            download_laser_scanner_point_clouds_for_video(video_id, metadata, download_dir)

        for file_name in file_names:
            dst_zip = os.path.join(dst_dir, file_name)
            url = url_prefix.format(file_name)
            if HIGRES_DEPTH_ASSET_NAME in file_name:
                in_upsampling = metadata.loc[metadata['video_id'] == float(video_id), ['is_in_upsampling']].iat[0, 0]
                if not in_upsampling:
                    print(f"Skipping asset {file_name} for video_id {video_id} - Video not in upsampling dataset")
                    continue  # highres_depth asset only available for videos in upsampling dataset

            if download_file(url, file_name, dst_dir):
                # unzipping data
                if file_name.endswith('.zip'):
                    command = f"unzip -o {dst_zip} -d {dst_dir}"
                    subprocess.check_call(command, shell=True)
                    if not keep_zip:
                        os.remove(dst_zip)

    if dataset == 'upsampling' and VALIDATION in splits:
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
        "--download_laser_scanner_point_cloud",
        action='store_true'
    )

    parser.add_argument(
        "--raw_dataset_assets",
        nargs='+',
        default=['mov', 'annotation', 'mesh', 'confidence', 'highres_depth', 'lowres_depth',
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
                  args.raw_dataset_assets,
                  args.download_laser_scanner_point_cloud)
