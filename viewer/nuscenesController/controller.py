import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image


class NuScenesVisualizer:
    def __init__(self, dataset_root, version="v1.0-mini"):
        """
        nuScenes のデータセットを読み込むクラス
        :param dataset_root: nuScenes のデータセットパス
        :param version: 使用するバージョン (デフォルト: "v1.0-mini")
        """
        self.nusc = NuScenes(version=version, dataroot=dataset_root, verbose=True)
        self.camera_channels = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
        ]
        self.lidar_channel = "LIDAR_TOP"

    def get_first_sample_token(self, scene_index=0):
        """
        指定したシーンの最初のフレームのサンプルトークンを取得
        :param scene_index: 取得するシーンのインデックス (デフォルト: 0)
        :return: 最初のフレームの sample_token
        """
        return self.nusc.scene[scene_index]["first_sample_token"]

    def load_camera_images(self, sample):
        """
        カメラ画像を取得
        :param sample: nuScenes の sample データ
        :return: 画像のリスト
        """
        images = []
        for cam in self.camera_channels:
            cam_data = self.nusc.get("sample_data", sample["data"][cam])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            img = Image.open(img_path)
            images.append(img)
        return images

    def load_lidar_data(self, sample):
        """
        LiDAR の点群を取得
        :param sample: nuScenes の sample データ
        :return: LiDAR の点群データ (numpy array)
        """
        lidar_data = self.nusc.get("sample_data", sample["data"][self.lidar_channel])
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        return pc.points  # (4, N) の numpy 配列

    def visualize_sample(self, sample_token):
        """
        指定したサンプルトークンのカメラ画像と LiDAR 点群を可視化
        :param sample_token: 可視化するサンプルのトークン
        """
        sample = self.nusc.get("sample", sample_token)
        images = self.load_camera_images(sample)
        lidar_points = self.load_lidar_data(sample)

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.ravel()

        # LiDAR 点群をプロット
        ax_lidar = plt.subplot(3, 3, 5)

        ax_lidar.scatter(lidar_points[0, :], lidar_points[1, :], s=1, c=lidar_points[2, :], cmap="viridis")
        #ax_lidar.set_title("LiDAR Point Cloud")
        ax_lidar.axis("equal")
        ax_lidar.grid(True)
        # カメラ画像をプロット
        for i, img in enumerate(images):
            place_idx = i
            if i>=3:
                place_idx +=  3
            axes[place_idx].imshow(img)
            axes[i].axis("off")
            axes[place_idx].axis("off")
            axes[place_idx].set_title(self.camera_channels[i])

        txt_info= plt.subplot(3, 3, 6)
        sample_token_txt = f"Sample Data: {sample_token}\n"
        scene_index_txt = f"Scene Index: {sample['scene_token']}\n"
        
        cam = self.camera_channels[1]
        sample_data_token = sample['data'][cam]
        sample_data = self.nusc.get('sample_data', sample_data_token)
        filename_txt=f"{cam}: {sample_data['filename']}"

        txt_info.text(0.5, 0.5, sample_token_txt, fontsize=14, ha='center', va='center')
        txt_info.text(0.5, 0.4, scene_index_txt, fontsize=14, ha='center', va='center')
        txt_info.text(0.5, 0.3, filename_txt, fontsize=7, ha='center', va='center')



        plt.tight_layout()
        plt.show()

    def visualize_n_frames(self, n_frames=5, scene_index=1):
        """
        指定したシーンの最初の n フレームを可視化
        :param n_frames: 可視化するフレーム数
        :param scene_index: 可視化するシーンのインデックス
        """
        sample_token = self.get_first_sample_token(scene_index)
        for _ in range(n_frames):
            if not sample_token:
                break
            self.visualize_sample(sample_token)
            sample_token = self.nusc.get("sample", sample_token)["next"]

