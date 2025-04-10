import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import shutil


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
        self.selection_start = None
        self.selection_end = None
        self.current_ax = None
        self.current_img = None
        self.rect = None
        self.current_img_path = None
        self.edited_dataset_root = os.path.join(os.path.dirname(dataset_root), "v1.0-mini-edited")
        
        # 編集用のフォルダを作成
        if not os.path.exists(self.edited_dataset_root):
            os.makedirs(self.edited_dataset_root)
            # 元のデータセットの構造をコピー
            for root, dirs, files in os.walk(dataset_root):
                for dir_name in dirs:
                    src_dir = os.path.join(root, dir_name)
                    dst_dir = src_dir.replace(dataset_root, self.edited_dataset_root)
                    os.makedirs(dst_dir, exist_ok=True)
                for file_name in files:
                    src_file = os.path.join(root, file_name)
                    dst_file = src_file.replace(dataset_root, self.edited_dataset_root)
                    shutil.copy2(src_file, dst_file)

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

    def on_press(self, event):
        """マウスボタンが押された時のイベントハンドラ"""
        if event.inaxes != self.current_ax:
            return
        self.selection_start = (event.xdata, event.ydata)
        self.rect = plt.Rectangle(self.selection_start, 0, 0, fill=False, edgecolor='red', linewidth=2)
        self.current_ax.add_patch(self.rect)

    def on_release(self, event):
        """マウスボタンが離された時のイベントハンドラ"""
        if event.inaxes != self.current_ax or not self.selection_start:
            return
        self.selection_end = (event.xdata, event.ydata)
        
        # 選択範囲の座標を整数に変換
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        if isinstance(self.current_img, np.ndarray):  # LiDAR点群の場合
            # 点群データのコピーを作成
            points = self.current_img.copy()
            
            # 選択範囲内の点を黒塗り（z座標を0に設定）
            mask = (points[0, :] >= x1) & (points[0, :] <= x2) & \
                   (points[1, :] >= y1) & (points[1, :] <= y2)
            
            # 選択範囲内の点のz座標を0に設定
            points[2, mask] = 0
            
            # 点群を更新して表示
            self.current_ax.clear()
            scatter = self.current_ax.scatter(points[0, :], points[1, :], s=1, c=points[2, :], cmap="viridis")
            self.current_ax.axis("equal")
            self.current_ax.grid(True)
            
            # カラーバーを更新
            if hasattr(self, 'colorbar'):
                self.colorbar.remove()
            self.colorbar = plt.colorbar(scatter, ax=self.current_ax)
            self.colorbar.set_label('Z coordinate')
            
            plt.draw()
            
        else:  # カメラ画像の場合
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # PIL Imageをnumpy配列に変換
            img_array = np.array(self.current_img)
            # 選択範囲を黒塗り
            img_array[y1:y2, x1:x2] = 0
            # numpy配列をPIL Imageに変換
            blacked_img = Image.fromarray(img_array)
            
            # 編集した画像を保存
            if self.current_img_path:
                edited_img_path = self.current_img_path.replace(self.nusc.dataroot, self.edited_dataset_root)
                blacked_img.save(edited_img_path)
            
            # 元の画像を更新
            self.current_ax.clear()
            self.current_ax.imshow(blacked_img)
            self.current_ax.axis("off")
            plt.draw()
        
        # 選択範囲のリセット
        self.selection_start = None
        self.selection_end = None
        if self.rect:
            self.rect.remove()
            self.rect = None

    def on_motion(self, event):
        """マウスが移動した時のイベントハンドラ"""
        if event.inaxes != self.current_ax or not self.selection_start or not self.rect:
            return
        x, y = event.xdata, event.ydata
        width = x - self.selection_start[0]
        height = y - self.selection_start[1]
        self.rect.set_width(width)
        self.rect.set_height(height)
        plt.draw()

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
        scatter = ax_lidar.scatter(lidar_points[0, :], lidar_points[1, :], s=1, c=lidar_points[2, :], cmap="viridis")
        ax_lidar.axis("equal")
        ax_lidar.grid(True)
        
        # カラーバーを追加
        self.colorbar = plt.colorbar(scatter, ax=ax_lidar)
        self.colorbar.set_label('Z coordinate')

        # LiDAR点群のマウスイベントを接続
        def on_lidar_click(event, ax=ax_lidar, points=lidar_points):
            if event.inaxes == ax:
                self.current_ax = ax
                self.current_img = points.copy()  # 点群データのコピーを保持
                self.current_img_path = None
                self.selection_start = None
                self.selection_end = None
                if self.rect:
                    self.rect.remove()
                    self.rect = None

        fig.canvas.mpl_connect('button_press_event', on_lidar_click)

        # カメラ画像をプロット
        for i, img in enumerate(images):
            place_idx = i
            if i >= 3:
                place_idx += 3
            axes[place_idx].imshow(img)
            axes[i].axis("off")
            axes[place_idx].axis("off")
            axes[place_idx].set_title(self.camera_channels[i])

            # マウスイベントの接続
            def on_click(event, ax=axes[place_idx], image=img, img_path=None):
                if event.inaxes == ax:
                    self.current_ax = ax
                    self.current_img = image
                    self.current_img_path = img_path
                    self.selection_start = None
                    self.selection_end = None
                    if self.rect:
                        self.rect.remove()
                        self.rect = None

            # 画像のパスを取得
            cam_data = self.nusc.get("sample_data", sample["data"][self.camera_channels[i]])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            
            fig.canvas.mpl_connect('button_press_event', lambda event, ax=axes[place_idx], image=img, img_path=img_path: on_click(event, ax, image, img_path))

        # マウスイベントの接続
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        txt_info = plt.subplot(3, 3, 6)
        sample_token_txt = f"Sample Data: {sample_token}\n"
        scene_index_txt = f"Scene Index: {sample['scene_token']}\n"
        
        cam = self.camera_channels[1]
        sample_data_token = sample['data'][cam]
        sample_data = self.nusc.get('sample_data', sample_data_token)
        filename_txt = f"{cam}: {sample_data['filename']}"

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

