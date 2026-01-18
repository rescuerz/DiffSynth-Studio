"""
DiffSynth-Studio 的通用数据集抽象。

`UnifiedDataset` 统一了训练脚本从元数据中加载样本的方式：
  - 图片 / 视频 / GIF（在 metadata 里以文件路径形式出现）
  - 可选的额外模态（音频、参考图、控制视频等）

两种工作模式：
  1) 元数据模式（提供 `metadata_path`）：
       - 读取 CSV / JSON / JSONL 到 `self.data`
       - 对 `data_file_keys` 中的每个字段，加载并预处理其对应资源
  2) 缓存模式（`metadata_path` 为 None）：
       - 递归搜索 `base_path` 下的 `.pth` 缓存张量
       - 直接产出缓存张量（用于拆分训练）

视频训练的典型 metadata.csv：
    video,prompt
    video_1.mp4,"a dog running"
    video_2.mp4,"a cat jumping"
"""

from .operators import *
import torch, json, pandas


class UnifiedDataset(torch.utils.data.Dataset):
    """小而灵活的数据集封装。

    输入：
      - `base_path`：元数据中相对路径的根目录。
      - `metadata_path`：CSV/JSON/JSONL；若为 None 则启用缓存模式。
      - `data_file_keys`：每条元数据中需要通过 operator 加载的字段（如 `("video", "image")`）。
      - `main_data_operator`：对非 special key 的通用处理算子。
      - `special_operator_map`：对特定字段的专用处理算子（按 key 覆盖）。

    输出（`__getitem__`）：
      - 元数据模式：返回 dict，字段值已被加载为对象（如 `video` -> list[PIL.Image]）。
      - 缓存模式：返回从 `.pth` 中加载的缓存 tensor/tuple。
    """
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        """图像字段的默认 operator。

        - 同时支持单张图片路径（str）与图片路径列表（list[str]）。
        - 使用裁剪/缩放以满足模型约束（division factor 对齐）。
        """
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        """视频字段的默认 operator。

        - 按扩展名分发：
            * 图片 -> 单帧“视频”（list 内仅 1 张 PIL.Image）
            * gif/video -> 采样得到的帧序列（list[PIL.Image]）
        - 对帧采样与裁剪/缩放使用一致逻辑。
        """
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        """递归查找 `path` 下的 `.pth` 缓存文件（缓存模式）。"""
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        """加载元数据，或在缓存模式下发现缓存数据。"""
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        """返回单条训练样本（batch 逻辑由 DataLoader 负责）。"""
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        """应用 `repeat` 后的数据集长度。"""
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
