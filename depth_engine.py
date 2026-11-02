"""
depth_engine.py
---------------
Модуль для роботи з моделлю Depth Anything V2:
- Монокулярна оцінка глибини з фотографій (Small, Base, Large)
- Генерація нормалізованих карт глибини та 16-бітних карт для 3D/VFX
- Колірне картування (Inferno, Viridis, Turbo, Plasma, Magma тощо)
- Створення 3D хмари точок (.ply) та 3D полігонального мешу (.obj)
- Симуляція глибини різкості (Bokeh / Depth-of-Field портретний ефект)
- Генерація стереоскопічної 3D Wiggle GIF анімації (паралакс-ефект)
"""

import os
import sys
import shutil
from typing import Dict, Any, Tuple, Optional, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Забезпечуємо наявність ffmpeg.exe у PATH для Gradio та відеокодеків
try:
    import imageio_ffmpeg
    _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    if _ffmpeg_src and os.path.exists(_ffmpeg_src):
        _ffmpeg_dir = os.path.dirname(_ffmpeg_src)
        _ffmpeg_alias = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(_ffmpeg_alias):
            try:
                shutil.copy2(_ffmpeg_src, _ffmpeg_alias)
            except Exception:
                pass
        if _ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib
import matplotlib.cm as cm
from transformers import pipeline

try:
    from depth_anything_3.api import DepthAnything3
    HAS_DA3 = True
except Exception:
    DepthAnything3 = None
    HAS_DA3 = False

# Доступні моделі Depth Anything V3, V2, V1 та MiDaS
MODELS = {
    # Depth Anything 3 (DA3 — Останнє покоління ByteDance Seed)
    "Depth Anything 3 - Small (~34M, надшвидка)": "da3:depth-anything/DA3-SMALL",
    "Depth Anything 3 - Base (~115M, збалансована точність)": "da3:depth-anything/DA3-BASE",
    "Depth Anything 3 - Large (~400M, максимальна якість)": "da3:depth-anything/DA3-LARGE",
    "Depth Anything 3 - Mono Large (~300M, монокулярна)": "da3:depth-anything/DA3MONO-LARGE",
    "Depth Anything 3 - Metric Large (~300M, метрична глибина)": "da3:depth-anything/DA3METRIC-LARGE",
    "Depth Anything 3 - Nested Giant Large (~1.4B, ультра-якість)": "da3:depth-anything/DA3NESTED-GIANT-LARGE",
    # Depth Anything V2 (Перевірена основа)
    "Depth Anything V2 - Small (~25M, найшвидша)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 - Base (~97M, відмінна чіткість)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 - Large (~335M, висока точність)": "depth-anything/Depth-Anything-V2-Large-hf",
    # Depth Anything V1
    "Depth Anything V1 - Small (~25M)": "LiheYoung/depth-anything-small-hf",
    "Depth Anything V1 - Base (~97M)": "LiheYoung/depth-anything-base-hf",
    # MiDaS
    "MiDaS v2.1 - Small (~21M, класична)": "midas:MiDaS_small",
    "MiDaS v3.0 - DPT Hybrid (~120M)": "midas:DPT_Hybrid",
    "MiDaS v3.0 - DPT Large (~345M)": "midas:DPT_Large",
}

DEFAULT_MODEL_KEY = "Depth Anything 3 - Small (~34M, надшвидка)"

MODEL_ALIASES = {
    "da3": "da3:depth-anything/DA3-SMALL",
    "da3-small": "da3:depth-anything/DA3-SMALL",
    "da3-base": "da3:depth-anything/DA3-BASE",
    "da3-large": "da3:depth-anything/DA3-LARGE",
    "da3-mono": "da3:depth-anything/DA3MONO-LARGE",
    "da3-metric": "da3:depth-anything/DA3METRIC-LARGE",
    "da3-giant": "da3:depth-anything/DA3NESTED-GIANT-LARGE",
    "v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
    "v1-small": "LiheYoung/depth-anything-small-hf",
    "v1-base": "LiheYoung/depth-anything-base-hf",
    "midas": "midas:MiDaS_small",
    "midas-small": "midas:MiDaS_small",
    "midas-hybrid": "midas:DPT_Hybrid",
    "midas-large": "midas:DPT_Large",
}

COLORMAPS = [
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "turbo",
    "cividis",
    "spectral",
    "twilight",
    "gist_earth",
    "gray"
]


class DepthAnythingV2Engine:
    """
    Клас-обгортка для роботи з Depth Anything V2, V1 та MiDaS:
    забезпечує кешування моделі, вибір пристрою (CUDA/CPU) та набір пост-обробок.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.current_model_key: Optional[str] = None
        self.pipeline = None
        self.midas_model = None
        self.midas_transform = None
        self.da3_model = None

    @staticmethod
    def get_cache_paths() -> Dict[str, Any]:
        """Повертає шляхи, де зберігаються моделі на комп'ютері, та їх розмір"""
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
        torch_home = os.environ.get("TORCH_HOME") or os.path.expanduser("~/.cache/torch/hub")

        def get_dir_size_mb(path: str) -> float:
            total = 0
            if os.path.exists(path):
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total += os.path.getsize(fp)
                        except OSError:
                            pass
            return round(total / (1024 * 1024), 1)

        hf_models = []
        if os.path.exists(hf_home):
            for item in os.listdir(hf_home):
                if item.startswith("models--"):
                    hf_models.append(item.replace("models--", "").replace("--", "/"))

        torch_checkpoints = []
        chk_dir = os.path.join(torch_home, "checkpoints")
        if os.path.exists(chk_dir):
            torch_checkpoints = os.listdir(chk_dir)

        return {
            "huggingface_cache": os.path.abspath(hf_home),
            "huggingface_size_mb": get_dir_size_mb(hf_home),
            "huggingface_downloaded": hf_models,
            "torch_cache": os.path.abspath(torch_home),
            "torch_size_mb": get_dir_size_mb(torch_home),
            "torch_checkpoints": torch_checkpoints,
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Повертає інформацію про поточний пристрій та GPU"""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Немає (CPU)",
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            info["vram_total_mb"] = round(vram_total, 1)
            info["vram_allocated_mb"] = round(vram_allocated, 1)
        return info

    def load_model(self, model_key_or_name: str):
        """Завантажує обрану модель (Depth Anything 3, Depth Anything V2/V1 або MiDaS)"""
        m_clean = model_key_or_name.strip().lower()
        target = MODEL_ALIASES.get(m_clean)
        if target is None:
            target = MODELS.get(model_key_or_name)
        if target is None:
            for k, v in MODELS.items():
                if m_clean in k.lower():
                    target = v
                    break
            if target is None:
                target = model_key_or_name

        if (target.startswith("depth-anything/DA3") or target.startswith("DA3")) and not target.startswith("da3:"):
            target = f"da3:{target}"

        if self.current_model_key == target:
            return

        print(f"[*] Завантаження моделі: {model_key_or_name} ({target})...")

        if target.startswith("da3:"):
            if not HAS_DA3 or DepthAnything3 is None:
                raise RuntimeError(
                    "Пакет depth-anything-3 не встановлено. "
                    "Встановіть його командою: pip install depth-anything-3"
                )
            da3_id = target.split(":", 1)[1]
            self.da3_model = DepthAnything3.from_pretrained(da3_id).to(device=self.device).eval()
            self.pipeline = None
            self.midas_model = None
            self.midas_transform = None
            self.current_model_key = target
        elif target.startswith("midas:"):
            midas_type = target.split(":", 1)[1]
            # Завантаження MiDaS через torch.hub
            self.midas_model = torch.hub.load("intel-isl/MiDaS", midas_type).to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if midas_type == "MiDaS_small":
                self.midas_transform = midas_transforms.small_transform
            else:
                self.midas_transform = midas_transforms.dpt_transform

            self.pipeline = None
            self.da3_model = None
            self.current_model_key = target
        else:
            # Завантаження Depth Anything V2 / V1 через transformers pipeline
            device_arg = 0 if "cuda" in self.device else -1
            self.pipeline = pipeline(
                task="depth-estimation",
                model=target,
                device=device_arg,
            )
            self.midas_model = None
            self.midas_transform = None
            self.da3_model = None
            self.current_model_key = target

        print(f"[OK] Модель {target} успішно активована на {self.device}!")

    def estimate_depth(
        self,
        image: Image.Image,
        model_name: str = DEFAULT_MODEL_KEY,
        invert: bool = False
    ) -> Dict[str, Any]:
        """
        Виконує передбачення глибини для вхідного зображення.

        Повертає словник:
        - 'depth_norm': float32 масив у діапазоні [0.0, 1.0] (за замовчуванням 1.0 = близько, 0.0 = далеко)
        - 'depth_raw': вихідний масив передбаченої глибини / відстані
        - 'depth_uint16': 16-бітний масив [0, 65535] для професійного 3D/VFX
        - 'depth_uint8': 8-бітний масив [0, 255]
        - 'original_rgb': np.ndarray (RGB) оригінального зображення
        - 'size': кортеж (ширина, висота)
        - 'confidence', 'intrinsics', 'extrinsics' (якщо підтримуються моделлю, наприклад DA3)
        """
        self.load_model(model_name)

        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        extra_meta = {}

        if self.da3_model is not None:
            # Інференс для моделей Depth Anything 3 (DA3)
            with torch.no_grad():
                prediction = self.da3_model.inference([image])

            depth_pred = prediction.depth[0]
            if isinstance(depth_pred, torch.Tensor):
                depth_arr = depth_pred.detach().cpu().numpy().astype(np.float32)
            else:
                depth_arr = np.array(depth_pred, dtype=np.float32)

            if depth_arr.shape != (orig_h, orig_w):
                depth_arr = cv2.resize(
                    depth_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                )

            # У DA3 значення представляють геометричну/відносну відстань Z:
            # менші значення = ближче до камери, більші = далі.
            # Для відповідності єдиному стандарту студії (1.0 = близько, 0.0 = далеко):
            d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
            if d_max > d_min:
                depth_norm = (d_max - depth_arr) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_arr, dtype=np.float32)

            if hasattr(prediction, "conf") and prediction.conf is not None:
                extra_meta["confidence"] = prediction.conf[0]
            if hasattr(prediction, "intrinsics") and prediction.intrinsics is not None:
                extra_meta["intrinsics"] = prediction.intrinsics[0]
            if hasattr(prediction, "extrinsics") and prediction.extrinsics is not None:
                extra_meta["extrinsics"] = prediction.extrinsics[0]
            extra_meta["prediction"] = prediction

        elif self.midas_model is not None:
            # Інференс для моделей MiDaS
            img_np = np.array(image)
            input_batch = self.midas_transform(img_np).to(self.device)
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_arr = prediction.detach().cpu().numpy()

            d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
            if d_max > d_min:
                depth_norm = (depth_arr - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_arr, dtype=np.float32)

        else:
            # Інференс для моделей Depth Anything (V1 / V2)
            res = self.pipeline(image)
            pred_depth = res["predicted_depth"]
            if isinstance(pred_depth, torch.Tensor):
                depth_arr = pred_depth.detach().cpu().numpy()
            else:
                depth_arr = np.array(pred_depth, dtype=np.float32)

            if depth_arr.shape != (orig_h, orig_w):
                depth_arr = cv2.resize(
                    depth_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                )

            # Нормалізація в [0, 1]
            d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
            if d_max > d_min:
                depth_norm = (depth_arr - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_arr, dtype=np.float32)

        if invert:
            depth_norm = 1.0 - depth_norm

        depth_uint16 = (depth_norm * 65535.0).clip(0, 65535).astype(np.uint16)
        depth_uint8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        orig_rgb = np.array(image, dtype=np.uint8)

        ret = {
            "depth_norm": depth_norm.astype(np.float32),
            "depth_raw": depth_arr,
            "depth_uint16": depth_uint16,
            "depth_uint8": depth_uint8,
            "original_rgb": orig_rgb,
            "size": (orig_w, orig_h),
        }
        ret.update(extra_meta)
        return ret

    @staticmethod
    def colorize_depth(depth_norm: np.ndarray, colormap: str = "inferno") -> np.ndarray:
        """
        Застосовує колірну шкалу до нормалізованої карти глибини [0.0, 1.0].
        Повертає RGB np.ndarray (H, W, 3) uint8.
        """
        cmap_name = colormap.lower()
        if cmap_name == "gray" or cmap_name == "grayscale":
            gray_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
            return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("inferno")

        colored = cmap(depth_norm)[:, :, :3]  # RGB, діапазон [0, 1]
        return (colored * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def apply_bokeh(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        focus_depth: float = 0.5,
        depth_of_field: float = 0.15,
        blur_strength: int = 15,
    ) -> np.ndarray:
        """
        Створює реалістичний ефект розмиття фону / портретного боке:
        - focus_depth: глибина площини різкості [0.0, 1.0]
        - depth_of_field: товщина зони різкості [0.02, 0.5]
        - blur_strength: радіус розмиття об'єктиву
        """
        if blur_strength <= 0:
            return image_rgb.copy()

        # Розрахунок відстані від площини фокусування
        diff = np.abs(depth_norm - focus_depth)

        # Коефіцієнт розмиття (0 - різко, 1 - максимальне розмиття)
        dof_safe = max(depth_of_field, 0.01)
        blur_factor = np.clip((diff - dof_safe) / (1.0 - dof_safe + 1e-6), 0.0, 1.0)
        # Згладжуємо перехід
        blur_factor = cv2.GaussianBlur(blur_factor, (7, 7), 0)
        blur_mask_3ch = np.repeat(blur_factor[:, :, np.newaxis], 3, axis=2)

        # Кількарівневе якісне розмиття
        ksize = blur_strength * 2 + 1
        blurred = cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)

        # Змішування різкого та розмитого зображення
        result = (image_rgb.astype(np.float32) * (1.0 - blur_mask_3ch) +
                  blurred.astype(np.float32) * blur_mask_3ch)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def create_wiggle_gif(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        max_shift_px: int = 14,
        num_frames: int = 14,
        fps: int = 14,
    ) -> str:
        """
        Створює стереоскопічну 3D wiggle GIF-анімацію (ефект паралаксу).
        """
        h, w = depth_norm.shape
        frames: List[Image.Image] = []

        # Базова сітка координат
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)

        # Центруємо глибину навколо 0.5
        disparity = depth_norm.astype(np.float32) - 0.5

        for i in range(num_frames):
            phase = 2.0 * np.pi * (i / num_frames)
            shift = max_shift_px * np.sin(phase)

            # Розрахунок зсуву для кожного пікселя залежно від його глибини
            map_x = x_coords - shift * disparity
            map_y = y_coords

            warped = cv2.remap(
                image_rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            frames.append(Image.fromarray(warped))

        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        return output_path

    @staticmethod
    def export_pointcloud_ply(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        max_points: int = 120000,
        focal_scale: float = 1.0,
        depth_scale: float = 2.5,
    ) -> str:
        """
        Створює кольорову 3D хмару точок у форматі .PLY:
        - image_rgb: кольори для кожної точки
        - depth_norm: карта глибини
        - output_path: шлях до .ply файлу
        """
        h, w = depth_norm.shape
        total_pixels = h * w

        # Крок проріджування для оптимізації розміру
        step = max(1, int(np.sqrt(total_pixels / max_points)))
        sub_depth = depth_norm[::step, ::step]
        sub_rgb = image_rgb[::step, ::step]

        sub_h, sub_w = sub_depth.shape

        u, v = np.meshgrid(np.arange(sub_w), np.arange(sub_h))
        u = u.astype(np.float32) * step
        v = v.astype(np.float32) * step

        cx = w / 2.0
        cy = h / 2.0
        focal = max(w, h) * focal_scale

        # Z глибина: чим більше значення depth_norm (ближче), тим менша відстань
        # Додаємо безпечну базу, щоб уникнути ділення на 0
        z = (1.5 - sub_depth * 0.9) * depth_scale
        x = (u - cx) * z / focal
        y = (v - cy) * z / focal

        points = np.stack([x.ravel(), -y.ravel(), -z.ravel()], axis=-1)
        colors = sub_rgb.reshape(-1, 3)

        # Запис у бінарний або ascii PLY формат
        num_points = points.shape[0]
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            for i in range(num_points):
                f.write(
                    f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
                )

        return output_path

    @staticmethod
    def export_3d_mesh_obj(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        grid_size: int = 160,
        depth_scale: float = 0.35,
        mode: str = "room",
    ) -> str:
        """
        Створює 3D меш (.obj) з полігональною сіткою та кольорами вершин
        для швидкого інтерактивного перегляду у браузері / Blender.

        Параметр mode:
        - "room" (за замовчуванням): перспективне розгортання простору приміщення вглиб камери
        - "relief": барельєфна площина
        """
        orig_h, orig_w = depth_norm.shape
        aspect = orig_h / orig_w
        mesh_w = grid_size
        mesh_h = max(20, int(grid_size * aspect))

        d_small = cv2.resize(depth_norm, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.resize(image_rgb, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)

        # Координати X [-1, 1], Y [-aspect, aspect]
        xs = np.linspace(-1.0, 1.0, mesh_w)
        ys = np.linspace(aspect, -aspect, mesh_h)
        xx, yy = np.meshgrid(xs, ys)

        if mode == "relief":
            # Барельєф
            zz = d_small * depth_scale
            verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        else:
            # 3D приміщення (перспективна розгортка):
            # Більша відстань (задні стіни, кути) віддаляється в глибину -Z,
            # розкриваючи стелю, підлогу та стіни кімнати у повноцінний 3D об'єм
            dist = 1.0 / (0.15 + 0.85 * np.clip(d_small, 0.01, 1.0)) * depth_scale * 1.5
            px = xx * dist
            py = yy * dist
            pz = -dist
            verts = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])

        colors = rgb_small.reshape(-1, 3) / 255.0

        # Створення полігональних граней (трикутників)
        faces = []
        for r in range(mesh_h - 1):
            for c in range(mesh_w - 1):
                idx = r * mesh_w + c
                faces.append([idx + 1, idx + 2, idx + mesh_w + 1])
                faces.append([idx + 2, idx + mesh_w + 2, idx + mesh_w + 1])

        # Запис OBJ файлу
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 3D Room / Scene Mesh\n")
            for v, c in zip(verts, colors):
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        return output_path

    @staticmethod
    def export_scene_glb(
        prediction,
        output_file_or_dir: str,
        num_max_points: int = 500000,
        show_cameras: bool = True,
        image_rgb: Optional[np.ndarray] = None,
        depth_norm: Optional[np.ndarray] = None,
        grid_size: int = 160,
        depth_scale: float = 0.35,
    ) -> Optional[str]:
        """
        Експортує повноцінну 3D карту приміщення / сцени у формат .GLB:
        - Для Depth Anything 3 (коли доступний prediction): будує просторову хмару точок у координатах світу з положенням камери.
        - Для інших моделей (V2 / V1 / MiDaS): генерує кольоровий 3D меш у форматі .GLB через trimesh.
        """
        try:
            import shutil

            if output_file_or_dir.lower().endswith(".glb"):
                out_dir = os.path.dirname(os.path.abspath(output_file_or_dir)) or "."
                target_glb = os.path.abspath(output_file_or_dir)
            else:
                out_dir = os.path.abspath(output_file_or_dir)
                target_glb = os.path.join(out_dir, "scene.glb")

            os.makedirs(out_dir, exist_ok=True)

            if prediction is not None:
                from depth_anything_3.utils.export import export_to_glb

                res_path = export_to_glb(
                    prediction,
                    export_dir=out_dir,
                    export_depth_vis=False,
                    num_max_points=num_max_points,
                    show_cameras=show_cameras,
                )
                if os.path.abspath(res_path) != target_glb and os.path.exists(res_path):
                    shutil.copy2(res_path, target_glb)
                    return target_glb
                return res_path

            if image_rgb is not None and depth_norm is not None:
                import trimesh

                orig_h, orig_w = depth_norm.shape
                aspect = orig_h / orig_w
                mesh_w = grid_size
                mesh_h = max(20, int(grid_size * aspect))

                d_small = cv2.resize(depth_norm, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)
                rgb_small = cv2.resize(image_rgb, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)

                xs = np.linspace(-1.0, 1.0, mesh_w)
                ys = np.linspace(aspect, -aspect, mesh_h)
                xx, yy = np.meshgrid(xs, ys)

                dist = 1.0 / (0.15 + 0.85 * np.clip(d_small, 0.01, 1.0)) * depth_scale * 1.5
                px = xx * dist
                py = yy * dist
                pz = -dist
                verts = np.column_stack([px.ravel(), py.ravel(), pz.ravel()]).astype(np.float32)

                rgba = np.ones((verts.shape[0], 4), dtype=np.uint8) * 255
                rgba[:, :3] = rgb_small.reshape(-1, 3)

                faces = []
                for r in range(mesh_h - 1):
                    for c in range(mesh_w - 1):
                        idx = r * mesh_w + c
                        faces.append([idx, idx + 1, idx + mesh_w])
                        faces.append([idx + 1, idx + mesh_w + 1, idx + mesh_w])
                faces_arr = np.array(faces, dtype=np.int64)

                mesh = trimesh.Trimesh(vertices=verts, faces=faces_arr, vertex_colors=rgba, process=False)
                scene = trimesh.Scene(mesh)
                scene.export(target_glb)
                return target_glb

            return None
        except Exception as e:
            print(f"[-] Помилка експорту GLB карти приміщення: {e}")
            return None

    def process_video(
        self,
        video_input_path: str,
        video_output_path: str,
        model_name: str = DEFAULT_MODEL_KEY,
        colormap: str = "turbo",
        layout: str = "side_by_side",
        invert: bool = False,
        max_duration_sec: Optional[float] = None,
        target_height: Optional[int] = 480,
        frame_stride: int = 1,
        target_fps: Optional[float] = None,
        temporal_smoothing: float = 0.2,
        focus_depth: float = 0.6,
        blur_strength: int = 15,
        export_3d_dir: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Обробляє відеофайл покадрово:
        - генерує просторову карту глибини для кожного кадру зі згладжуванням між кадрами (anti-flicker EMA)
        - кодує у формат H.264 MP4 для сумісності з будь-яким веб-браузером
        - підтримує компонування: side_by_side (поруч), depth_only (тільки глибина), stacked (верх/низ),
          overlay (накладання), bokeh (кінематографічне розмиття фону)
        - опціонально будує 3D сцену (.glb, .obj) з ключових кадрів відео
        """
        import time
        import imageio

        self.load_model(model_name)

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise ValueError(f"Не вдалося відкрити вхідний відеофайл: {video_input_path}")

        in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Визначаємо кількість кадрів для обробки
        if max_duration_sec and max_duration_sec > 0:
            max_frames_limit = int(max_duration_sec * in_fps)
            total_target_frames = min(total_frames, max_frames_limit) if total_frames > 0 else max_frames_limit
        else:
            total_target_frames = total_frames if total_frames > 0 else 0

        stride = max(1, int(frame_stride))
        if target_fps and target_fps > 0:
            out_fps = float(target_fps)
        else:
            out_fps = max(1.0, float(in_fps) / float(stride))

        # Розрахунок геометрії кадрів
        if target_height and target_height > 0 and orig_h > target_height:
            scale = target_height / float(orig_h)
            frame_h = int(target_height)
            frame_w = int(orig_w * scale)
        else:
            frame_h = orig_h
            frame_w = orig_w

        # Парні розміри для кодека H.264
        frame_w = frame_w if frame_w % 2 == 0 else frame_w - 1
        frame_h = frame_h if frame_h % 2 == 0 else frame_h - 1

        out_dir = os.path.dirname(os.path.abspath(video_output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        writer = imageio.get_writer(
            video_output_path,
            fps=out_fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=1,
            output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
        )

        frame_idx = 0
        processed_count = 0
        t_start = time.time()
        final_shape = None
        ema_depth: Optional[np.ndarray] = None
        alpha = float(np.clip(temporal_smoothing, 0.0, 0.85))

        keyframe_rgb: Optional[np.ndarray] = None
        keyframe_depth: Optional[np.ndarray] = None
        keyframe_pred: Optional[Any] = None
        keyframes_pil: List[Image.Image] = []
        est_total_processed = max(1, (total_target_frames // stride) if total_target_frames > 0 else 30)
        keyframe_step = max(1, est_total_processed // 4)

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if total_target_frames > 0 and frame_idx >= total_target_frames:
                    break

                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if (frame_w, frame_h) != (orig_w, orig_h):
                    rgb = cv2.resize(rgb, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

                pil_img = Image.fromarray(rgb)
                res = self.estimate_depth(pil_img, model_name=model_name, invert=invert)
                depth_norm = res["depth_norm"]

                # Часове згладжування глибини між сусідніми кадрами (усунення мерехтіння)
                if alpha > 0.0 and ema_depth is not None and ema_depth.shape == depth_norm.shape:
                    depth_norm = (1.0 - alpha) * depth_norm + alpha * ema_depth
                ema_depth = depth_norm.copy()

                # Зберігаємо ключовий кадр для 3D реконструкції
                if export_3d_dir is not None:
                    if keyframe_rgb is None or processed_count == est_total_processed // 2:
                        keyframe_rgb = rgb.copy()
                        keyframe_depth = depth_norm.copy()
                        keyframe_pred = res.get("prediction")
                    if processed_count % keyframe_step == 0 and len(keyframes_pil) < 4:
                        keyframes_pil.append(pil_img)

                if layout == "bokeh":
                    composed = self.apply_bokeh(
                        rgb,
                        depth_norm,
                        focus_depth=focus_depth,
                        depth_of_field=0.15,
                        blur_strength=blur_strength,
                    )
                else:
                    colored_depth = self.colorize_depth(depth_norm, colormap=colormap)
                    if layout == "side_by_side":
                        divider = np.full((frame_h, 4, 3), 40, dtype=np.uint8)
                        composed = np.hstack([rgb, divider, colored_depth])
                    elif layout == "stacked":
                        divider = np.full((4, frame_w, 3), 40, dtype=np.uint8)
                        composed = np.vstack([rgb, divider, colored_depth])
                    elif layout == "overlay":
                        composed = cv2.addWeighted(rgb, 0.5, colored_depth, 0.5, 0)
                    else:  # depth_only
                        composed = colored_depth

                comp_h, comp_w = composed.shape[:2]
                if comp_w % 2 != 0 or comp_h % 2 != 0:
                    new_w = comp_w if comp_w % 2 == 0 else comp_w - 1
                    new_h = comp_h if comp_h % 2 == 0 else comp_h - 1
                    composed = composed[:new_h, :new_w]

                writer.append_data(composed)
                final_shape = composed.shape
                processed_count += 1
                frame_idx += 1

                if progress_callback is not None:
                    denom = max(1, (total_target_frames // stride) if total_target_frames > 0 else processed_count)
                    progress_callback(processed_count, denom)

        finally:
            cap.release()
            writer.close()

        glb_path = None
        obj_path = None
        if export_3d_dir is not None and keyframe_rgb is not None and keyframe_depth is not None:
            os.makedirs(export_3d_dir, exist_ok=True)
            multi_pred = keyframe_pred
            if self.da3_model is not None and len(keyframes_pil) > 1:
                try:
                    with torch.no_grad():
                        multi_pred = self.da3_model.inference(keyframes_pil)
                except Exception:
                    multi_pred = keyframe_pred

            glb_target = os.path.join(export_3d_dir, "video_scene_3d.glb")
            glb_path = self.export_scene_glb(
                multi_pred,
                glb_target,
                num_max_points=400000,
                image_rgb=keyframe_rgb,
                depth_norm=keyframe_depth,
            )
            obj_target = os.path.join(export_3d_dir, "video_mesh_3d.obj")
            obj_path = self.export_3d_mesh_obj(
                keyframe_rgb,
                keyframe_depth,
                obj_target,
                grid_size=150,
                mode="room",
            )

        elapsed = time.time() - t_start
        fps_speed = processed_count / elapsed if elapsed > 0 else 0

        return {
            "output_path": os.path.abspath(video_output_path),
            "frames_processed": processed_count,
            "elapsed_sec": round(elapsed, 2),
            "fps_speed": round(fps_speed, 1),
            "output_fps": round(out_fps, 1),
            "resolution": (final_shape[1], final_shape[0]) if final_shape else (frame_w, frame_h),
            "glb_path": glb_path,
            "obj_path": obj_path,
        }
