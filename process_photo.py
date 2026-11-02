"""
process_photo.py
----------------
Консольний скрипт для пакетної або одиночної обробки фотографій
за допомогою нейромережі Depth Anything V2.

Приклади використання:
  python process_photo.py --image test.jpg --pointcloud --bokeh --wiggle
  python process_photo.py --image portrait.png --colormap turbo --model base
  python process_photo.py --input-dir ./my_photos --output-dir ./results --colormap viridis
"""

import os
import sys
from pathlib import Path

# Забезпечуємо коректне кодування для Windows консолі
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import glob
import argparse
from PIL import Image
import cv2
import numpy as np

from depth_engine import DepthAnythingV2Engine, MODELS, COLORMAPS, DEFAULT_MODEL_KEY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Генерація монокулярних карт глибини за допомогою Depth Anything V2"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-i", "--image", type=str, help="Шлях до вхідного зображення (jpg, png, webp...)"
    )
    group.add_argument(
        "-v", "--video", type=str, help="Шлях до вхідного відеофайлу (mp4, mov, avi, webm...)"
    )
    group.add_argument(
        "-d", "--input-dir", type=str, help="Папка із зображеннями для пакетної обробки"
    )
    group.add_argument(
        "--cache-info",
        action="store_true",
        help="Показати інформацію про кеш моделей на диску (шляхи та обсяг)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./output",
        help="Папка для збереження результатів (за замовчуванням ./output)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="da3-small",
        choices=[
            "da3-small", "da3-base", "da3-large", "da3-mono", "da3-metric", "da3-giant", "da3",
            "v2-small", "v2-base", "v2-large",
            "v1-small", "v1-base",
            "midas-small", "midas-hybrid", "midas-large",
            "small", "base", "large", "midas"
        ],
        help="Вибір моделі: da3-small/base/large/mono/metric/giant (Depth Anything 3), v2-small/base/large, midas-*",
    )
    parser.add_argument(
        "-c",
        "--colormap",
        type=str,
        default="inferno",
        choices=COLORMAPS,
        help="Колірна схема для візуалізації глибини",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Інвертувати карту глибини (0 замість 1 для близьких об'єктів)",
    )
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Згенерувати кольорову 3D хмару точок у форматі .ply",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Згенерувати 3D полігональний меш у форматі .obj",
    )
    parser.add_argument(
        "--bokeh",
        action="store_true",
        help="Згенерувати фото з портретним розмиттям фону (Depth-of-Field)",
    )
    parser.add_argument(
        "--focus-depth",
        type=float,
        default=0.6,
        help="Глибина фокусування для боке (0.0 - найдальше, 1.0 - найближче)",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=15,
        help="Радіус розмиття для боке",
    )
    parser.add_argument(
        "--wiggle",
        action="store_true",
        help="Згенерувати анімований 3D стерео-паралакс (GIF)",
    )
    parser.add_argument(
        "--raw16",
        action="store_true",
        default=True,
        help="Зберегти точну 16-бітну карту глибини (PNG) для Blender/VFX",
    )
    parser.add_argument(
        "--glb",
        action="store_true",
        help="Згенерувати 3D карту приміщення / сцену у форматі .glb (для DA3)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="side_by_side",
        choices=["side_by_side", "depth_only", "stacked", "overlay", "bokeh"],
        help="Режим компонування відео (side_by_side, depth_only, stacked, overlay, bokeh)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Обмеження тривалості відео в секундах (0 = все відео)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Вихідна висота кадру відео в пікселях (0 = оригінальна)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Крок кадрів для відео (1 = кожен кадр, 2 = кожен другий)",
    )

    return parser.parse_args()


def process_single_file(
    engine: DepthAnythingV2Engine,
    img_path: Path,
    out_dir: Path,
    args,
    model_key: str,
):
    print(f"\n[+] Обробка: {img_path.name}")
    try:
        image = Image.open(img_path)
    except Exception as e:
        print(f"[-] Помилка відкриття файлу {img_path}: {e}")
        return

    # Передбачення глибини
    res = engine.estimate_depth(image, model_name=model_key, invert=args.invert)
    depth_norm = res["depth_norm"]
    orig_rgb = res["original_rgb"]
    depth_uint16 = res["depth_uint16"]

    stem = img_path.stem

    # 1. Колірна карта глибини
    color_depth = engine.colorize_depth(depth_norm, colormap=args.colormap)
    color_path = out_dir / f"{stem}_depth_{args.colormap}.png"
    Image.fromarray(color_depth).save(color_path)
    print(f"  -> Збережено кольорову карту: {color_path.name}")

    # 2. 16-бітна карта для 3D/VFX
    if args.raw16:
        raw_path = out_dir / f"{stem}_depth_raw16.png"
        cv2.imwrite(str(raw_path), depth_uint16)
        print(f"  -> Збережено 16-бітну карту: {raw_path.name}")

    # 3. Боке / портретний режим
    if args.bokeh:
        bokeh_img = engine.apply_bokeh(
            orig_rgb,
            depth_norm,
            focus_depth=args.focus_depth,
            depth_of_field=0.15,
            blur_strength=args.blur_radius,
        )
        bokeh_path = out_dir / f"{stem}_bokeh.jpg"
        Image.fromarray(bokeh_img).save(bokeh_path, quality=95)
        print(f"  -> Збережено фото з боке: {bokeh_path.name}")

    # 4. 3D Wiggle GIF
    if args.wiggle:
        gif_path = out_dir / f"{stem}_wiggle3d.gif"
        engine.create_wiggle_gif(orig_rgb, depth_norm, str(gif_path))
        print(f"  -> Збережено 3D анімацію: {gif_path.name}")

    # 5. Хмара точок (.ply)
    if args.pointcloud:
        ply_path = out_dir / f"{stem}_pointcloud.ply"
        engine.export_pointcloud_ply(orig_rgb, depth_norm, str(ply_path))
        print(f"  -> Збережено 3D хмару точок: {ply_path.name}")

    # 6. Меш (.obj)
    if args.mesh:
        obj_path = out_dir / f"{stem}_mesh.obj"
        engine.export_3d_mesh_obj(orig_rgb, depth_norm, str(obj_path), mode="room")
        print(f"  -> Збережено 3D меш: {obj_path.name}")

    # 7. 3D Карта приміщення / Сцена (.glb)
    if args.glb:
        glb_path = out_dir / f"{stem}_room_scene.glb"
        engine.export_scene_glb(
            res.get("prediction"),
            str(glb_path),
            image_rgb=orig_rgb,
            depth_norm=depth_norm,
        )
        print(f"  -> Збережено 3D карту приміщення: {glb_path.name}")


def main():
    args = parse_args()

    # Якщо користувач запросив інформацію про збережені моделі
    if args.cache_info:
        paths = DepthAnythingV2Engine.get_cache_paths()
        print("=" * 65)
        print("  ІНФОРМАЦІЯ ПРО МІСЦЕ ЗБЕРЕЖЕННЯ МОДЕЛЕЙ (CACHE)")
        print("=" * 65)
        print(f"\n1. Кеш Hugging Face (Depth Anything V1/V2):")
        print(f"   Папка: {paths['huggingface_cache']}")
        print(f"   Загальний розмір: {paths['huggingface_size_mb']} MB")
        print(f"   Завантажені моделі:")
        for m in paths['huggingface_downloaded']:
            print(f"     - {m}")

        print(f"\n2. Кеш PyTorch Hub (MiDaS):")
        print(f"   Папка: {paths['torch_cache']}")
        print(f"   Загальний розмір: {paths['torch_size_mb']} MB")
        print(f"   Завантажені чекпоінти:")
        for chk in paths['torch_checkpoints']:
            print(f"     - {chk}")
        print("\n" + "=" * 65)
        return

    if not args.image and not args.video and not args.input_dir:
        print("Помилка: необхідно вказати --image <файл>, або --video <відео>, або --input-dir <папка>, або --cache-info")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Повна таблиця моделей
    model_map = {
        # Depth Anything 3 (DA3)
        "da3": "Depth Anything 3 - Small (~34M, надшвидка)",
        "da3-small": "Depth Anything 3 - Small (~34M, надшвидка)",
        "da3-base": "Depth Anything 3 - Base (~115M, збалансована точність)",
        "da3-large": "Depth Anything 3 - Large (~400M, максимальна якість)",
        "da3-mono": "Depth Anything 3 - Mono Large (~300M, монокулярна)",
        "da3-metric": "Depth Anything 3 - Metric Large (~300M, метрична глибина)",
        "da3-giant": "Depth Anything 3 - Nested Giant Large (~1.4B, ультра-якість)",
        # Depth Anything V2
        "v2-small": "Depth Anything V2 - Small (~25M, найшвидша)",
        "v2-base": "Depth Anything V2 - Base (~97M, відмінна чіткість)",
        "v2-large": "Depth Anything V2 - Large (~335M, висока точність)",
        "small": "Depth Anything V2 - Small (~25M, найшвидша)",
        "base": "Depth Anything V2 - Base (~97M, відмінна чіткість)",
        "large": "Depth Anything V2 - Large (~335M, висока точність)",
        # Depth Anything V1
        "v1-small": "Depth Anything V1 - Small (~25M)",
        "v1-base": "Depth Anything V1 - Base (~97M)",
        # MiDaS
        "midas": "MiDaS v2.1 - Small (~21M, класична)",
        "midas-small": "MiDaS v2.1 - Small (~21M, класична)",
        "midas-hybrid": "MiDaS v3.0 - DPT Hybrid (~120M)",
        "midas-large": "MiDaS v3.0 - DPT Large (~345M)",
    }
    model_key = model_map.get(args.model, DEFAULT_MODEL_KEY)

    print("=" * 60)
    print("  Depth Anything 3 / V2 / MiDaS — Обробка монокулярної глибини")
    print("=" * 60)

    engine = DepthAnythingV2Engine()
    info = engine.get_device_info()
    print(f"Пристрій: {info['device']} ({info['gpu_name']})")
    if "vram_total_mb" in info:
        print(f"VRAM: {info['vram_total_mb']} MB")
    print(f"Обрана модель: {model_key}")
    print(f"Палітра кольорів: {args.colormap}")
    print(f"Папка виводу: {out_dir.resolve()}")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Помилка: файл {img_path} не знайдено!")
            sys.exit(1)
        process_single_file(engine, img_path, out_dir, args, model_key)

    elif args.video:
        vid_path = Path(args.video)
        if not vid_path.exists():
            print(f"Помилка: відеофайл {vid_path} не знайдено!")
            sys.exit(1)
        out_vid = out_dir / f"{vid_path.stem}_depth_{args.layout}.mp4"
        print(f"\n[+] Обробка відеофайлу: {vid_path.name} (layout={args.layout})...")
        stats = engine.process_video(
            video_input_path=str(vid_path),
            video_output_path=str(out_vid),
            model_name=model_key,
            colormap=args.colormap,
            layout=args.layout,
            invert=args.invert,
            max_duration_sec=args.max_seconds if args.max_seconds > 0 else None,
            target_height=args.height if args.height > 0 else None,
            frame_stride=args.stride,
            focus_depth=args.focus_depth,
            blur_strength=args.blur_radius,
            export_3d_dir=str(out_dir) if (args.glb or args.mesh) else None,
        )
        print(f"  -> Збережено відео: {stats['output_path']}")
        print(f"  -> Кадрів: {stats['frames_processed']} | Час: {stats['elapsed_sec']} с ({stats['fps_speed']} кадр/с)")
        if stats.get("glb_path"):
            print(f"  -> Збережено 3D сцену (.glb): {stats['glb_path']}")
        if stats.get("obj_path"):
            print(f"  -> Збережено 3D меш (.obj): {stats['obj_path']}")

    elif args.input_dir:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            print(f"Помилка: папка {in_dir} не знайдена!")
            sys.exit(1)

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in extensions:
            files.extend(in_dir.glob(ext))
            files.extend(in_dir.glob(ext.upper()))

        # Видаляємо дублікати
        files = sorted(list(set(files)))
        if not files:
            print(f"У папці {in_dir} не знайдено підтримуваних зображень.")
            sys.exit(0)

        print(f"Знайдено зображень для обробки: {len(files)}")
        for idx, f in enumerate(files, 1):
            print(f"[{idx}/{len(files)}]", end=" ")
            process_single_file(engine, f, out_dir, args, model_key)

    print("\n" + "=" * 60)
    print(f"Успішно завершено! Всі результати збережено в:\n{out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
