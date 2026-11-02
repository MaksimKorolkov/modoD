import sys
from pathlib import Path

# Забезпечуємо коректне кодування для Windows консолі
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageDraw

from depth_engine import DepthAnythingV2Engine


def create_sample_image(output_path: str = "sample_photo.jpg") -> str:
    """Створює синтетичне тестове зображення з об'єктами на різній глибині"""
    w, h = 640, 480
    img = Image.new("RGB", (w, h), color=(180, 210, 240))
    draw = ImageDraw.Draw(img)

    # Задній план (стіна або гори)
    draw.rectangle([0, 280, w, h], fill=(100, 160, 110))
    draw.polygon([(80, 280), (220, 140), (360, 280)], fill=(120, 130, 150))
    draw.polygon([(260, 280), (420, 120), (580, 280)], fill=(100, 110, 130))

    # Сонце
    draw.ellipse([480, 50, 560, 130], fill=(255, 230, 100))

    # Середній план (будиночок)
    draw.rectangle([120, 230, 240, 330], fill=(190, 100, 70))
    draw.polygon([(100, 230), (180, 170), (260, 230)], fill=(130, 50, 40))

    # Ближній план (великий м'яч/куля на траві)
    draw.ellipse([340, 260, 480, 400], fill=(230, 60, 60))
    # Тінь та блік на кулі
    draw.ellipse([370, 280, 410, 320], fill=(255, 150, 150))

    img.save(output_path, quality=95)
    print(f"[✓] Створено тестове зображення: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Тестування конвеєра Depth Anything V2...")
    print("=" * 60)

    # 1. Створення тестового фото
    sample_file = create_sample_image("sample_photo.jpg")
    img = Image.open(sample_file)

    # 2. Ініціалізація рушія
    print("[*] Ініціалізація DepthAnythingV2Engine...")
    engine = DepthAnythingV2Engine()
    info = engine.get_device_info()
    print(f"    Пристрій: {info['device']}, GPU: {info['gpu_name']}")

    # 3. Оцінка глибини Depth Anything 3 (DA3)
    print("[*] Оцінка карти глибини (модель Depth Anything 3 Small)...")
    res = engine.estimate_depth(img, model_name="da3-small")
    d_norm = res["depth_norm"]
    rgb = res["original_rgb"]
    d_u16 = res["depth_uint16"]
    print(f"    [OK] DA3 нормалізована глибина min={d_norm.min():.3f}, max={d_norm.max():.3f}")

    test_out = Path("./test_output")
    test_out.mkdir(parents=True, exist_ok=True)

    # 4. Колірна карта
    print("[*] Створення кольорової карти (Inferno)...")
    color_map = engine.colorize_depth(d_norm, colormap="inferno")
    Image.fromarray(color_map).save(test_out / "depth_inferno.png")

    # 5. 16-бітна карта
    import cv2
    cv2.imwrite(str(test_out / "depth_raw16.png"), d_u16)

    # 6. Боке
    print("[*] Симуляція розмиття фону (Bokeh)...")
    bokeh_img = engine.apply_bokeh(rgb, d_norm, focus_depth=0.8, depth_of_field=0.15, blur_strength=15)
    Image.fromarray(bokeh_img).save(test_out / "sample_bokeh.jpg")

    # 7. 3D Wiggle GIF
    print("[*] Створення 3D Wiggle GIF...")
    engine.create_wiggle_gif(rgb, d_norm, str(test_out / "sample_wiggle.gif"))

    # 8. 3D хмара точок (.ply)
    print("[*] Експорт 3D хмари точок (.ply)...")
    engine.export_pointcloud_ply(rgb, d_norm, str(test_out / "sample_pointcloud.ply"), max_points=50000)

    # 9. 3D меш (.obj)
    print("[*] Експорт 3D полігонального мешу (.obj)...")
    engine.export_3d_mesh_obj(rgb, d_norm, str(test_out / "sample_mesh.obj"), grid_size=120)

    # 10. 3D сцена (.glb)
    print("[*] Експорт 3D сцени (.glb)...")
    engine.export_scene_glb(
        res.get("prediction"),
        str(test_out / "sample_scene.glb"),
        image_rgb=rgb,
        depth_norm=d_norm,
    )

    print("\n" + "=" * 60)
    print(f"[✓] Всі тести успішно пройдено! Результати у папці:\n    {test_out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
