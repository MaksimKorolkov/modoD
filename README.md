# 🗺️ Depth Anything 3, V2 & MiDaS — Студія Монокулярної Карти Глибини

Автономний проект для отримання високоточних 3D карт глибини зі звичайних 2D фотографій за допомогою сучасних нейромереж: **Depth Anything 3 (DA3)**, **Depth Anything V2**, **Depth Anything V1** та **MiDaS**.

Проект оптимізовано під апаратне прискорення **NVIDIA CUDA** (перевірено на GeForce GTX 1660, 6GB VRAM) з можливістю автоматичного перемикання на CPU.

---

## 🚀 Швидкий старт

### 1. Графічний веб-інтерфейс (Web UI)
Запустіть додаток у терміналі:
```powershell
python app.py
```
Веб-інтерфейс автоматично відкриється у браузері за адресою:  
👉 **[http://127.0.0.1:7860](http://127.0.0.1:7860)**

### 2. Консольний режим (CLI)

```powershell
# 1. Базова карта глибини з новою моделлю Depth Anything 3 Small (за замовчуванням):
python process_photo.py --image photo.jpg

# 2. Високоточна модель Depth Anything 3 Base:
python process_photo.py --image photo.jpg --model da3-base

# 3. Флагманська модель Depth Anything 3 Large:
python process_photo.py --image photo.jpg --model da3-large

# 4. Метрична оцінка глибини Depth Anything 3 Metric:
python process_photo.py --image photo.jpg --model da3-metric

# 5. Моделі Depth Anything V2 (Small / Base / Large):
python process_photo.py --image photo.jpg --model v2-base

# 6. Використання моделі MiDaS v2.1 Small:
python process_photo.py --image photo.jpg --model midas-small

# 7. Повна обробка: палітра Turbo + портретне боке + 3D Wiggle GIF + 3D меш (.obj) та хмара (.ply):
python process_photo.py --image photo.jpg --model da3-base --colormap turbo --bokeh --wiggle --pointcloud --mesh

# 8. Пакетна обробка цілої папки фотографій:
python process_photo.py --input-dir ./my_photos --output-dir ./results --colormap inferno

# 9. Перевірка місця збереження та розміру кешу моделей на ПК:
python process_photo.py --cache-info
```

---

## 🧠 Доступні моделі та порівняння

Ви можете перемикати моделі у випадаючому списку веб-інтерфейсу або через аргумент `--model`:

| Сімейство | Назва в програмі | CLI ключ | Параметри / Розмір | Швидкість (GTX 1660) | Особливості |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Depth Anything 3** | `Depth Anything 3 - Small` *(за замовчуванням)* | `da3-small` або `da3` | ~34M (~135 МБ) | **~25 мс (надшвидка)** | **Нове покоління ByteDance Seed**: уніфікований простір, чітка геометрія. |
| **Depth Anything 3** | `Depth Anything 3 - Base` | `da3-base` | ~115M (~460 МБ) | **~75 мс (збалансована)** | **Рекомендовано**: найвища збалансованість точності та просторової узгодженості. |
| **Depth Anything 3** | `Depth Anything 3 - Large` | `da3-large` | ~400M (~1.6 ГБ) | **~220 мс** | Топове монокулярне розрізнення та ультра-деталізація мікроструктур. |
| **Depth Anything 3** | `Depth Anything 3 - Mono Large` | `da3-mono` | ~300M (~1.2 ГБ) | **~190 мс** | Спеціалізована для одиночних складних сцен. |
| **Depth Anything 3** | `Depth Anything 3 - Metric Large` | `da3-metric` | ~300M (~1.2 ГБ) | **~190 мс** | Справжні метричні відстані для точних інженерних та 3D завдань. |
| **Depth Anything 3** | `Depth Anything 3 - Nested Giant Large` | `da3-giant` | ~1.4B (~5.6 ГБ) | ~800 мс | Ультра-флагман з максимально можливим рівнем просторової точності. |
| **Depth Anything V2** | `Depth Anything V2 - Small` | `v2-small` або `small` | ~25M (~98 МБ) | **~20 мс** | Перевірена швидка класика для великих архівів. |
| **Depth Anything V2** | `Depth Anything V2 - Base` | `v2-base` або `base` | ~97M (~380 МБ) | **~60 мс** | Відмінний баланс чіткості тонких контурів. |
| **Depth Anything V2** | `Depth Anything V2 - Large` | `v2-large` або `large` | ~335M (~1.3 ГБ) | **~180 мс** | Висока точність другого покоління. |
| **Depth Anything V1** | `Depth Anything V1 - Small` | `v1-small` | ~25M (~98 МБ) | ~20 мс | Перша версія Depth Anything. |
| **Depth Anything V1** | `Depth Anything V1 - Base` | `v1-base` | ~97M (~380 МБ) | ~60 мс | Перша версія Base. |
| **MiDaS** | `MiDaS v2.1 - Small` | `midas-small` або `midas` | ~21M (~81 МБ) | ~25 мс | Класична легка модель MiDaS (швидка, плавна). |
| **MiDaS** | `MiDaS v3.0 - DPT Hybrid` | `midas-hybrid` | ~120M (~470 МБ) | ~70 мс | Гібридний трансформер MiDaS 3.0. |
| **MiDaS** | `MiDaS v3.0 - DPT Large` | `midas-large` | ~345M (~1.3 ГБ) | ~190 мс | Повнорозмірний DPT трансформер MiDaS 3.0. |

---

## 💾 Збереження моделей на комп'ютері (Офлайн-режим)

### Чи зберігаються моделі на ПК?
**Так, повністю.** Моделі завантажуються лише **один раз** при першому виборі, після чого назавжди залишаються на вашому накопичувачі. Програма працює **на 100% офлайн** без підключення до інтернету.

### Де вони розташовані за замовчуванням:
1. **Depth Anything (Hugging Face)**:
   ```
   C:\Users\<Ваш_Користувач>\.cache\huggingface\hub\
   ```
   *Папки: `models--depth-anything--Depth-Anything-V2-Small-hf`, тощо.*

2. **MiDaS (PyTorch Hub)**:
   ```
   C:\Users\<Ваш_Користувач>\.cache\torch\hub\checkpoints\
   ```
   *Файли: `midas_v21_small_256.pt`, `dpt_hybrid_384.pt`, тощо.*

### Як подивитися зайняте місце:
Запустіть:
```powershell
python process_photo.py --cache-info
```

### 📦 Портативний режим (збереження моделей у папку проекту):
Якщо ви бажаєте, щоб моделі завантажувалися прямо у папку проекту `modoD\models` (наприклад, для переносу на флешці або економії місця на диску C:):
```powershell
$env:HF_HOME = "c:\Users\m.korolkov\Desktop\modoD\models\huggingface"
$env:TORCH_HOME = "c:\Users\m.korolkov\Desktop\modoD\models\torch"
python app.py
```

---

## 🌟 Можливості системи

* **Інтерактивний слайдер порівняння (Image Slider)**: плавне перетягування межі між оригіналом та картою глибини.
* **10 колірних палітр**: `inferno`, `turbo`, `viridis`, `magma`, `plasma`, `spectral`, `cividis`, `twilight`, `gist_earth`, `gray`.
* **16-бітний PNG (RAW uint16)**: глибина з градацією у 65 536 рівнів для професійного 3D моделювання та VFX (Blender, Cinema 4D, Unreal Engine).
* **Симуляція оптичного боке (Portrait DoF)**: реалістичне розмиття переднього або заднього плану з вибором точки фокусу та радіуса апертури.
* **3D Wiggle GIF анімація**: стереоскопічний паралакс-ефект об'єму («живе 3D фото»).
* **3D Хмара точок (.ply)**: хмара точок з RGB текстурою для MeshLab, CloudCompare, Blender.
* **3D Меш (.obj) з інтерактивним переглядом**: обертання та наближення 3D моделі мишкою прямо у веб-браузері.
* **Пакетна обробка**: обробка необмеженої кількості фотографій з автоматичним пакуванням у **ZIP-архів**.

---

## 🎨 Покрокова інструкція: 3D рельєф у Blender з 16-бітної карти

1. Відкрийте **Blender** і видаліть куб за замовчуванням.
2. Створіть площину: `Shift + A` -> **Mesh** -> **Plane**.
3. Додайте модифікатор **Subdivision Surface**:
   - Режим: **Simple**.
   - Рівні (Levels Viewport / Render): встановіть `6` або `7` для високої деталізації сітки.
4. Додайте модифікатор **Displace**:
   - Натисніть кнопку **New** (створити нову текстуру).
   - Перейдіть у вкладку текстур (значок шахівниці знизу).
   - Натисніть **Open** і виберіть файл `depth_raw_16bit.png` (або `*_depth_raw16.png`).
   - У налаштуванні **Color Space** обов'язково встановіть: `Non-Color`.
5. Поверніться до налаштувань модифікатора **Displace**:
   - Зменшіть параметр **Strength** (наприклад, до `0.1` – `0.3`).
6. За бажанням додайте матеріал із базовою текстурою оригінального кольорового фото.
7. **Результат:** фотографія перетворюється на об'ємний 3D барельєф!

---

## 🗂️ Структура проекту

```
modoD/
├── app.py              # Інтерактивна веб-студія (Gradio UI з 3D переглядачем)
├── depth_engine.py     # Ядро нейромереж: Depth Anything V2/V1, MiDaS, 3D експорт, боке
├── process_photo.py    # Консольний скрипт (CLI) для пакетної та одиночної обробки
├── test_pipeline.py    # Тестовий скрипт верифікації конвеєра
├── requirements.txt    # Залежності Python
├── sample_photo.jpg    # Тестове зображення
├── test_output/        # Результати тестування
├── cli_output/         # Результати консольного тестування
└── README.md           # Документація проекту
```

---

## 💻 Системні вимоги

* **Операційна система**: Windows 10 / 11
* **Python**: 3.10+
* **GPU**: Будь-яка відеокарта NVIDIA з підтримкою CUDA (GTX 1060 / 1660 / RTX 2060 / 3060 / 4060 тощо) або центральний процесор (CPU).
* **VRAM**:
  - Для моделей **Small** (Depth Anything V2 / MiDaS): від 1 ГБ VRAM
  - Для моделей **Base**: від 2 ГБ VRAM
  - Для моделей **Large**: від 4 ГБ VRAM
