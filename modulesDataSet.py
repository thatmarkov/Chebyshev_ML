import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift
from scipy.special import eval_jacobi
from skimage.draw import disk, ellipse
import pandas as pd
from PIL import Image
import json
import numpy as np
from PIL import Image
import os
"""
-----------------------------------------------
1 БЛОК: ИНТЕГРАЛЬНЫЕ ОПТИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ
-----------------------------------------------
detectCamera(x) - регистрация квадрата поля (интенсивность)
FT(x) - Быстрое Фурье с учётом сдвига
iFT(x) - обратное Фурье
FresnelPropagator(E0, ps, lambda0, z) - расчёт поля E0 с пикселем ps и длиной волны lambda0 на расстояние z
load_beam_info_image(i, folder_path, device='cuda') - загрузка пучков 
load_mask_info(i, folder_path, device='cuda') - загрузка масок
difraction_batch(indices, masks, device='cuda', pixelsizeSLM = 32e-6, lam = 532e-9, z = 0.2, folder_path = './dataset/beams'): - батчевый расчёт трехканальных картин дифракции
----------------------------------------------
2 БЛОК: ГЕНЕРАЦИЯ ЭЛЕМЕНТОВ ДАТАСЕТА ЦЕРНИКЕ. Сейчас не используется, поскольку полиномы Цернике не ортогональны на квадрате. Пережиток старой версии 
----------------------------------------------
generate_aperture(image_size=1024, center=None, radius=40, softness=2, irregularity=0.0) - генерация амплитудных профилей пучка      <-- Амплитудный профиль
zernike_polynomial(n, m, rho, theta) - Вычисляет полином Цернике (n, m) в точке (rho, theta).
generate_random_zernike_coefficients() - Генерирует случайные коэффициенты для полиномов Цернике.
zernike_coefficients_to_text(coefficients) - Преобразует коэффициенты Цернике в читаемый текст (для визуализации!)
generate_phase_aberrations(amplitude_mask, coefficients, center) - Генерирует фазовые аберрации на основе полиномов Цернике.         <-- Фазовый профиль
generate_random_beam_sample(image_size=256) - Генерирует один полностью случайный sample амплитуды и фазы.
create_dataset(num_samples=10000, image_size=256) - Создает полный датасет со случайными параметрами.                             <-- Метадата
visualize_random_samples(num_samples=5) - Визуализирует несколько случайных samples


----------------------------------------------
3 БЛОК: ГЕНЕРАЦИЯ ЭЛЕМЕНТОВ ДАТАСЕТА ЧЕБЫШЁВА
----------------------------------------------
chebyshev_poly(n, x): - Вычисляет значения полинома Чебышева первого рода T_n(x) для массива x. Используется рекуррентное соотношение.
generate_chebyshev_dataset(num_samples=1000, size=256, max_order=7,  coeff_ranges=None, seed=None): - Генерирует датасет фазовых масок и коэффициентов 2D полиномов Чебышева.
 Параметры:
        num_samples : int - число примеров
        size : int - размер квадратной сетки (size x size)
        max_order : int - максимальный суммарный порядок p+q
        coeff_ranges : list of tuple - для каждого полинома (в порядке обхода)
                       задаётся диапазон (min, max). Если None, используются значения по умолчанию.
        seed : int - для воспроизводимости
    
    Возвращает:
        phases : np.array (num_samples, size, size) - фазовые маски в радианах
        coeffs : np.array (num_samples, M) - коэффициенты a_{pq}
        poly_list : list of tuples (p, q) - порядки полиномов, соответствующие столбцам coeffs

        



-----------------------------------------------
ИСТОРИЯ ИЗМЕНЕНИЙ
-----------------------------------------------

19.06.2026 - Добавлены функции генерации к полиномам Чебышева. Цернике имеют существенные минусы.

"""



#--------------Код-------------------------------
"""
=======================================================================
Блок 1 - torch реализации (тензорные) для GPU просессинга ИОП
=======================================================================

"""
def detectCamera(x):
    return torch.real(x * torch.conj(x))

def FT(x):
    # Применяет 2D FFT по последним двум осям
    return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2,-1)), dim=(-2,-1))

def iFT(x):
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2,-1)), dim=(-2,-1))


def FresnelPropagator(E0, ps, lambda0, z, device='cuda'):
    # E0 - комплексный тензор, последние два измерения - пространственные (..., H, W)
    # Определяем размеры
    if E0.dim() < 2:
        raise ValueError("E0 must have at least 2 dimensions")
    # Получаем H, W из последних двух размеров
    ny, nx = E0.shape[-2], E0.shape[-1]
    grid_sizex = ps * nx
    grid_sizey = ps * ny

    # Частотные координаты как тензоры на нужном устройстве
    fx = torch.linspace(-(nx - 1) / 2 * (1 / grid_sizex),
                        (nx - 1) / 2 * (1 / grid_sizex), nx, device=device)
    fy = torch.linspace(-(ny - 1) / 2 * (1 / grid_sizey),
                        (ny - 1) / 2 * (1 / grid_sizey), ny, device=device)
    Fx, Fy = torch.meshgrid(fx, fy, indexing='ij')

    # Создаём мнимую единицу как тензор
    j = torch.tensor(1j, device=device)

    # Постоянная и квадратичная фаза
    const_phase = j * (2 * np.pi / lambda0) * z
    quadratic_phase = j * np.pi * lambda0 * z * (Fx**2 + Fy**2)
    H = torch.exp(const_phase) * torch.exp(quadratic_phase)  # (H, W)

    # Добавляем размерности к H для broadcasting с E0
    # E0 имеет форму (..., H, W), H имеет (H, W). Чтобы broadcasting работал,
    # нужно добавить размерности перед пространственными: H.unsqueeze(0)...
    # Общее правило: привести H к форме (1,)* (E0.dim()-2) + (H, W)
    # Проще: использовать view и expand
    for _ in range(E0.dim() - 2):
        H = H.unsqueeze(0)  # добавляем batch-измерения
    # Теперь H имеет форму (1,1,...,H,W), что будет транслироваться на (...,H,W)

    # Применяем FT (по последним двум осям)
    E0fft = torch.fft.fftshift(torch.fft.fft2(E0), dim=(-2,-1))
    G = H * E0fft
    Ef = torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2,-1)))
    return Ef


def load_beam_info_image(i, folder_path, device='cuda'):
    """
    Загружает изображение beam по индексу i, возвращает амплитуду и фазу как тензоры.
    Параметры:
        i: int - индекс изображения
        folder_path: str - путь к папке с beams
        device: str - устройство ('cuda' или 'cpu')
    Возвращает:
        amp: torch.Tensor (H, W) float, значения в [0, 1]
        phase: torch.Tensor (H, W) float, значения в [-π, π]
    """
    filename = f'beam_{i:06d}.png'
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден")

    img = Image.open(filepath)
    img_array = np.array(img)  # (H, W, 3) uint8

    # Извлекаем каналы и нормируем
    amp_uint8 = img_array[..., 0]  # (H, W) uint8
    phase_uint8 = img_array[..., 1]  # (H, W) uint8

    # Преобразуем в float и нормируем
    amp = amp_uint8.astype(np.float32) / 255.0
    phase = phase_uint8.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi

    # Переводим в тензоры и переносим на устройство
    amp_t = torch.from_numpy(amp).to(device)
    phase_t = torch.from_numpy(phase).to(device)

    return amp_t, phase_t


def load_mask_info(i, folder_path, device='cuda'):
    """
    Загружает маску по индексу i, возвращает фазовую маску как тензор.
    Параметры:
        i: int - индекс маски
        folder_path: str - путь к папке с масками
        device: str - устройство
    Возвращает:
        phase_mask: torch.Tensor (H, W) float, значения в [0, 2π]
    """
    filename = str(i) + 'mask256.png'
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден")

    img = Image.open(filepath).convert('L')  # переводим в灰度, чтобы получить один канал
    img_array = np.array(img, dtype=np.float32)  # (H, W) uint8 или float

    # Нормируем в диапазон [0, 2π]
    phase_mask = img_array / 255.0 * (2 * np.pi)

    # Преобразуем в тензор и переносим на устройство
    phase_mask_t = torch.from_numpy(phase_mask).to(device)

    return phase_mask_t



def difraction_batch(indices, masks, device='cuda', pixelsizeSLM = 32e-6, lam = 532e-9, z = 0.2, folder_path = './dataset/beams'):
    """
    Генерирует трёхканальные дифракционные изображения для списка индексов.
    Параметры:
        indices: list или 1D тензор индексов (длина B)
        masks: тензор (3, H, W) с фазовыми масками для каждого канала
        device: устройство
    Возвращает:
        torch.Tensor (B, H, W, 3) типа uint8
    """
    B = len(indices)
    # Загружаем амплитуды и фазы для всех индексов (можно улучшить, загружая параллельно)
    amps = []
    phases = []
    for idx in indices:
        amp, phase = load_beam_info_image(idx, device=device, folder_path=folder_path)
        amps.append(amp)
        phases.append(phase)
    amps = torch.stack(amps, dim=0)       # (B, H, W)
    phases = torch.stack(phases, dim=0)   # (B, H, W)

    # Расширяем маски на batch-измерение
    masks_batch = masks.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 3, H, W)

    # Расширяем амплитуды и фазы на каналы
    amps_3ch = amps.unsqueeze(1).expand(-1, 3, -1, -1)      # (B, 3, H, W)
    phases_3ch = phases.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)

    # Комплексное поле
    E0 = amps_3ch * torch.exp(1j * (phases_3ch + masks_batch))  # (B, 3, H, W) complex

    # Дифракция (пропагатор + детектор)
    propagated = FresnelPropagator(E0, pixelsizeSLM, lam, z, device=device)  # (B, 3, H, W) complex
    intensity = detectCamera(propagated)  # (B, 3, H, W) float

    # Нормировка каждого изображения и канала отдельно
    # Находим максимум для каждого (B, канал)
    max_vals = intensity.view(B, 3, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)  # (B, 3, 1, 1)
    max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
    intensity_norm = intensity / max_vals * 255   # (B, 3, H, W)

    # Приводим к uint8 и переставляем оси
    intensity_uint8 = intensity_norm.to(torch.uint8)           # (B, 3, H, W)
    img_batch = intensity_uint8.permute(0, 2, 3, 1)            # (B, H, W, 3)

    return img_batch

"""
==================================================================
Блок 2 Цернике датасет - не используется сейчас
==================================================================
"""
def generate_aperture(image_size=1024, center=None, radius=40, softness=2,
                     irregularity=0.0):
    """
    Генерирует мягкую апертуру со случайными вариациями формы.
    irregularity: уровень неровности края (0.0 = идеальный круг)
    """
    if center is None:
        center = (image_size // 2, image_size // 2)
    mask = np.zeros((image_size, image_size))

    # Создаем базовую форму
    if  irregularity == 0.0:
        # Идеальный круг
        
        rr, cc = disk(center, radius, shape=mask.shape)
        mask[rr, cc] = 1.0
    else:
        # Добавляем неровности к радиусу
        theta = np.linspace(0, 2*np.pi, 360)
        r_irregular = radius * (1 - irregularity * np.abs(np.random.randn(len(theta))))
        points = []
        for i, t in enumerate(theta):
            x = center[1] + r_irregular[i] * np.cos(t)
            y = center[0] + r_irregular[i] * np.sin(t)
            points.append((int(y), int(x)))

        # Заполняем полигон
        from skimage.draw import polygon
        rr, cc = polygon([p[0] for p in points], [p[1] for p in points])
        valid = (rr >= 0) & (rr < image_size) & (cc >= 0) & (cc < image_size)
        mask[rr[valid], cc[valid]] = 1.0
        

    # Размываем маску для создания мягкого края
    soft_mask = gaussian_filter(mask, sigma=softness)
    soft_mask = soft_mask / np.max(soft_mask)

    return soft_mask

def zernike_polynomial(n, m, rho, theta):
    """Вычисляет полином Цернике (n, m) в точке (rho, theta)."""
    if m == 0:
        return np.sqrt(n + 1) * eval_jacobi(n//2, 0, 0, 2*rho**2 - 1)
    elif m > 0:
        R = np.sqrt(2 * (n + 1)) * rho**m * eval_jacobi((n - m)//2, m, 0, 2*rho**2 - 1)
        return R * np.cos(m * theta)
    else:
        m_abs = -m
        R = np.sqrt(2 * (n + 1)) * rho**m_abs * eval_jacobi((n - m_abs)//2, m_abs, 0, 2*rho**2 - 1)
        return R * np.sin(m_abs * theta)

def generate_random_zernike_coefficients():
    """Генерирует случайные коэффициенты для полиномов Цернике."""
    coefficients = {}

    # Базовые аберрации с разной вероятностью и амплитудой
    aberrations = [
        # (n, m), вероятность, макс. амплитуда
        ((0, 0), 1.0, np.pi),   # Постоянное смещение
        ((1, 1), 0.8, np.pi),   # Наклон по X
        ((1, -1), 0.8, np.pi),  # Наклон по Y
        ((2, 0), 0.8, np.pi/2),   # Дефокус
        ((2, 2), 0.5, np.pi/2),   # Астигматизм 0°/90°
        ((2, -2), 0.5, np.pi/2),  # Астигматизм 45°
        ((3, 1), 0.4, np.pi/2),   # Кома по X 0.2 
        ((3, -1), 0.4, np.pi/2),  # Кома по Y 0.2 
        ((3, 3), 0.0, np.pi/2),   # Треугольный астигматизм 0.1 
        ((3, -3), 0.0, np.pi/2),  # Треугольный астигматизм 0.1
        ((4, 0), 0.4, np.pi/2),   # Сферическая аберрация
    ]

    for (n, m), prob, max_amp in aberrations:
        if np.random.random() < prob:
            # Случайная амплитуда с нормальным распределением
            strength = np.random.normal(0, max_amp/2)
            # Ограничиваем максимальную амплитуду
            strength = np.clip(strength, -max_amp, max_amp)
            coefficients[(n, m)] = strength

    return coefficients

def zernike_coefficients_to_text(coefficients):
    """Преобразует коэффициенты Цернике в читаемый текст."""
    names = {
        (0, 0): "Постоянное смещение фазы",
        (1, 1): "Наклон волнового фронта по X",
        (1, -1): "Наклон волнового фронта по Y",
        (2, 0): "Дефокус (расфокусировка)",
        (2, 2): "Астигматизм 0°/90°",
        (2, -2): "Астигматизм 45°",
        (3, 1): "Кома по X",
        (3, -1): "Кома по Y",
        (3, 3): "Треугольный астигматизм",
        (3, -3): "Треугольный астигматизм",
        (4, 0): "Сферическая аберрация",
    }

    text = "Аберрации волнового фронта:\n"
    text += "-----------------------------\n"

    # Сортируем по порядку n
    sorted_coeffs = sorted(coefficients.items(), key=lambda x: (x[0][0], x[0][1]))

    for (n, m), strength in sorted_coeffs:
        name = names.get((n, m), f"Неизвестная аберрация ({n}, {m})")
        text += f"{name}: {strength:.3f} рад\n"

    # Добавляем общую информацию
    rms = np.sqrt(sum(s**2 for s in coefficients.values()))
    text += f"-----------------------------\n"
    text += f"Среднеквадратичное отклонение: {rms:.3f} рад\n"
    text += f"Общее количество аберраций: {len(coefficients)}\n"

    return text

def generate_phase_aberrations(amplitude_mask, coefficients, center, beam_radius):
    """Генерирует фазовые аберрации на основе полиномов Цернике."""
    image_size = amplitude_mask.shape[0]


    # Нормированные полярные координаты относительно центра
    y, x = np.mgrid[0:image_size, 0:image_size]
    rho = np.sqrt((x - center[1])**2 + (y - center[0])**2) / beam_radius
    theta = np.arctan2(y - center[0], x - center[1])

    # Маска для области внутри апертуры
    valid_mask = (rho <= 1.0) & (amplitude_mask > 0.01)
    #plt.imshow(rho <= 1.0)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(amplitude_mask > 0.1)
    #plt.colorbar()
    #plt.show()

    # Инициализируем фазу
    phase = np.zeros_like(amplitude_mask)

    # Добавляем вклады каждого полинома Цернике
    for (n, m), strength in coefficients.items():
        zernike_val = zernike_polynomial(n, m, rho, theta)
        phase += strength * zernike_val

    # Обнуляем фазу вне области апертуры
    phase[~valid_mask] = 0

    return phase


def generate_random_beam_sample(image_size=256, 
                                center_offset_x_amp = 256//15,
                                center_offset_y_amp = 256//15,
                                radius_min = 256//4,
                                radius_max = 256//2 - 256//15,
                                softness_min = 1.5, #1.5
                                softness_max = 4.0, #4,0
                                irregularity_min = 0.0,
                                irregularity_max = 0.05,
                                gaussian_sigma_min = 256//4,
                                gaussian_sigma_max = 256//2,
                                noise_strenght_min = 0.01,
                                noise_strenght_max = 0.06
                                ):
    """Генерирует один полностью случайный sample амплитуды и фазы."""

    # # Случайные параметры апертуры
    center_offset_x = np.random.randint(-center_offset_x_amp, center_offset_x_amp + 1)
    center_offset_y = np.random.randint(-center_offset_y_amp, center_offset_y_amp + 1)
    center = (image_size//2 + center_offset_y, image_size//2 + center_offset_x)

    radius = np.random.randint(radius_min, radius_max)
    softness = np.random.uniform(softness_min, softness_max)
    irregularity = np.random.uniform(irregularity_min, irregularity_max)

    # 1. Генерируем апертуру
    aperture = generate_aperture(
        image_size=image_size,
        center=center,
        radius=radius,
        softness=softness,
        irregularity=irregularity
    )

    # 2. Добавляем амплитудные неоднородности
    gaussian_sigma = np.random.uniform(gaussian_sigma_min, gaussian_sigma_max)
    noise_strength = np.random.uniform(noise_strenght_min, noise_strenght_max)

    y, x = np.mgrid[0:image_size, 0:image_size]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    gaussian_profile = np.exp(-(r**2) / (2 * gaussian_sigma**2))


    noise = np.random.normal(1.0, noise_strength, aperture.shape)

    amplitude = aperture * gaussian_profile * noise
    amplitude = np.clip(amplitude, 0, None)
    amplitude /= np.max(amplitude)

    

    # 4. Генерируем случайные фазовые аберрации
    zernike_coeffs = generate_random_zernike_coefficients()
    phase = generate_phase_aberrations(amplitude, zernike_coeffs, center, radius)

    # Параметры для метаданных
    params = {
        'center_offset': (center_offset_x, center_offset_y),
        'radius': radius,
        'softness': float(softness),
        'irregularity': float(irregularity),
        'gaussian_sigma': float(gaussian_sigma),
        'noise_strength': float(noise_strength),
        'zernike_coefficients': {f"{n}_{m}": float(strength) for (n, m), strength in zernike_coeffs.items()}
    }
    aberration_text = zernike_coefficients_to_text(zernike_coeffs)

    return amplitude, phase, params, aberration_text


def normalize_phase(phase):
    phase = phase.copy()
    while np.any(phase > np.pi):
        phase[phase > np.pi] -= 2*np.pi
    while np.any(phase < -np.pi):
        phase[phase < -np.pi] += 2*np.pi
    return phase

def create_dataset(num_samples=10000, filepath = 'C:/Users/thatm/VSCodeProjects/dataset2',
                   
                   image_size=256, 
                   center_offset_x_amp = 256//15,
                    center_offset_y_amp = 256//15,
                    radius_min = 256//4,
                    radius_max = 256//2, #256//2 - 256//15,
                    softness_min = 1.5, #1.5
                    softness_max = 4.0, #4,0
                    irregularity_min = 0.0,
                    irregularity_max = 0.05,
                    gaussian_sigma_min = 256//4,
                    gaussian_sigma_max = 256//2,
                    noise_strenght_min = 0.01,
                    noise_strenght_max = 0.06             
                   ):
    """Создает полный датасет со случайными параметрами.------"""

    metadata = []

    for i in range(num_samples):
        # Генерация sample
        amp, phase, params, text = generate_random_beam_sample(image_size,
                                                               center_offset_x_amp,
                                                               center_offset_y_amp,
                                                               radius_min,
                                                               radius_max,
                                                               softness_min,
                                                                softness_max,
                                irregularity_min,
                                irregularity_max,
                                gaussian_sigma_min,
                                gaussian_sigma_max,
                                noise_strenght_min, noise_strenght_max)
                                

        # Нормализация для сохранения в изображение
        amp_uint8 = (amp * 255).astype(np.uint8)
        # Фазу нормализуем из [-pi, pi] в [0, 255]

        
        phase_norm = normalize_phase(phase) #[-pi, pi]
        phase_norm = ((phase_norm + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        # Создаем двухканальное изображение
        img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        img_array[..., 0] = amp_uint8  # Красный канал - амплитуда
        img_array[..., 1] = phase_norm # Зеленый канал - фаза

        # Сохранение изображения
        filename = f'beam_{i:06d}.png'
        Image.fromarray(img_array).save(f'{filepath}/beams/{filename}')

        # Добавление метаданных
        metadata.append({
            'filename': 'difracted_' + filename, #МЕТАДАННЫЕ! ЭТО НАЗВАНИЕ ИСПОЛЬЗОВАТЬ ДЛЯ СОХРАНЕНИЯ КАРТИН ДИФРАКЦИИИ!!!!!!!!!
            **params
        })

        if (i + 1) % 1000 == 0:
            print(f"Сгенерировано {i + 1}/{num_samples} samples")

    # Сохранение метаданных
    with open(filepath + '/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata

# Визуализация нескольких случайных примеров
def visualize_random_samples(num_samples=3):
    """Визуализирует несколько случайных samples."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(40, 10*num_samples))

    for i in range(num_samples):
        amp, phase, params, text = generate_random_beam_sample()

        axes[i, 0].imshow(amp, cmap='viridis')
        axes[i, 0].set_title(f'Амплитуда {i+1}\nСмещение: {params["center_offset"]}')
        axes[i, 0].axis('off')
       
        axes[i, 1].imshow(phase, cmap='hsv')
        axes[i, 1].set_title(f'Фаза {i+1}\nАберраций: {len(params["zernike_coefficients"])}')
        axes[i, 1].axis('off')

        # Профиль пучка
        center_y = amp.shape[0] // 2
        axes[i, 2].plot(amp[center_y, :], label='Амплитуда')
        axes[i, 2].plot((phase[center_y, :]))
        axes[i, 2].set_title('Профиль пучка')
        axes[i, 2].legend()
        axes[i, 2].grid(True)

        axes[i, 3].text(0.1, 0.5, text, fontsize=20, va='center', ha='left')
        axes[i, 3].set_title(f' {i+1}: Параметры аберраций')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


import cv2
# Интерполяция циклических изображений

def interpolate_cyclic_image(src, scale_factor, interpolation=cv2.INTER_LINEAR):
    """
    Интерполяция изображения с циклическими данными (0 ~ 255).

    Параметры:
        src: Входное одноканальное изображение (фаза, угол и т.д.).
        scale_factor: Во сколько раз увеличить (например, 2.0).
        interpolation: Метод для шага 2 (cv2.INTER_LINEAR, cv2.INTER_CUBIC).

    Возвращает:
        Интерполированное циклическое изображение того же типа.
    """
    # 1. Нормализуем данные в радианы: [0, 255] -> [0, 2*pi]
    #    Или оставляем как есть, если atan2 работает корректно.
    phase_normalized = src.astype(np.float32) * (2 * np.pi / 255.0)

    # 2. Преобразуем в синус и косинус
    sin_img = np.sin(phase_normalized)
    cos_img = np.cos(phase_normalized)

    # 3. Определяем новый размер
    new_size = (int(src.shape[1] * scale_factor), int(src.shape[0] * scale_factor))

    # 4. Интерполируем синус и косинус независимо
    sin_interp = cv2.resize(sin_img, new_size, interpolation=interpolation)
    cos_interp = cv2.resize(cos_img, new_size, interpolation=interpolation)

    # 5. Вычисляем обратную функцию (арктангенс) и конвертируем обратно
    #    atan2 возвращает значения в диапазоне [-pi, pi]
    phase_interp = np.arctan2(sin_interp, cos_interp)

    # 6. Приводим к диапазону [0, 2*pi], а затем к [0, 255]
    phase_interp[phase_interp < 0] += 2 * np.pi
    result = (phase_interp * (255.0 / (2 * np.pi))).astype(src.dtype)

    return result


"""
==================================================================
Блок 3 Функции для генерации датасета с коэффициентами полиномов Чебышёва
==================================================================
"""

def chebyshev_poly(n, x):
    """
    Вычисляет значения полинома Чебышева первого рода T_n(x) для массива x.
    Используется рекуррентное соотношение.
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T_n_2 = np.ones_like(x)   # T_0
        T_n_1 = x                 # T_1
        for k in range(2, n+1):
            T_n = 2 * x * T_n_1 - T_n_2
            T_n_2, T_n_1 = T_n_1, T_n
        return T_n_1

def reconstruct_phase_from_chebyshev(coeffs_dict, size=256):
    """
    Восстанавливает фазовую маску (радианы) по словарю коэффициентов a_{p,q}.
    
    coeffs_dict: dict с ключами вида 'p_q' и значениями коэффициентов (в радианах).
    size: размер квадратной сетки.
    Возвращает: np.array (size, size) фаза в радианах.
    """
    # Сетка в диапазоне [-1, 1]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phase = np.zeros((size, size), dtype=np.float32)
    
    for key, a in coeffs_dict.items():
        if key == '0_0':
            continue   # постоянное смещение не влияет на интенсивность, можно игнорировать
        p, q = map(int, key.split('_'))
        Tp = chebyshev_poly(p, X)
        Tq = chebyshev_poly(q, Y)
        phase += a * Tp * Tq
    
    # Опционально: нормализуем фазу в [-π, π]
    phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
    return phase



def generate_chebyshev_dataset(num_samples=1000, size=256, max_order=7, 
                               coeff_ranges=None, seed=None):
    """
    Генерирует датасет фазовых масок и коэффициентов 2D полиномов Чебышева.
    
    Параметры:
        num_samples : int - число примеров
        size : int - размер квадратной сетки (size x size)
        max_order : int - максимальный суммарный порядок p+q
        coeff_ranges : list of tuple - для каждого полинома (в порядке обхода)
                       задаётся диапазон (min, max). Если None, используются значения по умолчанию.
        seed : int - для воспроизводимости
    
    Возвращает:
        phases : np.array (num_samples, size, size) - фазовые маски в радианах
        coeffs : np.array (num_samples, M) - коэффициенты a_{pq}
        poly_list : list of tuples (p, q) - порядки полиномов, соответствующие столбцам coeffs
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Генерация сетки
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y, indexing='ij')  # X, Y размером (size, size)
    
    # Список полиномов: все пары (p,q) с p+q <= max_order, упорядоченные по возрастанию суммы
    poly_list = []
    for s in range(max_order + 1):
        for p in range(s + 1):
            q = s - p
            poly_list.append((p, q))
    M = len(poly_list)  # количество полиномов
    
    # Задание диапазонов по умолчанию, если не переданы
    if coeff_ranges is None:
        coeff_ranges = []
        for p, q in poly_list:
            s = p + q
            if s == 0:
                coeff_ranges.append((-np.pi, np.pi))
            elif s <= 2:
                coeff_ranges.append((-np.pi, np.pi))
            elif s == 3:
                coeff_ranges.append((-np.pi/2, np.pi/2))
            elif s == 4:
                coeff_ranges.append((-np.pi/3, np.pi/3))
            elif s == 5:
                coeff_ranges.append((-np.pi/4, np.pi/4))
            elif s == 6:
                coeff_ranges.append((-np.pi/5, np.pi/5))
            else:  # s == 7
                coeff_ranges.append((-np.pi/6, np.pi/6))
    
    # Вычисление базисных функций для всех полиномов (один раз, т.к. они не зависят от коэффициентов)
    basis = []
    for p, q in poly_list:
        Tp = chebyshev_poly(p, X)
        Tq = chebyshev_poly(q, Y)
        basis.append(Tp * Tq)  # размер (size, size)
    basis = np.array(basis)    # (M, size, size)
    
    # Массивы для результатов
    phases = np.zeros((num_samples, size, size), dtype=np.float32)
    coeffs = np.zeros((num_samples, M), dtype=np.float32)
    
    for i in range(num_samples):
        # Генерация случайных коэффициентов
        a = np.zeros(M)
        for j, (low, high) in enumerate(coeff_ranges):
            a[j] = np.random.uniform(low, high)
        coeffs[i, :] = a
        
        # Фаза = сумма a_j * basis_j
        phi = np.tensordot(a, basis, axes=(0, 0))  # (size, size)
        # (Опционально) нормализуем фазу в диапазон [-pi, pi]
        phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
        phases[i, :, :] = phi
    
    return phases, coeffs, poly_list



def generate_amplitudes(size=256, mode=None, params=None, scale=1.0, seed=None):
    """
    Генерирует одну амплитудную маску размером (size, size) в диапазоне [0, 1].
    Если mode не задан, выбирается случайно. Параметры (центр, сигма и т.д.)
    также генерируются случайно, если не переданы в params.

    Возвращает:
        amplitude : np.array (size, size) - маска амплитуды.
        used_params : dict - использованные параметры (режим и значения).
    """
    if seed is not None:
        np.random.seed(seed)

    modes = ['uniform', 'gaussian']
    if mode is None:
        mode = np.random.choice(modes)
    elif mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Available: {modes}")

    if params is None:
        params = {}

    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y, indexing='ij')

    if mode == 'uniform':
        amp = np.ones((size, size), dtype=np.float32)
        used_params = {'mode': mode}

    elif mode == 'gaussian':
        cx = params.get('center_x', np.random.randint(int(0.3*size), int(0.7*size)))
        cy = params.get('center_y', np.random.randint(int(0.3*size), int(0.7*size)))
        sigma = params.get('sigma', np.random.uniform(size/2, size*2))
        amp_max = params.get('amplitude_max', np.random.uniform(0.7, 1.0))

        r2 = (X - cx)**2 + (Y - cy)**2
        amp = amp_max * np.exp(-r2 / (2 * sigma**2))
        used_params = {'mode': mode, 'center_x': cx, 'center_y': cy, 'sigma': sigma, 'amplitude_max': amp_max}

    
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    amp = np.clip(amp * scale, 0, 1).astype(np.float32)
    return amp, used_params


def generate_amplitude_dataset(num_samples=1000, size=256, mode=None, seed=None):
    """
    Генерирует набор амплитудных масок.

    Параметры:
        num_samples : int - количество примеров.
        size : int - размер маски (size x size).
        mode : str или None - режим (если None, каждый пример получает случайный режим).
        seed : int - для воспроизводимости.

    Возвращает:
        amplitudes : np.array (num_samples, size, size) - массив масок.
        amp_params_list : list of dict - список параметров для каждого примера.
    """
    if seed is not None:
        np.random.seed(seed)

    amplitudes = np.zeros((num_samples, size, size), dtype=np.float32)
    amp_params_list = []

    for i in range(num_samples):
        amp, params = generate_amplitudes(size=size, mode=mode, seed=None)
        amplitudes[i, :, :] = amp
        amp_params_list.append(params)

    return amplitudes, amp_params_list