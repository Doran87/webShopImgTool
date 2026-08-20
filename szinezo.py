import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Output méret
OUTPUT_SIZE = 1000  # 1000x1000 px

# WebP minőség
QUALITY = 100
WEBP_LOSSLESS = True

# A termék a végső kép kb. hány százalékát töltse ki
# 0.90 = 90%
PRODUCT_FILL = 0.90

# Átszínezés paraméterei
LIGHTNESS_STRENGTH = 0.72
CONTRAST_STRENGTH = 1.10

# Alpha threshold
ALPHA_THRESHOLD = 5

# Outline / halo csökkentés
HALO_ERODE_PX = 1
HALO_BLUR_PX = 1
HALO_WHITE_THRESHOLD = 220

# ============================================================
# SZÍNEK
# ============================================================

COLORS = {
    "szurke": "#616871",
    "bezs": "#C6B193",
    "fekete": "#1B1B1B",
    "khaki": "#797C69"
}

# ============================================================
# SEGÉDFÜGGVÉNYEK
# ============================================================

def hex_to_rgb(hex_str):
    """HEX színkód -> RGB tuple."""
    hex_str = hex_str.strip().lstrip("#")

    if len(hex_str) != 6:
        raise ValueError(
            f"Hibás HEX kód: {hex_str}. "
            "Példa helyes formátumra: #616871"
        )

    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(hex_str):
    """HEX színkód -> BGR tuple OpenCV-hez."""
    r, g, b = hex_to_rgb(hex_str)
    return (b, g, r)


def remove_white_halo(img_rgba, threshold=220):
    """
    Csökkenti a háttérből megmaradó fehéres/szürkés halót
    a félig áttetsző széleken.
    """
    arr = np.array(img_rgba).astype(np.float32)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] / 255.0

    # Csak a félig áttetsző pixeleknél dolgozunk
    edge_mask = (alpha > 0.0) & (alpha < 1.0)

    # További szűrés: főleg világos pixeleket korrigálunk
    bright_mask = np.mean(rgb, axis=2, keepdims=True) >= threshold

    correction_mask = edge_mask & bright_mask

    safe_alpha = np.maximum(alpha, 0.05)

    # Fehér háttérből származó beégés visszaszámítása
    corrected = (rgb - 255.0 * (1.0 - alpha)) / safe_alpha
    corrected = np.clip(corrected, 0, 255)

    rgb = np.where(correction_mask, corrected, rgb)

    arr[:, :, :3] = rgb
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def clean_alpha_edges(img_rgba, erode_px=1, blur_px=1):
    """
    Finomítja az alpha csatorna szélét, hogy kisebb legyen az outline.
    """
    arr = np.array(img_rgba).copy()
    alpha = arr[:, :, 3]

    if erode_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=erode_px)

    if blur_px > 0:
        k = blur_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def crop_to_content(img, alpha_threshold=5):
    """
    Levágja az átlátszó széleket a termék körül.
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    mask = alpha > alpha_threshold
    if not np.any(mask):
        return img

    ys, xs = np.where(mask)

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    return img.crop((left, top, right, bottom))


# ============================================================
# SZÍNEZÉS
# ============================================================

def recolor_product(
    product_img,
    target_hex,
    lightness_strength=0.72,
    contrast_strength=1.10
):
    """
    A terméket a megadott HEX színre színezi úgy,
    hogy a textúra, árnyékok és fények megmaradjanak.
    """
    product_img = product_img.convert("RGBA")
    img_rgba = np.array(product_img)

    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    target_bgr = np.uint8([[hex_to_bgr(target_hex)]])
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0]

    target_l = float(target_lab[0])
    target_a = int(target_lab[1])
    target_b = int(target_lab[2])

    mask = alpha > ALPHA_THRESHOLD
    if not np.any(mask):
        return product_img

    original_l = l.astype(np.float32)
    product_l = original_l[mask]

    mean_l = float(np.mean(product_l))
    detail = product_l - mean_l
    detail *= contrast_strength

    base_l = mean_l * (1.0 - lightness_strength) + target_l * lightness_strength
    new_product_l = np.clip(base_l + detail, 0, 255)

    new_l = original_l.copy()
    new_l[mask] = new_product_l
    new_l = np.clip(new_l, 0, 255).astype(np.uint8)

    new_a = a.copy()
    new_b = b.copy()

    new_a[mask] = target_a
    new_b[mask] = target_b

    new_lab = cv2.merge([new_l, new_a, new_b])
    new_bgr = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
    new_rgb = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB)

    final_rgba = np.dstack((new_rgb, alpha))

    return Image.fromarray(final_rgba.astype(np.uint8), "RGBA")


# ============================================================
# FEHÉR, 1000x1000 VÉGSŐ KÉP
# ============================================================

def create_square_image(product_img, output_size=1000, fill_ratio=0.90):
    """
    Fehér hátterű, fix 1000x1000 képet készít.
    A terméket középre teszi.
    Nem nagyítja fel az eredeti kivágott terméket 1:1 fölé.
    """
    product_img = product_img.convert("RGBA")

    usable_size = int(output_size * fill_ratio)

    width, height = product_img.size
    if width <= 0 or height <= 0:
        raise ValueError("A termék képmérete hibás.")

    scale = min(
        usable_size / width,
        usable_size / height
    )

    # Ne nagyítsunk felfelé, hogy minél jobb maradjon a minőség
    scale = min(scale, 1.0)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    if (new_width, new_height) != (width, height):
        product_img = product_img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    background = Image.new(
        "RGB",
        (output_size, output_size),
        (255, 255, 255)
    )

    x = (output_size - product_img.width) // 2
    y = (output_size - product_img.height) // 2

    background.paste(product_img, (x, y), product_img)

    return background


# ============================================================
# FŐ FELDOLGOZÁS
# ============================================================

def process_product_image(input_path, output_path, target_hex, quality=100):
    """
    1. Betöltés
    2. Háttér eltávolítás
    3. Halo / outline csökkentés
    4. Körbevágás
    5. Átszínezés
    6. Fehér 1000x1000 háttér
    7. WebP mentés
    """
    input_img = Image.open(input_path).convert("RGBA")

    # Háttér eltávolítás
    no_bg_img = remove(input_img).convert("RGBA")

    # Halo / outline csökkentés
    no_bg_img = remove_white_halo(
        no_bg_img,
        threshold=HALO_WHITE_THRESHOLD
    )

    no_bg_img = clean_alpha_edges(
        no_bg_img,
        erode_px=HALO_ERODE_PX,
        blur_px=HALO_BLUR_PX
    )

    # Üres szélek levágása
    cropped_img = crop_to_content(
        no_bg_img,
        alpha_threshold=ALPHA_THRESHOLD
    )

    # Színezés
    recolored_img = recolor_product(
        cropped_img,
        target_hex,
        lightness_strength=LIGHTNESS_STRENGTH,
        contrast_strength=CONTRAST_STRENGTH
    )

    # Fehér hátteres 1000x1000
    final_img = create_square_image(
        recolored_img,
        output_size=OUTPUT_SIZE,
        fill_ratio=PRODUCT_FILL
    )

    # Mentés
    final_img.save(
        output_path,
        "WEBP",
        quality=quality,
        lossless=WEBP_LOSSLESS,
        method=6
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Futtatott fájl: {os.path.abspath(__file__)}")
    print("Webshop kép generátor indul...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Létrehoztam az '{INPUT_DIR}' mappát.")
        print("Másold bele a forrásképeket, majd indítsd újra a scriptet.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"Nincs feldolgozható kép az '{INPUT_DIR}' mappában.")
        return

    print(f"Talált képek száma: {len(image_files)}")
    print(f"Színváltozatok száma: {len(COLORS)}")
    print()

    for index, file_name in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"[{index}/{len(image_files)}] {file_name}")

        base_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(INPUT_DIR, file_name)

        for color_name, hex_code in COLORS.items():
            output_file_name = f"{base_name}_{color_name}.webp"
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            print()
            print(
                f"Generálás:"
                f"\n  {file_name}"
                f"\n  -> {output_file_name}"
                f"\n  Szín: {hex_code}"
            )

            try:
                process_product_image(
                    input_path,
                    output_path,
                    hex_code,
                    quality=QUALITY
                )

                with Image.open(output_path) as result:
                    print(f"  Kész: {result.width}x{result.height} px")

            except Exception as error:
                print(f"  HIBA: {error}")

    print()
    print("=" * 60)
    print(f"Folyamat kész! A képek az '{OUTPUT_DIR}' mappába kerültek.")


if __name__ == "__main__":
    main()