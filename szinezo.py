import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# --- BEÁLLÍTÁSOK ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
QUALITY = 85  # WebP minőség (1-100)

# A kívánt színváltozatok (Szín neve: HEX kód)
COLORS = {
    "piros": "#FF0000",
    "kek": "#1E90FF",
    "zold": "#2ECC71",
    "fekete": "#333333",
    "sarga": "#F1C40F"
}

def hex_to_bgr(hex_str):
    """HEX színkód átalakítása BGR formátumba OpenCV-hez."""
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def process_product_image(input_path, output_path, target_hex, quality=85):
    """Eltávolítja a hátteret, átszínezi a terméket és elmenti WebP-ként."""
    input_img = Image.open(input_path)
    no_bg_img = remove(input_img)
    
    img_rgba = np.array(no_bg_img)
    
    bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = img_rgba[:, :, 3]
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    target_bgr = np.uint8([[hex_to_bgr(target_hex)]])
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0][0]
    
    mask = alpha > 0
    a[mask] = target_lab[1]
    b[mask] = target_lab[2]
    
    new_lab = cv2.merge([l, a, b])
    new_bgr = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
    
    new_rgb = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB)
    final_rgba = np.dstack((new_rgb, alpha))
    
    final_img = Image.fromarray(final_rgba)
    final_img.save(output_path, "WEBP", quality=quality)

def main():
    # Mappák automatikus létrehozása
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Létrehoztam az '{INPUT_DIR}' mappát. Másold bele a forrásképeket, majd indítsd újra a scriptet!")
        return

    # Támogatott képformátumok szűrése
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Nincs feldolgozható kép az '{INPUT_DIR}' mappában!")
        return

    print(f"Talált képek száma: {len(image_files)}. Feldolgozás indítása...\n")

    for file_name in image_files:
        base_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(INPUT_DIR, file_name)

        for color_name, hex_code in COLORS.items():
            # Új fájlnév generálása pl.: termek1_piros.webp
            output_file_name = f"{base_name}_{color_name}.webp"
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            print(f"Generálás: {file_name} -> {output_file_name} ({hex_code})")
            process_product_image(input_path, output_path, hex_code, quality=QUALITY)

    print("\nFolyamat kész! A képek elmentve az 'output' mappába.")

if __name__ == "__main__":
    main()