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
    "szurke": "#616871",
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
    import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# --- BEÁLLÍTÁSOK ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
QUALITY = 85
MAX_SIZE = 1000  # Maximum 1000x1000 px

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


def make_square_white_background(img, max_size=1000):
    """
    A képet 1:1 arányú, fehér hátterű képre helyezi.
    Maximum max_size x max_size méretű lesz.
    Kisebb képet nem nagyít fel.
    """

    width, height = img.size

    # A négyzet mérete maximum 1000,
    # de kisebb eredeti képnél nem nagyítunk.
    square_size = min(max(width, height), max_size)

    # Ha az eredeti kép nagyobb a maximális méretnél,
    # arányosan lekicsinyítjük.
    scale = min(
        square_size / width,
        square_size / height,
        1.0
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    if (new_width, new_height) != (width, height):
        img = img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    # Fehér, négyzet alakú háttér
    background = Image.new(
        "RGB",
        (square_size, square_size),
        (255, 255, 255)
    )

    # Középre igazítás
    x = (square_size - new_width) // 2
    y = (square_size - new_height) // 2

    # Ha van alpha csatorna, azt maszkként használjuk
    if img.mode == "RGBA":
        background.paste(img, (x, y), img)
    else:
        background.paste(img, (x, y))

    return background


def process_product_image(input_path, output_path, target_hex, quality=85):
    """
    Eltávolítja a hátteret,
    átszínezi a terméket,
    fehér háttérre helyezi,
    1:1 képet készít,
    majd WebP-ként menti.
    """

    input_img = Image.open(input_path).convert("RGBA")

    # Háttér eltávolítása
    no_bg_img = remove(input_img)

    img_rgba = np.array(no_bg_img)

    bgr = cv2.cvtColor(
        img_rgba[:, :, :3],
        cv2.COLOR_RGB2BGR
    )

    alpha = img_rgba[:, :, 3]

    # LAB színtér: a fényességet megtartjuk,
    # a színezetet cseréljük.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    target_bgr = np.uint8(
        [[hex_to_bgr(target_hex)]]
    )

    target_lab = cv2.cvtColor(
        target_bgr,
        cv2.COLOR_BGR2LAB
    )[0][0]

    # Csak a nem átlátszó részeket színezzük
    mask = alpha > 0

    a[mask] = target_lab[1]
    b[mask] = target_lab[2]

    new_lab = cv2.merge([l, a, b])
    new_bgr = cv2.cvtColor(
        new_lab,
        cv2.COLOR_LAB2BGR
    )

    new_rgb = cv2.cvtColor(
        new_bgr,
        cv2.COLOR_BGR2RGB
    )

    final_rgba = np.dstack(
        (new_rgb, alpha)
    )

    product_img = Image.fromarray(
        final_rgba.astype(np.uint8),
        "RGBA"
    )

    # Fehér háttér + négyzetes kép
    final_img = make_square_white_background(
        product_img,
        max_size=MAX_SIZE
    )

    # WebP mentés
    final_img.save(
        output_path,
        "WEBP",
        quality=quality,
        method=6
    )


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)

        print(
            f"Létrehoztam az '{INPUT_DIR}' mappát. "
            "Másold bele a forrásképeket, majd indítsd újra a scriptet!"
        )
        return

    valid_extensions = (
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    )

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(
            f"Nincs feldolgozható kép az "
            f"'{INPUT_DIR}' mappában!"
        )
        return

    print(
        f"Talált képek száma: {len(image_files)}. "
        "Feldolgozás indítása...\n"
    )

    for file_name in image_files:

        base_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(
            INPUT_DIR,
            file_name
        )

        for color_name, hex_code in COLORS.items():

            output_file_name = (
                f"{base_name}_{color_name}.webp"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_file_name
            )

            print(
                f"Generálás: {file_name} -> "
                f"{output_file_name} ({hex_code})"
            )

            process_product_image(
                input_path,
                output_path,
                hex_code,
                quality=QUALITY
            )

    print(
        "\nFolyamat kész! "
        "A képek elmentve az 'output' mappába."
    )


if __name__ == "__main__":
    main()
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