from pathlib import Path

def get_images_path() -> list[Path]:
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data/portrait-0.1k'
    images = list(data_dir.rglob('*.jpg'))
    print(f"{len(images)} images found.")
    return images