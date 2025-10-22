import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import streamlit as st
import torch

def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

@st.cache_resource
def get_all_images_path() -> list[Path]:
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data/1/MAE 209SUP sample'
    # data_dir = script_dir / 'data/0/portrait-0.1k'
    all_images = list(data_dir.rglob('*.jpg'))
    
    valid_images = []
    for img_path in tqdm(all_images, desc="Verifying images", unit="img"):
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_images.append(img_path)
        except Exception as e:
            print(f'Error on {img_path}: {e}')

    print(f"{len(valid_images)} valid images found\n")
    return valid_images

@st.cache_resource
def generate_id(path: Path) -> str:
    filename = path.name.lower().strip()
    return hashlib.md5(filename.encode("utf-8")).hexdigest()