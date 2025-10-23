import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import streamlit as st
import torch
import logging

def get_logger(name=__name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("log.log", encoding="utf-8")]
    )
    return logging.getLogger(__name__)

logger = get_logger(__name__)

def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

@st.cache_resource
def get_local_images_path() -> list[Path]:
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

    logger.info(f"{len(valid_images)} valid images found\n")
    return valid_images

@st.cache_resource
def generate_id(path: Path | str) -> str:
    if isinstance(path, str):
        filename = path.lower().strip()
        return hashlib.md5(data=filename.encode("utf-8")).hexdigest()
    else:
        filename = path.name.lower().strip()
        return hashlib.md5(data=filename.encode("utf-8")).hexdigest()