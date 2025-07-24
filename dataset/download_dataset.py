import os
import urllib.request
import zipfile
import shutil

datasets = {
    "T91": "https://www.kaggle.com/api/v1/datasets/download/ll01dm/t91-image-dataset",
    "Set5": "https://www.kaggle.com/api/v1/datasets/download/bijaygurung/set5-superresolution",
    "Set14": "https://www.kaggle.com/api/v1/datasets/download/hliang001/set-14"
}

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

def download_and_extract(name, url):
    print(f"Tải {name} ...")
    zip_path = os.path.join(output_dir, f"{name}.zip")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  -> Đã tải xong: {zip_path}")

    # Giải nén vào temp folder
    temp_extract = os.path.join(output_dir, f"{name}_temp")
    os.makedirs(temp_extract, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
    os.remove(zip_path)

    # Tìm thư mục chứa ảnh thật
    final_dir = os.path.join(output_dir, name)
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(final_dir)

    for root, dirs, files in os.walk(temp_extract):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                src = os.path.join(root, file)
                dst = os.path.join(final_dir, file)
                shutil.move(src, dst)

    # Xóa temp folder
    shutil.rmtree(temp_extract)
    print(f"{name} đã sẵn sàng trong {final_dir}\n")


for name, url in datasets.items():
    download_and_extract(name, url)

print("Hoàn tất tải T91, Set5, Set14.")
