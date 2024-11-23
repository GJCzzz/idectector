import argparse
import glob
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score

from process_utils.utils import get_network, str2bool


def init_necessary_args(
        model_path: str,
        use_cpu: bool = False,
):
    model = get_network("resnet50")
    state_dict = torch.load(model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    if not use_cpu:
        model.cuda()
    trans = transforms.Compose(
        (
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        )
    )
    return model, trans


def eval_single_image(
        model: torch.nn.Module,
        img_path: str,
        use_cpu: bool = False,
        aug_norm: bool = True,
        trans: transforms.Compose = None,
):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Invalid image path: '{img_path}'")

    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    if not use_cpu:
        in_tens = in_tens.cuda()

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
    return prob


def eval_whole_folder(
        folder_path: str,
        model: torch.nn.Module,
        trans: transforms.Compose,
        use_cpu: bool = False,
        aug_norm: bool = True,
):
    if os.path.isdir(folder_path):
        file_list = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(
                os.path.join(folder_path, "*.JPEG")))
        print(f"Testing images from '{folder_path}'")
        is_real = folder_path.split("/")[-1] == "0_real"
    else:
        raise FileNotFoundError(f"Invalid file path: '{folder_path}'")

    y_true, y_pred = [], []
    for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
        prob = eval_single_image(
            model=model,
            img_path=img_path,
            use_cpu=use_cpu,
            aug_norm=aug_norm,
            trans=trans
        )
        y_pred.append(prob)
        y_true.append(1 if is_real else 0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    results = {
        "ACC": acc,
        "AP": ap,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
    }
    return results


def iterate_over_folders(
        folder: str,
        model_path: str,
        use_cpu: bool = False,
        aug_norm: bool = True,
):
    model, trans = init_necessary_args(model_path, use_cpu)
    result = {}
    for subfolder in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.isdir(subfolder):
            subfolder_name = os.path.basename(subfolder)
            print(f"Processing subfolder: {subfolder_name}")  # 调试信息
            for subsubfolder in sorted(glob.glob(os.path.join(subfolder, "*"))):
                if os.path.isdir(subsubfolder):
                    subsubfolder_name = os.path.basename(subsubfolder)
                    print(f"Processing subsubfolder: {subsubfolder_name}")  # 调试信息
                    if subsubfolder_name in ["0_real", "1_fake"]:
                        result[f"{subfolder_name}_{subsubfolder_name}"] = eval_whole_folder(
                            folder_path=subsubfolder,
                            model=model,
                            trans=trans,
                            use_cpu=use_cpu,
                            aug_norm=aug_norm
                        )
    return result


if __name__ == "__main__":
    result = iterate_over_folders("data/organized_dataset", "data/exp/ckpt/dire_ckpt/lsun_adm.pth", use_cpu=False, aug_norm=True) #修改为自己的文件路径
    print(result)