import argparse
import glob
import os

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm

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
            rob = model(in_tens).sigmoid().item()
    return rob


def eval_whole_folder(
    folder_path: str,
    model: torch.nn.Module,
    trans: transforms.Compose,
    use_cpu: bool = False,
    aug_norm: bool = True,
):
    if os.path.isdir(folder_path):
        file_list = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))+glob.glob(os.path.join(folder_path, "*.JPEG")))
        print(f"Testing images from '{folder_path}'")
        is_real = folder_path.split("/")[-1] == "0_real"
    else:
        raise FileNotFoundError(f"Invalid file path: '{folder_path}'")

    result = 0
    for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
        prob = eval_single_image(
            model=model, 
            img_path=img_path,
            use_cpu=use_cpu, 
            aug_norm=aug_norm, 
            trans=trans
        )
        if is_real and prob < 0.5:
            result = result + 1
        elif not is_real and prob >= 0.5:
            result = result + 1
    return result / len(file_list)
        
        
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
            result.update(iterate_over_folders(subfolder, model_path, use_cpu, aug_norm))
        elif os.path.isfile(subfolder):
            result[folder] = eval_whole_folder(
                folder_path=folder, 
                model=model, 
                trans=trans,
                use_cpu=use_cpu, 
                aug_norm=aug_norm
            )
            break
    return result


if __name__ == "__main__":
    result = iterate_over_folders("organized_dataset", "data/dire_ckpt/lsun_adm.pth", use_cpu=True, aug_norm=True)
    print(result)