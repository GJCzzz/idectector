import argparse
import glob
import os
from PIL import Image
import torch
import torch.onnx
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from process_utils.utils import get_network, str2bool  # 确保utils模块中有str2bool

def exportOnnx(model: torch.nn.Module, output_path: str = "deploy_model.onnx"):
    """导出PyTorch模型为ONNX格式"""
    # 强制使用CPU导出以避免设备不匹配问题 后续需要修改
    model.cpu()
    
    # 生成与预处理一致的示例输入 (1,3,224,224)
    dummy_input = torch.randn(1, 3, 224, 224).cpu()
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # 支持动态batch
            "output": {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True  # 启用常量折叠优化
    )
    print(f"[EXPORT] ONNX模型已保存至: {os.path.abspath(output_path)}")

def init_necessary_args(
    model_path: str,
    use_cpu: bool = False,
    export_onnx: bool = False,
    onnx_output_path: str = "deploy_model.onnx"
):
    """初始化模型和预处理，支持ONNX导出"""
    # 加载模型
    model = get_network("resnet50")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    model.eval()
    
    # 设备配置
    if not use_cpu:
        model.cuda()
    
    # ONNX导出
    if export_onnx:
        exportOnnx(model, onnx_output_path)
    
    # 定义预处理流程（需与训练时完全一致）
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return model, trans

def eval_single_image(
    model: torch.nn.Module,
    img_path: str,
    use_cpu: bool = False,
    aug_norm: bool = True,
    trans: transforms.Compose = None,
):
    """单张图像推理"""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"[ERROR] 图像不存在: {img_path}")
    
    # 预处理流程
    img = Image.open(img_path).convert("RGB")
    img_tensor = trans(img)
    
    if aug_norm:  # 保持与训练相同的归一化
        img_tensor = TF.normalize(
            img_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    # 设备转移
    img_tensor = img_tensor.unsqueeze(0)
    if not use_cpu:
        img_tensor = img_tensor.cuda()
    
    # 推理
    with torch.no_grad():
        prob = model(img_tensor).sigmoid().item()
    return prob

def eval_whole_folder(
    folder_path: str,
    model: torch.nn.Module,
    trans: transforms.Compose,
    use_cpu: bool = False,
    aug_norm: bool = True,
):
    """批量测试文件夹中的图像"""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"[ERROR] 路径不存在: {folder_path}")
    
    # 获取图像列表
    img_patterns = ("*.jpg", "*.png", "*.JPEG")
    file_list = sorted([
        f for p in img_patterns 
        for f in glob.glob(os.path.join(folder_path, p))
    ])
    
    print(f"[INFER] 正在处理文件夹: {os.path.basename(folder_path)}")
    is_real_class = folder_path.split("/")[-1] == "0_real"
    
    correct = 0
    for img_path in tqdm(file_list, desc="Processing", dynamic_ncols=True):
        prob = eval_single_image(
            model=model, 
            img_path=img_path,
            use_cpu=use_cpu,
            aug_norm=aug_norm,
            trans=trans
        )
        
        # 统计正确率
        if (is_real_class and prob < 0.5) or (not is_real_class and prob >= 0.5):
            correct += 1
            
    return correct / len(file_list) if len(file_list) > 0 else 0.0

def iterate_over_folders(
    root_folder: str,
    model_path: str,
    use_cpu: bool = False,
    aug_norm: bool = True,
):
    """递归遍历所有子文件夹"""
    model, trans = init_necessary_args(model_path, use_cpu)
    results = {}
    
    for entry in sorted(glob.glob(os.path.join(root_folder, "*"))):
        if os.path.isdir(entry):
            # 递归处理子目录
            results.update(iterate_over_folders(entry, model_path, use_cpu, aug_norm))
        else:
            # 处理当前目录的文件
            parent_dir = os.path.dirname(entry)
            if parent_dir not in results:
                acc = eval_whole_folder(
                    folder_path=parent_dir,
                    model=model,
                    trans=trans,
                    use_cpu=use_cpu,
                    aug_norm=aug_norm
                )
                results[parent_dir] = acc
    return results

if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="模型评估与ONNX导出")
    parser.add_argument("--data_dir", type=str, default="../lsun_bedroom_release/adm")
    parser.add_argument("--model_path", type=str, default="data/dire_ckpt/imagenet_adm.pth")
    parser.add_argument("--use_cpu", type=str2bool, default=True)
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    parser.add_argument("--export_onnx", type=str2bool, default=True)
    parser.add_argument("--onnx_output", type=str, default="oonx_img.onnx")
    args = parser.parse_args()
    
    # 初始化模型（自动触发ONNX导出）
    model, trans = init_necessary_args(
        model_path=args.model_path,
        use_cpu=args.use_cpu,
        export_onnx=args.export_onnx,
        onnx_output_path=args.onnx_output
    )
    
    # 执行评估
    results = iterate_over_folders(
        root_folder=args.data_dir,
        model_path=args.model_path,
        use_cpu=args.use_cpu,
        aug_norm=args.aug_norm
    )
    
    # 输出结果
    print("\n评估结果:")
    for path, acc in results.items():
        print(f"{os.path.basename(path):<20} | Accuracy: {acc:.4f}")