from model.MGUnet import MGUNet
from util import read_yml
import torch
import sys
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np

assert len(sys.argv) - 1 == 2, "result folder, test image folder"
result_folder, img_folder = Path(sys.argv[1]), Path(sys.argv[2])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
input_size = 192

torch.set_grad_enabled(False)


def model_from_cfg(cfg_path):
    cfg = read_yml(cfg_path)
    model = MGUNet(cfg["Network"]).to(device)
    model.eval()
    return model


def binary_dice(s, g):
    """
    Calculate the Dice score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    :param resize: (optional, bool)
        If s and g have different shapes, resize s to match g.
        Default is `True`.

    :return: The Dice value.
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0 * s0 + 1e-5) / (s1 + s2 + 1e-5)
    return dice


def metrics(pred, gt, class_num):
    class_dice = []
    for i in range(class_num):
        p, g = (pred == i), (gt == i)
        class_dice.append(binary_dice(p, g))
    return class_dice


gt_folder = list(Path(img_folder).glob("*_gt.nii.gz"))
result = {}
for method in result_folder.iterdir():
    if not method.is_dir():
        continue
    model = model_from_cfg(str(method / "config.yml"))
    result[method.name] = []
    for ckpoint in sorted(list((method / "checkpoint").glob("c[0,1,2,3,4,5,6,7]_best*"))):
        dice_list = []
        model.load_state_dict(torch.load(str(ckpoint))["model_state_dict"])
        for gt_path in gt_folder:
            img_path = str(gt_path)[:-10] + ".nii.gz"
            img = sitk.ReadImage(img_path)
            img_npy = sitk.GetArrayFromImage(img)[:, None]

            *_, h, w = img_npy.shape
            zoomed_img = zoom(img_npy, (1, 1, input_size / h, input_size / w), order=1,
                              mode='nearest')
            zoomed_img = torch.from_numpy(zoomed_img).cuda()
            output, _, _ = model(zoomed_img)
            output = torch.stack(output).mean(0)
            pred_volume = zoom(output.cpu().numpy(), (1, 1, h / input_size, w / input_size), order=1,
                               mode='nearest')
            batch_pred_mask = pred_volume.argmax(axis=1)

            gt = sitk.ReadImage(str(gt_path))
            gt_npy = sitk.GetArrayFromImage(gt)
            dice = np.mean(metrics(batch_pred_mask, gt_npy, class_num=np.max(gt_npy))[1:])
            dice_list.append(dice)
        result[method.name].append(np.mean(dice_list))
    print(method.name, result[method.name])
    with open(str(result_folder / "plot.txt"), "w+") as fp:
        fp.write(str(result))
