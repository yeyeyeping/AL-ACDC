import numpy as np
import torch
import os
from util import read_yml
import sys
from model.MGUnet import MGUNet
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom

assert len(sys.argv) - 1 == 4, "cfg_path, img_folder, ckpath, out_dir"
cfg_path, img_folder, ckpath, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

assert os.path.exists(img_folder) and os.path.exists(ckpath)

input_size = 192
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs(out_dir, exist_ok=True)

cfg = read_yml(cfg_path)
model = MGUNet(cfg["Network"]).to(device)

model.load_state_dict(torch.load(ckpath, map_location=device)["model_state_dict"])
model.eval()
with torch.no_grad():
    gt = Path(img_folder).glob("*_gt.nii.gz")
    for g in gt:
        img_path = str(g)[:-10] + ".nii.gz"
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
        np.save(os.path.join(out_dir, str(g.name)), batch_pred_mask)

