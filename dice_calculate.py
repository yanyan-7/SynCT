import os

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import numpy as np
import torch
import voxelmorph as vxm
import nibabel as nib


def dice_compute(moving_path, fixed_path, labels):
    moving_seg = vxm.py.utils.load_volfile(moving_path)
    fixed_seg = vxm.py.utils.load_volfile(fixed_path)

    moving_seg = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=4)
    moving_seg = torch.from_numpy(moving_seg).float().permute(0, 4, 1, 2, 3)
    fixed_seg = np.expand_dims(np.expand_dims(fixed_seg, axis=0), axis=4)
    fixed_seg = torch.from_numpy(fixed_seg).float().permute(0, 4, 1, 2, 3)

    overlap = vxm.py.utils.dice(moving_seg.cpu().numpy(), fixed_seg.cpu().numpy(), labels=labels)

    # print('Dice: %.4f +/- %.4f' % (np.mean(overlap), np.std(overlap)))
    return (np.mean(overlap), np.std(overlap))




# if __name__ == "__main__":
#     moving_path = r'F:\Registration-CorrMLP\rwSpm_T1_rlabels.nii.gz'
#     fixed_path = r'F:\Registration-CorrMLP\synthmorph_joint_rlabels.nii.gz'
#     labels = [1, 2, 3, 4]
#     dice_compute(moving_path, fixed_path, labels)
