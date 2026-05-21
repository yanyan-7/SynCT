import nibabel as nib

img_moving = nib.load("nrbrain_T1.nii")
img_fixed = nib.load("rspm152.nii")
print(img_moving.shape)  # 输出形状
print(img_moving.get_data_dtype())  # 输出数据类型
print(img_fixed.shape)  # 输出形状
print(img_fixed.get_data_dtype())  # 输出数据类型