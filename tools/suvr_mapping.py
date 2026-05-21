import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import os

def deform_img_based_on_other_img(original_img_path, refer_img_path):
    """
    基于参考图像对原始图像进行空间变换
    
    参数:
        original_img_path: 原始图像路径
        refer_img_path: 参考图像路径
    
    返回:
        new_img_data: 变换后的图像数据
        refer_affine: 参考图像的仿射矩阵
    """
    # 加载图像
    refer_img = nib.load(refer_img_path)
    original_img = nib.load(original_img_path)
    
    # 获取图像数据
    refer_data = refer_img.get_fdata()
    original_data = original_img.get_fdata()
    
    # 获取仿射矩阵
    refer_affine = refer_img.affine
    original_affine = original_img.affine
    
    # 获取图像尺寸
    refer_x, refer_y, refer_z = refer_data.shape
    origin_x, origin_y, origin_z = original_data.shape
    
    # 初始化新图像
    new_img = np.zeros((refer_x, refer_y, refer_z))
    
    # 计算从参考图像空间到原始图像空间的变换矩阵
    transform_matrix = np.linalg.inv(original_affine) @ refer_affine
    
    # 创建坐标网格（更高效的方法）
    i, j, k = np.meshgrid(
        np.arange(refer_x), 
        np.arange(refer_y), 
        np.arange(refer_z),
        indexing='ij'
    )
    
    # 将坐标展平以便批量处理
    coords = np.stack([i.flatten(), j.flatten(), k.flatten(), np.ones(i.size)]).T
    
    # 应用变换矩阵
    physical_coords = coords @ refer_affine.T
    origin_coords = physical_coords @ np.linalg.inv(original_affine).T
    
    # 提取坐标并重塑
    origin_i = origin_coords[:, 0].reshape(refer_x, refer_y, refer_z)
    origin_j = origin_coords[:, 1].reshape(refer_x, refer_y, refer_z)
    origin_k = origin_coords[:, 2].reshape(refer_x, refer_y, refer_z)
    
    # 使用插值获取原始图像值（最近邻插值，与MATLAB代码一致）
    new_img = map_coordinates(
        original_data, 
        [origin_i, origin_j, origin_k], 
        order=0,  # 0=最近邻插值
        mode='constant', 
        cval=0.0
    )
    
    return new_img, refer_affine

class PETNormalizerWithRegistration:
    def __init__(self):
        self.pet_img = None
        self.ref_mask_img = None
        self.pet_data = None
        self.ref_mask_data = None
        self.registered_mask = None
        
    def load_images(self, pet_path, ref_mask_path):
        """加载PET图像和参考脑区mask"""
        try:
            # 加载PET图像
            self.pet_img = nib.load(pet_path)
            self.pet_data = self.pet_img.get_fdata()
            
            # 加载参考脑区mask
            self.ref_mask_img = nib.load(ref_mask_path)
            self.ref_mask_data = self.ref_mask_img.get_fdata()
            
            print(f"PET图像尺寸: {self.pet_data.shape}")
            print(f"PET图像分辨率: {self.pet_img.header.get_zooms()}")
            print(f"参考脑区尺寸: {self.ref_mask_data.shape}")
            print(f"参考脑区分辨率: {self.ref_mask_img.header.get_zooms()}")
            
            return True
            
        except Exception as e:
            print(f"图像加载错误: {e}")
            return False
    
    def check_image_compatibility(self):
        """检查图像兼容性"""
        pet_shape = self.pet_data.shape
        ref_shape = self.ref_mask_data.shape
        pet_affine = self.pet_img.affine
        ref_affine = self.ref_mask_img.affine
        
        # 检查尺寸是否一致
        shape_match = (pet_shape == ref_shape)
        
        # 检查仿射矩阵是否一致（允许小的数值差异）
        affine_match = np.allclose(pet_affine, ref_affine, atol=1e-3)
        
        # 检查分辨率
        pet_zooms = self.pet_img.header.get_zooms()
        ref_zooms = self.ref_mask_img.header.get_zooms()
        resolution_match = np.allclose(pet_zooms, ref_zooms, atol=0.1)
        
        print(f"图像尺寸匹配: {shape_match}")
        print(f"仿射矩阵匹配: {affine_match}")
        print(f"分辨率匹配: {resolution_match}")
        
        return shape_match and affine_match and resolution_match
    
    def register_mask_to_pet(self):
        """
        使用您提供的MATLAB代码方法将参考脑区mask配准到PET图像空间
        """
        print("开始图像配准...")
        
        # 保存临时文件用于配准
        temp_pet_path = "temp_pet.nii"
        temp_mask_path = "temp_mask.nii"
        
        try:
            # 保存临时文件
            nib.save(self.pet_img, temp_pet_path)
            nib.save(self.ref_mask_img, temp_mask_path)
            
            # 使用配准函数
            registered_mask_data, pet_affine = deform_img_based_on_other_img(
                temp_mask_path, temp_pet_path
            )
            
            # 创建新的mask图像
            self.registered_mask = nib.Nifti1Image(
                registered_mask_data.astype(np.uint8), 
                pet_affine, 
                self.pet_img.header
            )
            
            print("配准完成")
            return True
            
        except Exception as e:
            print(f"配准错误: {e}")
            return False
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_pet_path):
                os.remove(temp_pet_path)
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
    
    def calculate_suvr(self, use_registered_mask=True):
        """计算SUVR归一化图像"""
        print("计算SUVR...")
        
        # 选择使用的mask
        if use_registered_mask and self.registered_mask is not None:
            mask_data = self.registered_mask.get_fdata()
        else:
            mask_data = self.ref_mask_data
        
        # 确保mask与PET图像尺寸一致
        if mask_data.shape != self.pet_data.shape:
            print("警告: mask与PET图像尺寸不一致，使用配准后的mask")
            if self.registered_mask is not None:
                mask_data = self.registered_mask.get_fdata()
            else:
                raise ValueError("mask与PET图像尺寸不一致且无配准后的mask")
        
        # 提取参考脑区（mask值为1的区域）
        reference_region = self.pet_data[mask_data == 1]
        
        if len(reference_region) == 0:
            raise ValueError("参考脑区中没有有效的体素，请检查mask文件")
        
        # 计算参考脑区的平均强度
        reference_mean = np.mean(reference_region)
        print(f"参考脑区平均强度: {reference_mean:.4f}")
        print(f"参考脑区体素数量: {len(reference_region)}")
        
        # 计算SUVR
        suvr_data = self.pet_data / reference_mean
        
        # 创建新的NIfTI图像
        suvr_img = nib.Nifti1Image(suvr_data, self.pet_img.affine, self.pet_img.header)
        
        print(f"SUVR计算完成，范围: [{suvr_data.min():.4f}, {suvr_data.max():.4f}]")
        
        return suvr_img, suvr_data
    
    def save_suvr_image(self, suvr_img, output_path):
        """保存SUVR图像"""
        nib.save(suvr_img, output_path)
        print(f"SUVR图像已保存至: {output_path}")

def main():
    """主函数示例"""
    # 初始化归一化器
    normalizer = PETNormalizerWithRegistration()
    
    # 加载图像（请替换为实际路径）
    pet_path = r"E:\sustain\20251111\w002_S_0413_AV1451_2017-06-21_16_18_14.0.nii"
    ref_mask_path = r"E:\sustain\20251111\Desikan-Killiany_MNI_cerebellumGM.nii"  # 0为背景，1为小脑灰质
    
    if not normalizer.load_images(pet_path, ref_mask_path):
        return
    
    # 检查图像兼容性
    if not normalizer.check_image_compatibility():
        print("图像不兼容，进行配准...")
        # 进行配准（使用您提供的MATLAB代码方法）
        if not normalizer.register_mask_to_pet():
            print("配准失败，使用原始mask")
    else:
        print("图像兼容，跳过配准步骤")
    
    # 计算SUVR
    try:
        suvr_img, suvr_data = normalizer.calculate_suvr(use_registered_mask=True)
        
        # 保存结果
        output_path = r"E:\sustain\20251111\11w002_S_0413_AV1451_2017-06-21_16_18_14.0.nii"
        normalizer.save_suvr_image(suvr_img, output_path)
        
    except Exception as e:
        print(f"SUVR计算错误: {e}")

# 使用示例
if __name__ == "__main__":
    main()