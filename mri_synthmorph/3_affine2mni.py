import registration
import os
import h5py
import numpy as np
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
import nibabel as nib

def network_space(im, shape, center=None):
    """Construct transform from network space to the voxel space of an image.

    Constructs a coordinate transform from the space the network will operate
    in to the zero-based image index space. The network space has isotropic
    1-mm voxels, left-inferior-anterior (LIA) orientation, and no shear. It is
    centered on the field of view, or that of a reference image. This space is
    an indexed voxel space, not world space.

    Parameters
    ----------
    im : surfa.Volume
        Input image to construct the transform for.
    shape : (3,) array-like
        Spatial shape of the network space.
    center : surfa.Volume, optional
        Center the network space on the center of a reference image.

    Returns
    -------
    out : tuple of (3, 4) NumPy arrays
        Transform from network to input-image space and its inverse, thinking
        coordinates.

    """
    old = im.geom
    new = sf.ImageGeometry(
        shape=shape,
        voxsize=1,
        rotation='LIA',
        center=old.center if center is None else center.geom.center,
        shear=None,
    )

    net_to_vox = old.world2vox @ new.vox2world
    vox_to_net = new.world2vox @ old.vox2world
    return net_to_vox.matrix, vox_to_net.matrix

def is_affine_shape(shape):
    """
    Determine whether the given shape (single-batch) represents an N-dimensional affine matrix of
    shape (M, N + 1), with `N in (2, 3)` and `M in (N, N + 1)`.

    Parameters:
        shape: Tuple or list of integers excluding the batch dimension.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False


def validate_affine_shape(shape):
    """
    Validate whether the input shape represents a valid affine matrix of shape (..., M, N + 1),
    where N is the number of dimensions, and M is N or N + 1. Throws an error if the shape is
    invalid.

    Parameters:
        shape: Tuple or list of integers.
    """
    ndim = shape[-1] - 1
    rows = shape[-2]
    if ndim not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, got {ndim}D')
    if rows not in (ndim, ndim + 1):
        raise ValueError(f'{ndim}D affine matrix must have {ndim} or {ndim + 1} rows, got {rows}.')

def affine_to_dense_shift(matrix, shape, shift_center=True, warp_right=None):
    """
    Convert N-dimensional (ND) matrix transforms to dense displacement fields.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply matrices to each index coordinate.
        3. Subtract grid.

    Parameters:
        matrix: Affine matrix of shape (..., M, N + 1), where M is N or N + 1. Can have any batch
            dimensions.
        shape: ND shape of the output space.
        shift_center: Shift grid to image center.
        warp_right: Right-compose the matrix transform with a displacement field of shape
            (..., *shape, N), with batch dimensions broadcastable to those of `matrix`.

    Returns:
        Dense shift (warp) of shape (..., *shape, N).

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # coordinate grid
    mesh = (tf.range(s, dtype=matrix.dtype) for s in shape)
    if shift_center:
        mesh = (m - 0.5 * (s - 1) for m, s in zip(mesh, shape))
    mesh = [tf.reshape(m, shape=(-1,)) for m in tf.meshgrid(*mesh, indexing='ij')]
    mesh = tf.stack(mesh)  # N x nb_voxels
    out = mesh

    # optionally right-compose with warp field
    if warp_right is not None:
        if not tf.is_tensor(warp_right) or warp_right.dtype != matrix.dtype:
            warp_right = tf.cast(warp_right, matrix.dtype)
        flat_shape = tf.concat((tf.shape(warp_right)[:-1 - ndims], (-1, ndims)), axis=0)
        warp_right = tf.reshape(warp_right, flat_shape)  # ... x nb_voxels x N
        out += tf.linalg.matrix_transpose(warp_right)  # ... x N x nb_voxels

    # compute locations, subtract grid to obtain shift
    out = matrix[..., :ndims, :-1] @ out + matrix[..., :ndims, -1:]  # ... x N x nb_voxels
    out = tf.linalg.matrix_transpose(out - mesh)  # ... x nb_voxels x N

    # restore shape
    shape = tf.concat((tf.shape(matrix)[:-2], (*shape, ndims)), axis=0)
    return tf.reshape(out, shape)  # ... x in_shape x N


def transformS(vol, loc_shift, interp_method='nearest', fill_value=None,
              shift_center=True, shape=None):
    """Apply affine or dense transforms to images in N dimensions.

    Essentially interpolates the input ND tensor at locations determined by
    loc_shift. The latter can be an affine transform or dense field of location
    shifts in the sense that at location x we now have the data from x + dx, so
    we moved the data.

    Parameters:
        vol: tensor or array-like structure  of size vol_shape or
            (*vol_shape, C), where C is the number of channels.
        loc_shift: Affine transformation matrix of shape (N, N+1) or a shift
            volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
            where C is the number of channels, and D is the dimensionality
            D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
            transform applies to all channels of the input tensor.
        interp_method: 'linear' or 'nearest'.
        fill_value: Value to use for points sampled outside the domain. If
            None, the nearest neighbors will be used.
        shift_center: Shift grid to image center when converting affine
            transforms to dense transforms. Assumes the input and output spaces are identical.
        shape: ND output shape used when converting affine transforms to dense
            transforms. Includes only the N spatial dimensions. If None, the
            shape of the input image will be used. Incompatible with `shift_center=True`.

    Returns:
        Tensor whose voxel values are the values of the input tensor
        interpolated at the locations defined by the transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    if shape is not None and shift_center:
        raise ValueError('`shape` option incompatible with `shift_center=True`')

    # convert data type if needed
    ftype = tf.float32
    if not tf.is_tensor(vol) or not vol.dtype.is_floating:
        vol = tf.cast(vol, ftype)
    if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
        loc_shift = tf.cast(loc_shift, ftype)

    # convert affine to location shift (will validate affine shape)
    if is_affine_shape(loc_shift.shape):
        loc_shift = affine_to_dense_shift(loc_shift,
                                          shape=vol.shape[:-1] if shape is None else shape,
                                          shift_center=shift_center)

    # parse spatial location shape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing='ij')  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def transform(im, trans, shape=None, normalize=False, batch=False):
    """Apply a spatial transform to 3D image voxel data in dimensions.

    Applies a transformation matrix operating in zero-based index space or a
    displacement field to an image buffer.

    Parameters
    ----------
    im : surfa.Volume or NumPy array or TensorFlow tensor
        Input image to transform, without batch dimension.
    trans : array-like
        Transform to apply to the image. A matrix of shape (3, 4), a matrix
        of shape (4, 4), or a displacement field of shape (*space, 3),
        without batch dimension.
    shape : (3,) array-like, optional
        Output shape used for converting matrices to dense transforms. None
        means the shape of the input image will be used.
    normalize : bool, optional
        Min-max normalize the image intensities into the interval [0, 1].
    batch : bool, optional
        Prepend a singleton batch dimension to the output tensor.

    Returns
    -------
    out : float TensorFlow tensor
        Transformed image with a trailing feature dimension.

    """
    # Add singleton feature dimension if needed.
    if tf.rank(im) == 3:
        im = im[..., tf.newaxis]

    out = transformS(
        im, trans, fill_value=0, shift_center=False, shape=shape,
    )

    if normalize:
        out -= tf.reduce_min(out)
        out /= tf.reduce_max(out)

    if batch:
        out = out[tf.newaxis, ...]

    return out

def process_folder_pvc(input_dir, output_dir, fixed_file, start_name, dest_name, in_shape=(192,) * 3):
    """
    遍历输入文件夹，处理每个子文件夹中的 FDG_CT.nii.gz 文件，
    并将输出文件保存到输入文件所在的文件夹。

    :param input_dir: 输入的主文件夹路径
    :param fixed_file: 固定的文件路径 (MNI152_T1_1mm.nii)
    :param in_shape: 网络输入形状，默认为 (192, 192, 192)
    """
    # 配置线程并行参数
    tf.config.threading.set_inter_op_parallelism_threads(12)
    tf.config.threading.set_intra_op_parallelism_threads(12)

    count = 0

    # 遍历子文件夹
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        files.sort()
        for file in files:
            if file == start_name:  # 只处理目标文件
                moving_file = os.path.join(root, file)
                print(f"Processing: {moving_file}")

                # 加载输入数据
                mov = sf.load_volume(moving_file)
                fix = sf.load_volume(fixed_file)
                if not len(mov.shape) == len(fix.shape) == 3:
                    sf.system.fatal("Input images are not single-frame volumes")

                # 预处理
                center = fix
                net_to_mov, mov_to_net = network_space(mov, shape=in_shape, center=center)
                # net_to_fix, fix_to_net = network_space(fix, shape=in_shape)

                mov_to_ras = mov.geom.vox2world.matrix
                # fix_to_ras = fix.geom.vox2world.matrix

                inputs = (
                    transform(mov, net_to_mov, shape=in_shape, normalize=False, batch=True),
                    # transform(fix, net_to_fix, shape=in_shape, normalize=True, batch=True),
                )
                try:
                    # 输出路径设置为输入文件所在文件夹
                    inp_1 = os.path.join(output_dir, os.path.basename(os.path.dirname(root)), dest_name)
                    # inp_2 = os.path.join(root, "rMNI152_T1_1mm.nii.gz")
                    geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
                    # geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)

                    # 保存结果
                    sf.Volume(inputs[0][0], geom_1).save(inp_1)
                    # sf.Volume(inputs[1][0], geom_2).save(inp_2)

                    print(f"Saved processed file to: {root}")
                    count += 1
                except Exception as e:
                    print(f"fail to process file {inp_1}. Error is {e}")

    print(f'total process number is {count}')

def process_folder(input_dir, fixed_file, start_name, dest_name, in_shape=(192,) * 3):
    """
    遍历输入文件夹，处理每个子文件夹中的 FDG_CT.nii.gz 文件，
    并将输出文件保存到输入文件所在的文件夹。

    :param input_dir: 输入的主文件夹路径
    :param fixed_file: 固定的文件路径 (MNI152_T1_1mm.nii)
    :param in_shape: 网络输入形状，默认为 (192, 192, 192)
    """
    # 配置线程并行参数
    tf.config.threading.set_inter_op_parallelism_threads(12)
    tf.config.threading.set_intra_op_parallelism_threads(12)

    count = 0

    # 遍历子文件夹
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        files.sort()
        for file in files:
            if file == start_name:  # 只处理目标文件
                moving_file = os.path.join(root, file)
                print(f"Processing: {moving_file}")

                # 加载输入数据
                mov = sf.load_volume(moving_file)
                fix = sf.load_volume(fixed_file)
                if not len(mov.shape) == len(fix.shape) == 3:
                    sf.system.fatal("Input images are not single-frame volumes")

                # 预处理
                center = fix
                net_to_mov, mov_to_net = network_space(mov, shape=in_shape, center=center)
                # net_to_fix, fix_to_net = network_space(fix, shape=in_shape)

                mov_to_ras = mov.geom.vox2world.matrix
                # fix_to_ras = fix.geom.vox2world.matrix

                inputs = (
                    transform(mov, net_to_mov, shape=in_shape, normalize=False, batch=True),
                    # transform(fix, net_to_fix, shape=in_shape, normalize=True, batch=True),
                )
                try:
                    # 输出路径设置为输入文件所在文件夹
                    inp_1 = os.path.join(root, dest_name)
                    # inp_2 = os.path.join(root, "rMNI152_T1_1mm.nii.gz")
                    geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
                    # geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)

                    # 保存结果
                    sf.Volume(inputs[0][0], geom_1).save(inp_1)
                    # sf.Volume(inputs[1][0], geom_2).save(inp_2)

                    print(f"Saved processed file to: {root}")
                    count += 1
                except Exception as e:
                    print(f"fail to process file {inp_1}. Error is {e}")

    print(f'total process number is {count}')


# 单例处理
moving_file = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/FDG_CT/MSA154_JIANG_WEISHUANG_41803/wrT1mSUV_FDG.nii"
fixed_file = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/voxelmorph/rspm152.nii.gz"
out_dir = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/FDG_CT/MSA154_JIANG_WEISHUANG_41803"

# Parse arguments.
# in_shape = (192,) * 3
in_shape = (192,) * 3
# Threading.
tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)

# Input data.
mov = sf.load_volume(moving_file)
fix = sf.load_volume(fixed_file)
if not len(mov.shape) == len(fix.shape) == 3:
    sf.system.fatal('input images are not single-frame volumes')

center = fix
net_to_mov, mov_to_net = network_space(mov, shape=in_shape, center=center)
# net_to_fix, fix_to_net = network_space(fix, shape=in_shape)

mov_to_ras = mov.geom.vox2world.matrix
fix_to_ras = fix.geom.vox2world.matrix

inputs = (
        transform(mov, net_to_mov, shape=in_shape, normalize=False, batch=True),
        # transform(fix, net_to_fix, shape=in_shape, normalize=True, batch=True),
    )

os.makedirs(out_dir, exist_ok=True)
inp_1 = os.path.join(out_dir, 'rwSpm_T1_rmSUV_FDG.nii')
# inp_2 = os.path.join(out_dir, 'rMNI152_T1_1mm.nii.gz')
geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
# geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)
sf.Volume(inputs[0][0], geom_1).save(inp_1)
# sf.Volume(inputs[1][0], geom_2).save(inp_2)




# 批量处理_for_pvc_project
# input_dir = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/T1_recon_all/"  # 包含子文件夹的主文件夹路径
# output_dir = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/PVE_research/Dataset/"
# fixed_file = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/voxelmorph/rspm152.nii.gz"  # 固定的文件路径
# start_name = "aparc.DKTatlas+aseg.mgz"  # 需要进行处理的文件名
# dest_name = "rDKT.nii.gz"  # 处理后的文件名
# process_folder_pvc(input_dir,output_dir, fixed_file, start_name, dest_name)


# 批量处理
# input_dir = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/T1_recon_all/"  # 包含子文件夹的主文件夹路径
# fixed_file = "/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/voxelmorph/rspm152.nii.gz"  # 固定的文件路径
# start_name = "aparc.DKTatlas+aseg.mgz"  # 需要进行处理的文件名
# dest_name = "rDKT.nii.gz"  # 处理后的文件名
# process_folder_pvc(input_dir, fixed_file, start_name, dest_name)



# # chuli4wei
# moving_file = r"/usr/local/MATLAB/R2022b/toolbox/spm12/tpm/TPM.nii"
# fixed_file = r"/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/synthmorph/rMNI152_T1_1mm.nii.gz"
# out_dir = r'/media/liu/cfae391f-32f2-485c-9dda-6f04b4eb55b9/HKJ/synthmorph/resized_TPM.nii'
#
# # Parse arguments.
# in_shape = (192,) * 3
#
# # Threading.
# tf.config.threading.set_inter_op_parallelism_threads(12)
# tf.config.threading.set_intra_op_parallelism_threads(12)
#
# # Input data.
# img = sf.load_volume(moving_file)
# # 提取每个尺寸为(121, 145, 121)的图像
# images = [img[:, :, :, i] for i in range(6)]
#
# fix = sf.load_volume(fixed_file)
#
# processed_images = []
#
# for i in range(6):
#     if not len(images[i].shape) == len(fix.shape) == 3:
#         sf.system.fatal('input images are not single-frame volumes')
#
#     center = fix
#     net_to_mov, mov_to_net = network_space(images[i], shape=in_shape, center=center)
#
#     mov_to_ras = images[i].geom.vox2world.matrix
#     fix_to_ras = fix.geom.vox2world.matrix
#
#     inputs = (
#         transform(images[i], net_to_mov, shape=in_shape, normalize=False, batch=True),
#     )
#     geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
#
#     processed_images.append(sf.Volume(inputs[0][0], geom_1))
#
#
# # Combine processed images into a single NIfTI file
# combined_data = np.stack(processed_images, axis=-1)  # Combine along a new 4th dimension
#
# # Define affine matrix for NIfTI file. Using the affine from the fixed file for consistency.
# affine = fix.geom.vox2world.matrix
#
# # Create NIfTI image
# nii_image = nib.Nifti1Image(combined_data, affine)
#
# # Save the combined NIfTI file
# nib.save(nii_image, out_dir)
#
# print('done!')



