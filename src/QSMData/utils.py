"""
Utility functions for transforming susceptibility distributions. These functions are based on ones written by 
Jamie McClelland for use in the image registration part of the coursework for module MPHY0025 (IPMI), 2022.

"""

import numpy as np
from scipy import interpolate


# def matrixFor3dAffineFromParams(aff_params, nii_img: nib.nifti1.Nifti1Image=None):
#     """
#     function to calculate the matrix for a 3D affine transformation from the
#     affine parameters

#     SYNTAX:
#         aff_mat = matrixFor3dAffineFromParams(aff_params)
#         aff_mat = matrixFor3dAffineFromParams(aff_params, nii_hdr)

#     INPUTS:
#         aff_params - a 1D numpy.ndarray with 15 elements giving the affine parameters
#             the order of the parameters should be:
#                 [tx ty tz rx ry rz scx scy scz shxy shxz shyx shyz shzx shzy]
#             where:
#                 tx/ty/tz - the translation in x/y/z
#                 rx/ry/rz - the rotation about the x/y/z axis in degrees
#                 scx/scy/scz - the scaling along the x/y/z axis
#                 shxy - the shearing in the x axis relative to the y axis
#                 shxz/shyx/shyz/shzx/shzy - similar definition as above for 
#                 other axis combinations
#         nii_img - a nibabel.nifti1.Nifti1Image object as read by Nibabel's load function

#     OUTPUTS:
#         aff_mat - a 4 x 4 matrix representing the 3D affine transformation in
#         homogeneous coordinates

#     NOTES:
#         the transformations corresponding to the individual parameters are
#         applied in the following order:
#         rotation about z axis,
#         rotation about y axis,
#         rotation about x axis,
#         scalings,
#         shearings,
#         3D translation
   
#         if nii_hdr is provided the rotations, scalings, and shearings
#         are performed around the centre of the image defined in nii_hdr 
#     """
#     # copy the individual parameters from aff_params so easier to keep track of
#     #which parameter is which
#     tx = aff_params[0]
#     ty = aff_params[1]
#     tz = aff_params[2]
#     rx = aff_params[3]
#     ry = aff_params[4]
#     rz = aff_params[5]
#     scx = aff_params[6]
#     scy = aff_params[7]
#     scz = aff_params[8]
#     shxy = aff_params[9]
#     shxz = aff_params[10]
#     shyx = aff_params[11]
#     shyz = aff_params[12]
#     shzx = aff_params[13]
#     shzy = aff_params[14]
    
#     #convert rotations from degrees to radians
#     rx_rad = rx * np.pi / 180
#     ry_rad = ry * np.pi / 180
#     rz_rad = rz * np.pi / 180

#     srx = np.sin(rx_rad)
#     sry = np.sin(ry_rad)
#     srz = np.sin(rz_rad)

#     crx = np.cos(rx_rad)
#     cry = np.cos(ry_rad)
#     crz = np.cos(rz_rad)

#     #form matrices for individual transformations
#     #ADD CODE HERE
#     T_t = np.array(
#         [
#             [1, 0, 0, tx],
#             [0, 1, 0, ty],
#             [0, 0, 1, tz],
#             [0, 0, 0,  1]
#         ]
#     )
#     T_rx = np.array(
#         [
#             [1,   0,    0, 0],
#             [0, crx, -srx, 0],
#             [0, srx,  crx, 0],
#             [0,   0,    0, 1]
#         ]
#     )
#     T_ry = np.array(
#         [
#             [cry, 0, -sry, 0],
#             [  0, 1,    0, 0],
#             [sry, 0,  cry, 0],
#             [  0, 0,    0, 1]
#         ]
#     )
#     T_rz = np.array(
#         [
#             [crz, -srz, 0, 0],
#             [srz,  crz, 0, 0],
#             [  0,    0, 1, 0],
#             [  0,    0, 0, 1]
#         ]
#     )
#     T_sc = np.array(
#         [
#             [scx,   0,   0, 0],
#             [  0, scy,   0, 0],
#             [  0,   0, scz, 0],
#             [  0,   0,   0, 1]
#         ]
#     )
#     T_sh = np.array(
#         [
#             [   1, shyx, shzx, 0],
#             [shxy,    1, shzy, 0],
#             [shxz, shyz,    1, 0],
#             [   0,    0,    0, 1]
#         ]
#     )
    
#     #if nii_img provided make rotations, scalings, and shearings about centre
#     #of image
#     if nii_img:
#         #create matrices which translate to the centre of the image and back
#         shape = np.array(nii_img.shape)
#         zooms = np.array(nii_img.header.get_zooms())

#         real_shape = shape * zooms

#         cx = real_shape[0] / 2
#         cy = real_shape[1] / 2
#         cz = real_shape[2] / 2

#         T_pre = np.array(
#             [
#                 [1, 0, 0, -cx],
#                 [0, 1, 0, -cy],
#                 [0, 0, 1, -cz],
#                 [0, 0, 0,  1]
#             ]
#         )
#         T_post = np.array(
#             [
#                 [1, 0, 0, cx],
#                 [0, 1, 0, cy],
#                 [0, 0, 1, cz],
#                 [0, 0, 0,   1]
#             ]
#         )
#     else:
#         T_pre = np.eye(4)
#         T_post = np.eye(4)
    
#     #compose matrices
#     #ADD CODE HERE
#     aff_mat = T_t @ T_post @ T_sh @ T_sc @ T_rx @ T_ry @ T_rz @ T_pre
#     return aff_mat


# def calcSimForAffParams(source_nii: nib.nifti1.Nifti1Image, target_nii: nib.nifti1.Nifti1Image, aff_params, about_img_centre, similarity):
#     """
#     function to calculate the value of the similarity measure between a 
#     target image and a source image transformed by an affine transformation

#     SYNTAX:
#         sim_val = calcSimForAffParams(sourc_nii, target_nii, aff_params, about_img_centre, similarity)
#         [sim_val, resamp_image] = calcSimForAffParams(...)

#     INPUTS:
#         source_nii - the source image as a nibabel.nifti1.Nifti1Image object 
#         target_nii - the target image as a nibabel.nifti1.Nifti1Image object 
#         aff_params - a 1D numpy.ndarray with 15 elements giving the affine parameters
#             used to transform the source image into the space of the target
#             image. the order of the parameters should be:
#                 [tx ty tz rx ry rz scx scy scz shxy shxz shyx shyz shzx shzy]
#             where:
#                 tx/ty/tz - the translation in x/y/z
#                 rx/ry/rz - the rotation about the x/y/z axis in degrees
#                 scx/scy/scz - the scaling along the x/y/z axis
#                 shxy - the shearing in the x axis relative to the y axis
#                 shxz/shyx/shyz/shzx/shzy - similar definition as above for 
#                 other axis combinations
#         about_img_centre - a boolean value. If true the rotations, scalings,
#             and shearings should be performed about the centre of the target
#             image, if false they are performed about the origin (0,0,0) in
#             world coordinates
#         similarity - the similarity measure to be calculated between the target
#             image and the transformed source image. can be one of:
#                 'msd' - mean squared difference
#                 'ncc' - normalised cross correlation
#                 'nmi' - normalised mutual information

#     OUTPUTS:
#         sim_val - the value of the similarity measure between the target image
#             and the transformed source image
#         resamp_image - the source image resampled into the space of the target
#             image using the affine transformation, as a 3D numpy.ndarray.
#             sim_val is calculated between resamp_image and target_image    
#     """

#     #ADD CODE BELOW
#     if about_img_centre:
#         aff_matrix = matrixFor3dAffineFromParams(aff_params, target_nii)
#     else:
#         aff_matrix = matrixFor3dAffineFromParams(aff_params)

#     def_field = defFieldFrom3dAffineMatrixForNifti(aff_matrix, target_nii)
#     resamp_image = resampNiftiWithDefField(source_nii, def_field)

#     if similarity == 'msd':
#         #calculate mean squared error
#         sim_val = calcMSD(target_nii.get_fdata(), resamp_image)
#     elif similarity == 'ncc':
#         #calculate the normalised cross correlation
#         sim_val = calcNCC(target_nii.get_fdata(), resamp_image)
#     elif similarity == 'nmi':
#         #calculate the normalised mutual information
#         H_tr, H_target, H_resampled = calcEntropies(target_nii.get_fdata(), resamp_image)
#         sim_val = (H_target + H_resampled) / H_tr
#     else:
#         raise ValueError('similarity measure must be \'msd\', \'ncc\', or \'nmi\'')
    
#     return sim_val, resamp_image
    

# def calcMSD(A, B):
#   """
#     function to calculate the mean of squared differences between two images

#     SYNTAX:
#         MSD = calcMSD(A, B)

#     INPUTS:
#         A - an image stored as a numpy.ndarray
#         B - an image stored as a numpy.ndarray. B must the the same size as A

#     OUTPUTS:
#         MSD - the value of the mean of squared differences

#     NOTE:
#         if either of the images contain NaN values these pixels should be 
#         ignored when calculating the SSD.
#   """
#   # use nansum function to find sum of squared differences ignoring NaNs
#   return np.nanmean((A-B)*(A-B))


# def calcNCC(A, B):
#     """
#     function to calculate the normalised cross correlation between two images

#     SYNTAX:
#         NCC = calcNCC(A, B)

#     INPUTS:
#         A - an image stored as a numpy.ndarray
#         B - an image stored as a numpy.ndarray. B must the the same size as A

#     OUTPUTS:
#         NCC - the value of the normalised cross correlation

#     NOTE
#         if either of the images contain NaN values these pixels should be
#         ignored when calculating the NCC.
#     """
    
#     # REMOVE PIXELS THAT CONTAIN NAN IN EITHER IMAGE
#     nan_inds = np.logical_or(np.isnan(A), np.isnan(B))
#     A = A[np.logical_not(nan_inds)]
#     B = B[np.logical_not(nan_inds)]
    
#     # CALCULATE MEAN AND STD DEV OF EACH IMAGE
#     mu_A = np.mean(A)
#     mu_B = np.mean(B)
#     sig_A = np.std(A)
#     sig_B = np.std(B)
    
#     # CALCULATE AND RETURN NCC
#     NCC = np.sum((A - mu_A) * (B - mu_B)) / (A.size * sig_A * sig_B)
#     return NCC


# def calcEntropies(A, B, num_bins = [32,32]):
#   """
#     function to calculate the joint and marginal entropies for two images

#     SYNTAX:
#         [H_AB, H_A, H_B] = calcEntropies(A, B)
#         [H_AB, H_A, H_B] = calcEntropies(A, B, num_bins)

#     INPUTS:
#         A - an image stored as a numpy.ndarray
#         B - an image stored as a numpy.ndarray. B must the the same size as A
#         num_bins - a 2 element vector specifying the number of bins to in the
#             joint histogram for each image
#             default = 32, 32

#     OUTPUTS:
#         H_AB - the joint entropy between A and B
#         H_A - the marginal entropy in A
#         H_B - the marginal entropy in B

#     NOTE:
#         if either of the images contain NaN values these pixels should be
#         ignored when calculating the entropies.
#   """
#   #first remove NaNs and flatten
#   nan_inds = np.logical_or(np.isnan(A), np.isnan(B))
#   A = A[np.logical_not(nan_inds)]
#   B = B[np.logical_not(nan_inds)]
  
#   #use histogram2d function to generate joint histogram, an convert to
#   #probabilities
#   joint_hist, _, _ = np.histogram2d(A, B, bins = num_bins)
#   probs_AB = joint_hist / np.sum(joint_hist)
  
#   #calculate marginal probability distributions for A and B
#   probs_A = np.sum(probs_AB, axis=1)
#   probs_B = np.sum(probs_AB, axis=0)
    
#   #calculate joint entropy and marginal entropies
#   #note, when taking sums must check for nan values as
#   #0 * log(0) = nan
#   H_AB = -np.nansum(probs_AB * np.log(probs_AB))
#   H_A = -np.nansum(probs_A * np.log(probs_A))
#   H_B = -np.nansum(probs_B * np.log(probs_B))
#   return H_AB, H_A, H_B


# def getWorldCoordsFromNifti1Image(nii_img):
#     """
#     function to calculate the world coordinates of all the voxels in a 3D nifti image

#     SYNTAX:
#         world_coords = getWorldCoordsFromNifti1Image(nii_img)

#     INPUTS:
#         nii_img - a nibabel.nifti1.Nifti1Image object as read by Nibabel's load function

#     OUTPUTS:
#         world_coords - the world coordinates as homogenous coordinates in a
#             4 x num_vox matrix
#     """
#     #form 4 x num_vox matrix with voxel coordinates for all voxels in
#     #homongenous coordinates
#     vox_X, vox_Y, vox_Z = np.mgrid[0:nii_img.shape[0], 0:nii_img.shape[1], 0:nii_img.shape[2]]
#     vox_coords = np.stack((vox_X.flatten(), vox_Y.flatten(), vox_Z.flatten(), np.ones(np.size(vox_X))))
    
#     #use affine to transform voxel coordinates to world coordinates
#     world_coords = nii_img.affine @ vox_coords
#     return world_coords


# def defFieldFrom3dAffineMatrixForNifti(aff_mat, nii_img):
#     """
#     function to create a 3D deformation field from an affine matrix 

#     SYNTAX:
#         def_field = defFieldFrom3dAffineMatrixForNifti(aff_mat, nii_img)

#     INPUTS:
#         aff_mat - a 4 x 4 matrix representing the 3D affine transformation
#         nii_img - a nibabel.nifti1.Nifti1Image object as read by Nibabel's load function

#     OUTPUTS:
#         def_field - the deformation field stored as a 4D matrix

#     NOTES:
#         the size of the first 3 dimensions of def_field will be the same as the size
#         of the nibabel.nifti1.Nifti1Image object
#         the 4th dimension has a size of 3, and gives the x, y, and z coordinate
#         of the voxel in world coordinates after the affine transformation has
#         been applied
#     """
#     #calculate world coordinates
#     world_coords = getWorldCoordsFromNifti1Image(nii_img)
    
#     #apply transformation to world coordinates
#     trans_coords = aff_mat @ world_coords
    
#     #reshape into deformation field
#     def_field_x = np.reshape(trans_coords[0,:], nii_img.shape)
#     def_field_y = np.reshape(trans_coords[1,:], nii_img.shape)
#     def_field_z = np.reshape(trans_coords[2,:], nii_img.shape)
#     def_field = np.stack((def_field_x, def_field_y, def_field_z), axis=-1)
#     return def_field


# def resampNiftiWithDefField(source_nii, def_field, interp_method = 'linear', pad_value=np.NaN) -> np.ndarray:
#     """
#     function to resample a 3D volume loaded from a nifti file with a
#     deformation field

#     SYNTAX:
#         resamp_vol = resampNiftiWithDefField(source_nii, def_field)
#         resamp_vol = resampNiftiWithDefField(source_nii, def_field, interp_method)
#         resamp_vol = resampNiftiWithDefField(source_nii, def_field, interp_method, pad_value)

#     INPUTS:
#         source_nii - the source image as a nibabel.nifti1.Nifti1Image object 
#         def_field - the deformation field stored as a 4D matrix
#         interp_method - any of the interpolation methods accepted by interpn
#             function ('linear', 'nearest')
#             default = 'linear'
#         pad_value - the value to assign to pixels that are outside the source image
#             default = NaN

#     OUTPUTS:
#         resamp_vol - the resampled volume as a 3D numpy.ndarray

#     NOTES:
#         the deformation field should be a 4D matrix, where the size of the
#         first 3 dimensions defines the size of resamp_vol, and the 4th
#         dimension has a size of 3.
#         def_field(:,:,:,1) is x coords of the transformed voxels
#         def_field(:,:,:,2) is y coords of the transformed voxels
#         def_field(:,:,:,3) is z coords of the transformed voxels
#         values should be world coordinates 
#     """
#     #need to convert def_field from world coordinates into voxel coordinates in
#     #the source image
#     def_field_x = def_field[:,:,:,0]
#     def_field_y = def_field[:,:,:,1]
#     def_field_z = def_field[:,:,:,2]
#     def_field_world_hc = np.stack((def_field_x.flatten(), def_field_y.flatten(), def_field_z.flatten(), np.ones(np.size(def_field_x))))
#     def_field_vox_hc = np.linalg.inv(source_nii.affine) @ def_field_world_hc
#     def_field_vox_x = np.reshape(def_field_vox_hc[0,:], def_field_x.shape)
#     def_field_vox_y = np.reshape(def_field_vox_hc[1,:], def_field_x.shape)
#     def_field_vox_z = np.reshape(def_field_vox_hc[2,:], def_field_x.shape)
#     def_field_vox = np.stack((def_field_vox_x, def_field_vox_y, def_field_vox_z), axis=-1)
    
    
#     #resample source_vol using interpn function
#     x_coords = np.arange(source_nii.shape[0], dtype = 'float')
#     y_coords = np.arange(source_nii.shape[1], dtype = 'float')
#     z_coords = np.arange(source_nii.shape[2], dtype = 'float')
#     resamp_vol = interpolate.interpn((x_coords, y_coords, z_coords), source_nii.get_fdata(), def_field_vox, bounds_error=False, fill_value=pad_value, method=interp_method)
#     return resamp_vol


def deformation_field_from_affine_matrix(
    aff_mat: np.ndarray, 
    shape: tuple[int]
) -> np.ndarray:
    """Produce a deformation field from an affine transformation matrix.
    
    """
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    coords = np.stack((x.flatten(), y.flatten(), z.flatten(), np.ones(x.size)))

    trans_coords = aff_mat @ coords

    def_field_x = np.reshape(trans_coords[0, :], shape)
    def_field_y = np.reshape(trans_coords[1, :], shape)
    def_field_z = np.reshape(trans_coords[2, :], shape)
    
    def_field = np.stack((def_field_x, def_field_y, def_field_z), axis=-1)
    return def_field


def resample_with_deformation_field(
    img: np.ndarray,
    def_field: np.ndarray,
    interp_method: str = 'linear',
    pad_value: float = np.NaN
) -> np.ndarray:
    """Resample an image with a given deformation field.
    
    """
    shape = img.shape

    x = np.arange(shape[0], dtype='float')
    y = np.arange(shape[1], dtype='float')
    z = np.arange(shape[2], dtype='float')

    resamp_img = interpolate.interpn(
        (x, y, z), 
        img, 
        def_field, 
        bounds_error=False,
        fill_value=pad_value,
        method=interp_method
    )
    return resamp_img


def rand_angle(rng: np.random.Generator) -> tuple[float]:
    """Generates a random point on the surface of a sphere.

    Parameters
    ----------
    rng: np.random.Generator
        A numpy `Generator` object for producing pseudo-random numbers

    Returns
    -------
    (theta, phi): tuple[float, float]
        A point randomly distrubuted over the surface of a sphere.
    
    """
    theta = 2 * np.pi * rng.random()
    u = rng.random()

    phi = np.arccos(1 - 2 * u)
    return (theta, phi)


def rand_translation(
    rng: np.random.Generator, *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float
) -> tuple[float]:
    xrange = xmax - xmin
    yrange = ymax - ymin
    zrange = zmax - zmin

    x = (xrange * rng.random()) + xmin
    y = (yrange * rng.random()) + ymin
    z = (zrange * rng.random()) + zmin

    return (x, y, z)


def rand_scale(
    rng: np.random.Generator, *,
    xmax: float = 1,
    xmin: float = 0,
    ymax: float = 1,
    ymin: float = 0,
    zmax: float = 1,
    zmin: float = 0
) -> tuple[float]:
    xrange = xmax - xmin
    yrange = ymax - ymin
    zrange = zmax - zmin

    x = (xrange * rng.random()) + xmin
    y = (yrange * rng.random()) + ymin
    z = (zrange * rng.random()) + zmin

    return (x, y, z)


def rand_shear(
    rng: np.random.Generator, *,
    xymax: float = 1,
    xymin: float = -1,
    xzmax: float = 1,
    xzmin: float = -1,
    yxmax: float = 1,
    yxmin: float = -1,
    yzmax: float = 1,
    yzmin: float = -1,
    zxmax: float = 1,
    zxmin: float = -1,
    zymax: float = 1,
    zymin: float = -1
) -> tuple[float]:
    xyrange = xymax - xymin
    xzrange = xzmax - xzmin
    yxrange = yxmax - yxmin
    yzrange = yzmax - yzmin
    zxrange = zxmax - zxmin
    zyrange = zymax - zymin

    xy = (xyrange * rng.random()) + yxmin
    xz = (xzrange * rng.random()) + xzmin
    yx = (yxrange * rng.random()) + yxmin
    yz = (yzrange * rng.random()) + yzmin
    zx = (zxrange * rng.random()) + zxmin
    zy = (zyrange * rng.random()) + zymin

    return (xy, xz, yx, yz, zx, zy)