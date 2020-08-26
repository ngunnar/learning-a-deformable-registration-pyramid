import SimpleITK as sitk
import pyelastix
import numpy as np
import cv2
from dipy.viz import regtools
from dipy.data import get_fnames
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import copy

from skimage.metrics import structural_similarity

# B-spline with MSE
def bspline_intra_modal_registration(fixed_image, moving_image, metric,fixed_image_mask=None):
    R = sitk.ImageRegistrationMethod()    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, transformDomainMeshSize = mesh_size, order=3)    
    R.SetInitialTransform(initial_transform)
    
    # Get user-specified metric
    if (metric == "mi"):
        R.SetMetricAsMattesMutualInformation()
    elif (metric == "mse"):
        R.SetMetricAsMeanSquares()
    elif (metric == "corr"):
        R.SetMetricAsCorrelation()
    else:
        raise Exception("No metric specified")
    
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
        
    # Multi-resolution framework.
    # Pyramid
    R.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # Smooth Regularization
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Interpolation to next level
    R.SetInterpolator(sitk.sitkLinear)
    # Optimizer
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)    
    return R.Execute(fixed_image, moving_image)

def dvf_registration(fixed_image, moving_image, metric):
    R = sitk.ImageRegistrationMethod()
    # Get user-specified metric
    if (metric == "mi"):
        R.SetMetricAsMattesMutualInformation()
    elif (metric == "mse"):
        R.SetMetricAsMeanSquares()
    elif (metric == 'demon'):
        R.SetMetricAsDemons(10)
    elif (metric == "corr"):
        R.SetMetricAsCorrelation()
    else:
        raise Exception("No metric specified")
    
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)
    R.SetOptimizerScalesFromPhysicalShift()

    # Displacement field transform
    toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
    toDisplacementFilter.SetReferenceImage(fixed_image)
    DVF = sitk.DisplacementFieldTransform(
        toDisplacementFilter.Execute(sitk.Transform(moving_image.GetDimension(),sitk.sitkIdentity)))
    DVF.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,varianceForTotalField=1)
    
    R.SetInitialTransform(DVF)
    #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    # Multi-resolution setup
    # Pyramid
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    # Smooth
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Interpolation to next level
    R.SetInterpolator(sitk.sitkLinear)
    # Optimizer
    R.SetOptimizerAsGradientDescent(learningRate=10,numberOfIterations=200,convergenceWindowSize=15)    
    return R.Execute(fixed_image,moving_image)
'''
def elastix_transform(fixed_image, moving_image):
    # Get params and change a few values
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 10
    print(params.as_dict())
    # Apply the registration (im1 and im2 can be 2D or 3D)
    im2_deformed, field = pyelastix.register(sitk.GetArrayFromImage(moving_image), sitk.GetArrayFromImage(fixed_image), params)
    
    size = fixed_image.GetSize()
    f = np.zeros(size + (2,))
    f[:,:,0] = field[0]
    f[:,:,1] = field[1]

    transform_img = sitk.GetImageFromArray(f, isVector=True)
    transform_img = sitk.Cast(transform_img, sitk.sitkVectorFloat64)

    transform = sitk.DisplacementFieldTransform(transform_img.GetDimension())    
    transform.SetDisplacementField(transform_img)
    assert np.sum((f != sitk.GetArrayFromImage(transform.GetDisplacementField())).astype(int)) == 0
    return transform
'''
# Farnebäck
def cv2_farneback(fixed_image, moving_image):
    flow = cv2.calcOpticalFlowFarneback(prev = moving_image, next= fixed_image,
                                        flow = None, pyr_scale = 0.5,
                                        levels = 5,  winsize = 15,
                                        iterations = 10, poly_n = 5, poly_sigma = 1.2, flags = 0)
    t = -flow
    transform_img = sitk.GetImageFromArray(t, isVector=True)
    transform_img = sitk.Cast(transform_img, sitk.sitkVectorFloat64)

    transform = sitk.DisplacementFieldTransform(transform_img.GetDimension())
    transform.SetDisplacementField(transform_img)
    assert np.sum((flow != -sitk.GetArrayFromImage(transform.GetDisplacementField())).astype(int)) == 0
    return transform

#Symmetric Diffeomorphic Registration using dipy and SSD
def dipy_transform(fixed_image, moving_image, metric):
    dim = fixed_image.GetDimension()
    # Metric SSDMetric, CCMetric, EMMetric
    # Get user-specified metric
    if (metric == "em"):
        metric = EMMetric(dim)
    elif (metric == "mse"):
        metric = SSDMetric(dim)
    elif (metric == "corr"):
        metric = CCMetric(dim)
    else:
        raise Exception("No metric specified")    
    # Pyramid
    level_iters = [200, 100, 50, 25]

    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter = 50)    
    mapping = sdr.optimize(sitk.GetArrayFromImage(fixed_image), sitk.GetArrayFromImage(moving_image))
    flow = mapping.get_forward_field()
    f = copy.deepcopy(flow)
    f[:,:,0] = flow[:,:,1]
    f[:,:,1] = flow[:,:,0]

    transform_img = sitk.GetImageFromArray(f, isVector=True)
    transform_img = sitk.Cast(transform_img, sitk.sitkVectorFloat64)

    transform = sitk.DisplacementFieldTransform(transform_img.GetDimension())
    transform.SetDisplacementField(transform_img)
    assert np.sum((f != sitk.GetArrayFromImage(transform.GetDisplacementField())).astype(int)) == 0
    return transform

def get_score(fixed_image, warped_image):
    f = sitk.GetArrayFromImage(fixed_image)
    w = sitk.GetArrayFromImage(warped_image)
    (score, diff) = structural_similarity(f, w, data_range=w.max() - w.min(), full=True)
    mse = np.linalg.norm(f - w)
    return score, mse

if __name__ == "__main__":
    import time
    fixedImage = "/data/head/img/0001.png"
    movingImage = "/data/head/img/0002.png"
    moving_image = sitk.ReadImage(movingImage, sitk.sitkFloat32)
    moving_image = sitk.Extract(moving_image, (moving_image.GetWidth(), moving_image.GetHeight(), 0), (0, 0, 0))
    fixed_image = sitk.ReadImage(fixedImage, sitk.sitkFloat32)
    fixed_image = sitk.Extract(fixed_image, (fixed_image.GetWidth(), fixed_image.GetHeight(), 0), (0, 0, 0))
    
    t = time.time()
    transform_ffd_mse = bspline_intra_modal_registration(fixed_image, moving_image, 'mse')
    t_ffd_mse = time.time() - t
    
    t = time.time()
    transform_ffd_mi = bspline_intra_modal_registration(fixed_image, moving_image, 'mi')
    t_ffd_mi = time.time() - t
    
    t = time.time()
    transform_ffd_corr = bspline_intra_modal_registration(fixed_image, moving_image, 'corr')
    t_ffd_corr = time.time() - t

    t = time.time()
    transform_dvf_mse = dvf_registration(fixed_image, moving_image, 'mse')
    t_dvf_mse = time.time() - t

    t = time.time()
    transform_dvf_mi = dvf_registration(fixed_image, moving_image, 'mi')
    t_dvf_mi = time.time() - t

    t = time.time()
    transform_dvf_demon = dvf_registration(fixed_image, moving_image, 'demon')
    t_dvf_demon = time.time() - t
    
    t = time.time()
    transform_dvf_corr = dvf_registration(fixed_image, moving_image, 'corr')
    t_dvf_corr = time.time() - t
    
    t = time.time()
    transform_farneback = cv2_farneback(sitk.GetArrayFromImage(fixed_image), sitk.GetArrayFromImage(moving_image))
    t_farneback = time.time() - t

    #t = time.time()
    #transform_elastix = elastix_transform(fixed_image, moving_image)
    #t_elastix = time.time() - t

    t = time.time()
    transform_dipy_em = dipy_transform(fixed_image, moving_image, 'em')
    t_dipy_em = time.time() - t
    
    t = time.time()
    transform_dipy_mse = dipy_transform(fixed_image, moving_image, 'mse')
    t_dipy_mse = time.time() - t
    
    t = time.time()
    transform_dipy_corr = dipy_transform(fixed_image, moving_image, 'corr')
    t_dipy_corr = time.time() - t    
    

    interpolator = "cubic"
    # Create interpolator
    if (interpolator == "near"):
        interp = sitk.sitkNearestNeighbor
    elif (interpolator == "linear"):
        interp = sitk.sitkLinear
    elif (interpolator == "cubic"):
        interp = sitk.sitkBSpline
    else:
        raise Exception("No interpolator specified") # should never happen
    resampled_ffd_mse = sitk.Resample(moving_image, transform_ffd_mse, interp, 0.0, moving_image.GetPixelID())
    resampled_ffd_mi = sitk.Resample(moving_image, transform_ffd_mi, interp, 0.0, moving_image.GetPixelID())
    resampled_ffd_corr = sitk.Resample(moving_image, transform_ffd_corr, interp, 0.0, moving_image.GetPixelID())
    resampled_dvf_mse = sitk.Resample(moving_image, transform_dvf_mse, interp, 0.0, moving_image.GetPixelID())
    resampled_dvf_mi = sitk.Resample(moving_image, transform_dvf_mi, interp, 0.0, moving_image.GetPixelID())
    resampled_dvf_demon = sitk.Resample(moving_image, transform_dvf_demon, interp, 0.0, moving_image.GetPixelID())
    resampled_dvf_corr = sitk.Resample(moving_image, transform_dvf_corr, interp, 0.0, moving_image.GetPixelID())
    resampled_farneback = sitk.Resample(moving_image, transform_farneback, interp, 0.0, moving_image.GetPixelID())
    #resampled_elastix = sitk.Resample(moving_image, transform_elastix, interp, 0.0, moving_image.GetPixelID())
    resampled_dipy_em = sitk.Resample(moving_image, transform_dipy_em, interp, 0.0, moving_image.GetPixelID())
    resampled_dipy_mse = sitk.Resample(moving_image, transform_dipy_mse, interp, 0.0, moving_image.GetPixelID())
    resampled_dipy_corr = sitk.Resample(moving_image, transform_dipy_corr, interp, 0.0, moving_image.GetPixelID())
    
    score_ffd_mse, mse_ffd_mse = get_score(fixed_image, resampled_ffd_mse)
    score_ffd_mi, mse_ffd_mi = get_score(fixed_image, resampled_ffd_mi)
    score_ffd_corr, mse_ffd_corr = get_score(fixed_image, resampled_ffd_corr)
    score_dvf_mse, mse_dvf_mse = get_score(fixed_image, resampled_dvf_mse)
    score_dvf_mi, mse_dvf_mi = get_score(fixed_image, resampled_dvf_mi)
    score_dvf_demon, mse_dvf_demon = get_score(fixed_image, resampled_dvf_demon)
    score_dvf_corr, mse_dvf_corr = get_score(fixed_image, resampled_dvf_corr)
    score_farneback, mse_farneback = get_score(fixed_image, resampled_farneback)
    #score_elastix, mse_elastix = get_score(fixed_image, resampled_elastix)
    score_dipy_em, mse_dipy_em = get_score(fixed_image, resampled_dipy_em)
    score_dipy_mse, mse_dipy_mse = get_score(fixed_image, resampled_dipy_mse)
    score_dipy_corr, mse_dipy_corr = get_score(fixed_image, resampled_dipy_corr)
    
    from tabulate import tabulate
    result = [['B-SPLINE MSE', score_ffd_mse, mse_ffd_mse, t_ffd_mse],
              ['B-SPLINE MI', score_ffd_mi, mse_ffd_mi, t_ffd_mi],
              ['B-SPLINE CORR', score_ffd_corr, mse_ffd_corr, t_ffd_corr],
              ['DVF MSE', score_dvf_mse, mse_dvf_mse, t_dvf_mse],
              ['DVF MI', score_dvf_mi, mse_dvf_mi, t_dvf_mi],
              ['DVF DEMON', score_dvf_demon, mse_dvf_demon, t_dvf_demon],
              ['DVF CORR', score_dvf_corr, mse_dvf_corr, t_dvf_corr],
              ['FARNEBACK', score_farneback, mse_farneback, t_farneback],
              #['ELASTIX', score_elastix, mse_elastix, t_elastix],
              ['DIPY EM', score_dipy_em, mse_dipy_em, t_dipy_em],
              ['DIPY MSE', score_dipy_mse, mse_dipy_mse, t_dipy_mse],
              ['DIPY CORR', score_dipy_corr, mse_dipy_corr, t_dipy_corr]]
    print(tabulate(result, headers=['Method', 'SSIM', 'MSE', 'time(s)']))