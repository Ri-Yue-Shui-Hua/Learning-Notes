# 图像变换



## 图像变换



参考： [Python SimpleITK.ResampleImageFilter方法代码示例 - 纯净天空 (vimsky.com)](https://vimsky.com/examples/detail/python-method-SimpleITK.ResampleImageFilter.html)



### 重采样 resample_images



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def resample_images(
            image: sitk.Image,
            transforms: List[sitk.Euler3DTransform],
            interpolation: Interpolation,
            ) -> List[sitk.Image]:
        floating = reference = image
        default_value = np.float64(sitk.GetArrayViewFromImage(image).min())
        transforms = transforms[1:]  # first is identity
        images = [image]  # first is identity
        for transform in transforms:
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(get_sitk_interpolator(interpolation))
            resampler.SetReferenceImage(reference)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetDefaultPixelValue(default_value)
            resampler.SetTransform(transform)
            resampled = resampler.Execute(floating)
            images.append(resampled)
        return images 
```



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 
```



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 
```



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def produceRandomlyDeformedImage(image, label, numcontrolpoints, stdDef):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl>0.5).astype(dtype=np.float32)

    return outimg,outlbl 
```



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def apply_bspline_transform(
            self,
            tensor: torch.Tensor,
            affine: np.ndarray,
            bspline_params: np.ndarray,
            interpolation: Interpolation,
            ) -> torch.Tensor:
        assert tensor.dim() == 4
        assert len(tensor) == 1
        image = self.nib_to_sitk(tensor[0], affine)
        floating = reference = image
        bspline_transform = self.get_bspline_transform(
            image,
            self.num_control_points,
            bspline_params,
        )
        self.parse_free_form_transform(
            bspline_transform, self.max_displacement)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetTransform(bspline_transform)
        resampler.SetInterpolator(get_sitk_interpolator(interpolation))
        resampler.SetDefaultPixelValue(tensor.min().item())
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampled = resampler.Execute(floating)

        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()  # ITK to NumPy
        tensor[0] = torch.from_numpy(np_array)
        return tensor 
```





### 随机平移 produceRandomlyTranslatedImage



```python
# 需要导入模块: import SimpleITK [as 别名]
# 或者: from SimpleITK import ResampleImageFilter [as 别名]
def produceRandomlyTranslatedImage(image, label):
    sitkImage = sitk.GetImageFromArray(image, isVector=False)
    sitklabel = sitk.GetImageFromArray(label, isVector=False)

    itemindex = np.where(label > 0)
    randTrans = (0,np.random.randint(-np.min(itemindex[1])/2,(image.shape[1]-np.max(itemindex[1]))/2),np.random.randint(-np.min(itemindex[0])/2,(image.shape[0]-np.max(itemindex[0]))/2))
    translation = sitk.TranslationTransform(3, randTrans)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(translation)

    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=float)

    outlbl = sitk.GetArrayFromImage(outlabsitk) > 0
    outlbl = outlbl.astype(dtype=float)

    return outimg, outlbl 
```



### 旋转

参考：[SimpleITK Euler3D transform, problem with output size/resampling - Beginner Questions - ITK](https://discourse.itk.org/t/simpleitk-euler3d-transform-problem-with-output-size-resampling/4387)



```python
# -*- coding : UTF-8 -*-
# @file   : rotate2_img.py
# @Time   : 2022-08-16 16:27
# @Author : wmz

import SimpleITK as sitk
import itertools
import numpy as np
from glob import glob
import os



def rotation3d(image, theta_x, theta_y, theta_z, output_spacing=None, background_value=0.0):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively and resamples it to be isotropic.
    :param image: An sitk 3D image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param output_spacing: Scalar denoting the isotropic output image spacing. If None, then use the smallest
                           spacing from original image.
    :return: The rotated image
    """
    euler_transform = sitk.Euler3DTransform(
        image.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2.0 for sz in image.GetSize()]),
        np.deg2rad(theta_x),
        np.deg2rad(theta_y),
        np.deg2rad(theta_z))

    # compute the resampling grid for the transformed image
    max_indexes = [sz - 1 for sz in image.GetSize()]
    extreme_indexes = list(itertools.product(*(list(zip([0] * image.GetDimension(), max_indexes)))))
    extreme_points_transformed = [euler_transform.TransformPoint(image.TransformContinuousIndexToPhysicalPoint(p)) for p
                                  in extreme_indexes]

    output_min_coordinates = np.min(extreme_points_transformed, axis=0)
    output_max_coordinates = np.max(extreme_points_transformed, axis=0)

    # isotropic ouput spacing
    if output_spacing is None:
        output_spacing = min(image.GetSpacing())
    output_spacing = [output_spacing] * image.GetDimension()

    output_origin = output_min_coordinates
    output_size = [int(((omx - omn) / ospc) + 0.5) for ospc, omn, omx in
                   zip(output_spacing, output_min_coordinates, output_max_coordinates)]

    output_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    output_pixeltype = image.GetPixelIDValue()

    return sitk.Resample(image,
                         output_size,
                         euler_transform.GetInverse(),
                         sitk.sitkLinear,
                         output_origin,
                         output_spacing,
                         output_direction,
                         background_value,
                         output_pixeltype)


def write_sitk_img(sitk_img, file):
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    sitk.WriteImage(sitk_img, file)


if __name__ == '__main__':
    img_dir = r"D:\Dataset\dnFemurSegmentation\label"
    output_dir = r"D:\Dataset\dnFemurSegmentation\rotae_sitk_label"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = sorted(glob(os.path.join(img_dir, '*.nii.gz')))
    file = files[0]
    sitk_img = sitk.ReadImage(file)
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    angle_x, angle_y, angle_z = 40, 20, 30
    output_img = rotation3d(sitk_img, angle_x, angle_y, angle_z)
    dst_file = file.replace(img_dir, output_dir)
    write_sitk_img(output_img, dst_file)

```





### 图像镜像

参考：[How to just flip an image about z axes? - ITK](https://discourse.itk.org/t/how-to-just-flip-an-image-about-z-axes/3333)





### 空间变换

参考： [spatial_transformations (simpleitk.org)](https://simpleitk.org/SPIE2018_COURSE/spatial_transformations.pdf)





### 参考文档

[Pad an Image With a Constant — v5.3.0 (itk.org)](https://examples.itk.org/src/filtering/imagegrid/padanimagewithaconstant/documentation)