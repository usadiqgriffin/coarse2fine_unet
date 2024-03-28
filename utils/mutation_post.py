import numpy as np
import utils.image as im_util


# Adding a valid LV and/or RV contour to an empty image
def add(target_image3d, segs, headers):

    target_shape = target_image3d.shape
    # Finding a target_index for target_image3d
    empty_slices = []
    for i in range(target_shape[0]):
        if np.count_nonzero(target_image3d[i]) == 0:
            empty_slices.append(i)

    if len(empty_slices) == 0:
        return

    target_index = empty_slices[np.random.randint(len(empty_slices))]

    # Computing source_image2d

    # Picking the source image from the BioBank training set
    rand_index = np.random.randint(len(segs[0]))
    source_image3d = segs[0][rand_index]
    source_hdr = headers[0][rand_index]

    full_slices = []
    for i in range(source_image3d.shape[0]):
        if np.count_nonzero(source_image3d[i]) != 0:
            full_slices.append(i)

    rand_slice = full_slices[np.random.randint(len(full_slices))]
    source_image2d = np.array(source_image3d[rand_slice], np.float32)

    cropped_source_image2d = im_util.crop2d(source_image2d, target_shape[1], target_shape[2])

    # Mutating target_image3d[target_index]
    target_image3d[target_index] = cropped_source_image2d

    return target_image3d
