import sys
import argparse
import glob
import importlib
import pandas as pd
import numpy as np

import utils.image as im_util


HEADER = ["Study",
          "Total Dice", "Total Dice LV", "Total Dice RV", "Total Dice MYO",
          "Basal Dice", "Basal Dice LV", "Basal Dice RV", "Basal Dice MYO",
          "Mid Dice", "Mid Dice LV", "Mid Dice RV", "Mid Dice MYO",
          "Epical Dice", "Epical Dice LV", "Epical Dice RV", "Epical Dice MYO",
          "SegVol LV", "SegVol RV", "SegVol MYO",
          "PredVol LV", "PredVol RV", "PredVol MYO",]
BOXWIDTH = 192
BOXHEIGHT = 192
NUMLABELS = 4
PIXELWIDTH = 1.855 #mm
PIXELHEIGTH = 1.855 #mm
PIXELAREA = PIXELWIDTH * PIXELHEIGTH

if len(sys.argv) != 3:
    raise ValueError('Not enough arguments: [EXP_ID PHASE("ED" or "ES")]')

DATASET = '/mnt/SSD_1/biobank_3D_numpy/'
STUDIES = np.load(DATASET + 'val.npy')
my_module = importlib.import_module('experiments.' + sys.argv[1])
my_class = getattr(my_module, sys.argv[1])
my_experiment = my_class(exp_id = sys.argv[1], restore = True)
my_model = my_experiment.model


def ForwardPassNumpy(image):
    '''
    Applies resizing, scaling, and histogram equalization on the Numpy file
     and then crops the image. Feeds the cropped image to the netwok and
     outputs the predicted mask.
    input: numpy image path
    output: [0,:,:,0] which is filled by the preprocessed cropped image
    '''
    #image = np.load(npyPath)
    #### Scaling and Histogram Equalization ####
    min_val = np.min(image)
    max_val = np.max(image)
    image = np.round(np.multiply(np.divide(np.subtract(image, min_val), (max_val - min_val)), 255))
    image_view = np.array(image)
    image = im_util.his_equal3d2d(image)
    # Cropping a (self.BOXHEIGHT X self.BOXWIDTH) square at the center of image and seg
    cropped_image = im_util.Crop3D(image, BOXHEIGHT, BOXWIDTH)

    batch = np.empty([image.shape[0], BOXHEIGHT, BOXWIDTH, 1])
    batch[:,:,:,0] = cropped_image
    preds = my_model.session.run(my_model.prediction, feed_dict={my_model.x: batch, my_model.phase: 0})

    return preds

def MulticlassDice(img_gt, img_pred):
    """
    Function to compute the dice between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    Return
    ------
    A list of metrics in this order, [Total Dice, Dice LV, Dice RV, Dice Myo]
    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    res = []
    tot_intersect = 0
    tot_size = 0
    # Loop on each classes of the input images
    for c in [1, 3, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0
        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0
        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        intersect = np.count_nonzero(pred_c_i * gt_c_i)

        size_i1 = np.count_nonzero(pred_c_i)
        size_i2 = np.count_nonzero(gt_c_i)
        size = size_i1 + size_i2

        tot_intersect += intersect
        tot_size += size

        try:
            dice = 2. * intersect / float(size)
        except ZeroDivisionError:
            dice = None

        res += [dice]

    total_dice = 2. * tot_intersect / float(tot_size)
    res = [total_dice] + res
    res = ["{:.4f}".format(r) if (r is not None) else r for r in res]

    return res

def CalculateVol(img, pixel_area, slice_thickness):
    """
    Function to compute the volume of a given 3D input.

    Parameters
    ----------
    img: np.array
    Array of the segmentation map.

    pixel_area: float
    The area of each pixel used to compute the volumes.

    slice_thickness: float
    The distance between each slice used to compute the volumes.

    Return
    ------
    A list of volumes in this order, [Volume LV, Volume RV, Volume MYO]
    """
    res = []
    # Loop on each classes of the input images
    for c in [1, 3, 2]:
        # Copy the gt image to not alterate the input
        img_c_i = np.copy(img)
        img_c_i[img_c_i != c] = 0

        # Clip the value to compute the volumes
        img_c_i = np.clip(img_c_i, 0, 1)

        # Compute volume
        vol_img = img_c_i.sum() * pixel_area * slice_thickness / 1000. #mL

        res += [vol_img]

    return res

def main(phase):
    print(len(STUDIES), 'validation studies')
    result = []
    for study in STUDIES:
        study_id = study.split('/')[-2]
        print("Study No.", study_id)

        if phase == 'ED':
            try:
                Img = np.load(study + 'image3d_1.npy')
                Seg = np.load(study + 'seg3d_1.npy')
                Seg = im_util.Crop3D(Seg, BOXHEIGHT, BOXWIDTH)
            except:
                continue
        elif phase == 'ES':
            try:
                Img_list = glob.glob(study + 'image3d_??.npy')
                Seg_list = glob.glob(study + 'seg3d_??.npy')
                Img_list.sort()
                Seg_list.sort()
                Img = np.load(Img_list[0])
                Seg = np.load(Seg_list[0])
                Seg = im_util.Crop3D(Seg, BOXHEIGHT, BOXWIDTH)
            except:
                continue
        else:
            print("ERROR: The phase should be either 'ED' or 'ES'")

        Pred = ForwardPassNumpy(Img)

        # Calculate Dices
        TotalDice = MulticlassDice(Seg, Pred)
        BasalDice = MulticlassDice(Seg[0], Pred[0])
        MidDice = MulticlassDice(Seg[1:-1], Pred[1:-1])
        EpicalDice = MulticlassDice(Seg[-1], Pred[-1])

        # Calculate volumes
        PredVol = CalculateVol(Pred, PIXELAREA, 10)
        SegVol = CalculateVol(Seg, PIXELAREA, 10)
        result.append([study_id] +
                      TotalDice +
                      BasalDice +
                      MidDice +
                      EpicalDice +
                      SegVol +
                      PredVol)

    result = [[float(y) if (y is not None) else np.nan for y in x] for x in result]
    df = pd.DataFrame(result, columns=HEADER)
    df.set_index('Study', inplace=True)
    df.loc['mean'] = df.mean()
    df.to_csv("{}_{}.csv".format(args.EXP_ID, phase), index=True)
    print("Finished evaluation \n"
          "Results saved on {}_{}.csv".format(args.EXP_ID, phase))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute Dice for different regions of biobank.")
    parser.add_argument("EXP_ID", type=str, help="Experiment to load weights from")
    parser.add_argument("PHASE", type=str, help="Phase ('ED' or 'ES')")
    args = parser.parse_args()
    main(args.PHASE)
