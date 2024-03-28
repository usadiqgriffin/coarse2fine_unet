import numpy as np
import sys
# import network_stuff as ns
import os
from deploy_helper import get_seed_points_from_centerlinemap128
from ss.visualization import imshow
import utils.stn3d as stn
from medpy.metric.binary import hd, dc, jc
import timeit
from scipy.spatial import distance_matrix


# ground_truth, and predicted have ordered points
def compare_lines(predicted, ground_truth):
    eps = 0.0000001
    g_points = ground_truth
    p_points = predicted

    p_len = len(p_points)
    g_len = len(g_points)

    lengths = np.zeros(p_len)

    for p_i, p_coor in enumerate(p_points):
        dist_min = np.inf
        for g_j, g_coor in enumerate(g_points):
            if g_j == g_len-1:
                d = -1
            else:
                d = 1
            x = p_coor - g_coor
            y_f = g_points[g_j+d] - g_coor

            x_l = np.linalg.norm(x)
            y_f_l = np.linalg.norm(y_f)

            p_x_y_f = np.dot(x, y_f) / (y_f_l + eps)

            p_f_p = y_f*p_x_y_f/(y_f_l + eps)

            p_f_l = np.linalg.norm(p_f_p - y_f)

            if y_f_l > p_x_y_f + eps and y_f_l > p_f_l + eps:
                closest_point = p_f_p + g_coor
            else:
                closest_point = g_coor

            pot_dist_min = np.linalg.norm(closest_point - p_coor)
            if dist_min > pot_dist_min:
                dist_min = pot_dist_min

        lengths[p_i] = dist_min
    return lengths

def get_ordered_cl_from_tree(mTreeIter):
    #the output is a Nx3 array, and the dimension of the point
    #is arranged in (z,y,x) fashion
    vsPoint_list = []
    nodes = mTreeIter.getNodes()
    endNode = None
    num_of_nodes = len(nodes)
    print('NUM OF NODES: ',len(nodes))
    if num_of_nodes == 1:
        return np.array([nodes[0].x]), num_of_nodes
    elif num_of_nodes == 0:
        return None, num_of_nodes

    for node in nodes:
        if node.nodeType == 'end' and node.prevNode != None:
            endNode = node
    while(endNode != None and endNode.prevNode != None and not endNode.has_been_seen):
        if endNode.has_been_seen:
            print('Stopping do to cycle!')
            #input()
        vsPoint_list += [endNode.x]
        endNode.has_been_seen = True
        endNode = endNode.prevNode

    return np.array(vsPoint_list), num_of_nodes

def length_of_curve(points):
    le = 0.0
    for idx, p in enumerate(points[:-1]):
        le += np.linalg.norm(points[idx]-p)
    return le

def centerline_avg_min_length(study_info):#(gt, pred):
    print('Starting metric!')
    output = {}
    gt_dict = study_info.cl_pts
    preds_per_mult = study_info.output
    #print('output:',preds_per_mult)

    for num_sp in preds_per_mult:
        preds = preds_per_mult[num_sp]
        num_of_cor = 0
        for cor_name in preds:
            avg_gt = None
            max_gt = None

            avg_both = None
            max_both = None

            lengths = None
            lengths_both = None

            gt = gt_dict[cor_name]
            pr = preds[cor_name]
            #print(gt)
            #print(pr)
            #input('waiting')

            if pr is not None:

                pr_l = length_of_curve(pr)
                gt_l = length_of_curve(gt)

                if pr_l < gt_l:
                    longest = gt
                    s_l = pr_l
                else:
                    longest = pr
                    s_l = gt_l

                le = 0.0
                for idx, p in enumerate(longest[:-1]):
                    le += np.linalg.norm(longest[idx]-p)
                    if le > s_l:
                        break

                longest = longest[:idx]

                if pr_l < gt_l:
                    gt = longest
                else:
                    pr = longest

                lengths = compare_lines(pr, gt)
                # lengths_both = np.concatenate([lengths, compare_lines(gt,pr)], axis=0)

                avg_gt = np.mean(lengths)
                # max_gt = np.amax(lengths)

                # avg_both = np.mean(lengths_both)
                # max_both = np.amax(lengths_both)

            num_of_cor += 1 if avg_gt is not None and avg_gt < 50 else 0

        in_str = 'num_branches_under_50'
        output[in_str] = [num_of_cor] if in_str not in output else output[in_str] + [num_of_cor]

    return output

def centerline_avg_hdd(study_info):#(gt, pred):
    print('Starting metric!')
    output = {}
    gt_dict = study_info.cl_pts
    preds_per_mult = study_info.output
    #print('output:',preds_per_mult)

    for num_sp in preds_per_mult:
        preds = preds_per_mult[num_sp]
        for cor_name in preds:
            avg_gt = None
            max_gt = None

            avg_both = None
            max_both = None

            lengths = None
            lengths_both = None

            gt = gt_dict[cor_name]
            pr = preds[cor_name]
            #print(gt)
            #print(pr)
            #input('waiting')

            if pr is not None:

                lengths = compare_lines(pr, gt)
                lengths_both = np.concatenate([lengths, compare_lines(gt,pr)], axis=0)

                avg_gt = np.mean(lengths)
                max_gt = np.amax(lengths)

                avg_both = np.mean(lengths_both)
                max_both = np.amax(lengths_both)

            cor_name = cor_name.upper()
            in_str = 'mean_distance_to_gt_'+cor_name+' (pixels)'
            output[in_str] = [avg_gt] if in_str not in output else output[in_str] + [avg_gt]
            in_str = 'max_distance_to_gt_'+cor_name+' (pixels)'
            output[in_str] = [max_gt] if in_str not in output else output[in_str] + [max_gt]
            in_str = 'mean_distance_'+cor_name+' (pixels)'
            output[in_str] = [avg_both] if in_str not in output else output[in_str] + [avg_both]
            in_str = 'Hausdorff_distance_'+cor_name+' (pixels)'
            output[in_str] = [max_both] if in_str not in output else output[in_str] + [max_both]
        in_str = 'num of seed points'
        output[in_str] = [num_sp] if in_str not in output else output[in_str] + [num_sp]

    return output
    #return {'centerline_avg_gt': avg_gt, 'centerline_hdd_gt': hdd_gt,'centerline_avg_both': avg_both, 'centerline_hdd_both': hdd_both}

def centerline_avg_hdd_noise(study_info):#(gt, pred):
    print('Starting metric!')
    output = {}
    gt_dict = study_info.cl_pts
    preds_per_mult = study_info.output

    for noise_level in preds_per_mult:
        preds = preds_per_mult[noise_level]
        for cor_name in preds:
            avg_gt = None
            max_gt = None

            avg_both = None
            max_both = None

            lengths = None
            lengths_both = None

            gt = gt_dict[cor_name]
            pr = preds[cor_name]

            if pr is not None:
                lengths = compare_lines(pr, gt)
                lengths_both = np.concatenate([lengths, compare_lines(gt,pr)], axis=0)

                avg_gt = np.mean(lengths)
                max_gt = np.amax(lengths)

                avg_both = np.mean(lengths_both)
                max_both = np.amax(lengths_both)

            cor_name = cor_name.upper()
            in_str = 'mean_distance_to_gt_'+cor_name+' (pixels)'
            output[in_str] = [avg_gt] if in_str not in output else output[in_str] + [avg_gt]
            in_str = 'max_distance_to_gt_'+cor_name+' (pixels)'
            output[in_str] = [max_gt] if in_str not in output else output[in_str] + [max_gt]
            in_str = 'mean_distance_'+cor_name+' (pixels)'
            output[in_str] = [avg_both] if in_str not in output else output[in_str] + [avg_both]
            in_str = 'Hausdorff_distance_'+cor_name+' (pixels)'
            output[in_str] = [max_both] if in_str not in output else output[in_str] + [max_both]
        in_str = 'noise level'
        output[in_str] = [noise_level] if in_str not in output else output[in_str] + [noise_level]




    return output
    #return {'centerline_avg_gt': avg_gt, 'centerline_hdd_gt': hdd_gt,'centerline_avg_both': avg_both, 'centerline_hdd_both': hdd_both}

def compute_dice(study_info):
    seg = np.squeeze(study_info.seg_gt)
    # seg = seg[::2, ::2, ::2]
    prd = np.squeeze(study_info.compute_dict["Postprocess/seg_pred_final:0"][0])
    labels = study_info.seg_labels
    dsc = {}
    if seg is None:# or seg.shape is not prd.shape:
        for key, val in labels.items():
            dsc[val] = 'None'
        return dsc

    prd = prd.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    prd = prd[:seg.shape[0], :, :]
    seg = np.array(seg)
    prd = np.array(prd)
    for key, val in labels.items():

        curr_seg = seg == key
        curr_prd = prd == key
        den = np.sum(curr_seg) + np.sum(curr_prd)
        if den == 0:
            dsc[val] = 1
        else:
            num = 2 * np.bitwise_and(curr_seg, curr_prd).sum()
            dsc[val] = num/den
    return dsc


def compute_hausdorff_distance(study_info):  # seg, prd, voxel_size, connectivity, labels):
    seg = study_info.seg_gt
    prd = study_info.compute_dict["Postprocess/seg_pred_final:0"][0]
    labels = study_info.seg_hd_labels
    voxel_size = study_info.input_spacing
    cFlag = 0

    d = {}
    if seg is None:
        for key, val in labels.items():
            d[val] = 'None'
        return d

    prd = prd.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    prd = prd[:seg.shape[0], :, :]
    seg = np.array(seg)
    prd = np.array(prd)

    for key, val in labels.items():
        curr_seg = seg == key
        curr_prd = prd == key
        to_dataframe = {}
        # start = timeit.default_timer()
        try:
            d[val] = hd(curr_seg[::4,::4,::4], curr_prd[::4,::4,::4], voxelspacing=voxel_size*4, connectivity=1)
        except:
            d[val] = np.nan
        # stop = timeit.default_timer()
        # print('hd runtime: {:.3f}s'.format(stop - start))
    return d


def get_seed_points(vessel_probs128, img_shape, spacing, out_dims=np.array([128, 128, 128]), out_res=np.array([1.5, 1.5, 1.5])):
    trans_matrix = stn.get_transform(img_shape, spacing, out_dims, out_res)
    seed_points = get_seed_points_from_centerlinemap128(vessel_probs128, trans_matrix, img_shape)
    return seed_points.astype(np.uint16)


def seed_point_min_avg_and_hdd(predicted, ground, lengths=None):
    # for p_i, p_coor in enumerate(predicted):
    #     for g_j, g_coor in enumerate(ground):
    #         lengths[p_i, g_j] = np.linalg.norm(p_coor-g_coor)
    lengths_pr = distance_matrix(predicted, ground)
    lengths_gt = distance_matrix(ground, predicted)

    if lengths is None:
        p_shape = predicted.shape
        g_shape = ground.shape

        p_len = p_shape[0]
        g_len = g_shape[0]

        lengths = np.zeros((p_len, g_len))

    min_vals_gt = []
    min_vals_pr = []
    for l in lengths_gt:
        min_vals_gt += [np.amin(l)]
    for l in lengths_pr:
        min_vals_pr += [np.amin(l)]

    min_avg_gt = np.mean(min_vals_gt)
    hdd_gt = np.amax(min_vals_gt)

    min_avg_both = np.mean(min_vals_pr + min_vals_gt)
    hdd_both = np.amax(min_vals_pr + min_vals_gt)

    return min_avg_gt, hdd_gt, min_avg_both, hdd_both


# predicted : predicted seed points: has shape (3,N)
# ground : ground truth seed points: has shape (3,M)
def seed_point_metric(study_info):
    predicted = get_seed_points(study_info.compute_dict["Postprocess/centerlinemap:0"][0],
                                study_info.input_x.shape,
                                study_info.input_spacing)
    if study_info.gt_radii is None:
        return {'seed_points_avg': 'None', 'seed_points_hdd': 'None'}
    ground = np.array(np.where(study_info.gt_radii > 0))  # ns.get_seed_points(radii,img.shape,spacing)

    predicted = np.transpose(predicted, [1, 0])
    ground = np.transpose(ground, [1, 0])

    p_shape = predicted.shape
    g_shape = ground.shape

    p_len = p_shape[0]
    g_len = g_shape[0]

    lengths = np.zeros((p_len, g_len))

    min_avg_gt, hdd_gt, min_avg_both, hdd_both  = seed_point_min_avg_and_hdd(predicted, ground, lengths)

    return {'seed_points_avg_gt': min_avg_gt, 'seed_points_hdd_gt': hdd_gt,'seed_points_avg_both': min_avg_both, 'seed_points_hdd_both': hdd_both}


# 'Prediction/centerlinemap_rca',: in 128**3 space
# 'Prediction/centerlinemap_lm', in 128**3 space
# 'Prediction/centerlinemap_lad', in 128**3 space
# 'Prediction/centerlinemap_lcx', in 128**3 space
def seed_point_metric_each_branch(study_info):
    predicted_rca = get_seed_points(study_info.compute_dict["Prediction/centerlinemap_rca:0"][0],
                                    study_info.input_x.shape,
                                    study_info.input_spacing)
    predicted_lm = get_seed_points(study_info.compute_dict["Prediction/centerlinemap_lm:0"][0],
                                   study_info.input_x.shape,
                                   study_info.input_spacing)
    predicted_lad = get_seed_points(study_info.compute_dict["Prediction/centerlinemap_lad:0"][0],
                                    study_info.input_x.shape,
                                    study_info.input_spacing)
    predicted_lcx = get_seed_points(study_info.compute_dict["Prediction/centerlinemap_lcx:0"][0],
                                    study_info.input_x.shape,
                                    study_info.input_spacing)

    if study_info.gt_radii is None:
        return {'seed_points_avg_rca': None,
                'seed_points_avg_lm': None,
                'seed_points_avg_lad': None,
                'seed_points_avg_lcx': None,
                'seed_points_hdd_rca': None,
                'seed_points_hdd_lm': None,
                'seed_points_hdd_lad': None,
                'seed_points_hdd_lcx': None
                }

    ground_rca = np.array(np.where(study_info.gt_radii == 1))
    ground_lm = np.array(np.where(study_info.gt_radii == 2))
    ground_lad = np.array(np.where(study_info.gt_radii == 3))
    ground_lcx = np.array(np.where(study_info.gt_radii == 4))

    predicted_rca = np.transpose(predicted_rca, [1, 0])
    predicted_lm = np.transpose(predicted_lm, [1, 0])
    predicted_lad = np.transpose(predicted_lad, [1, 0])
    predicted_lcx = np.transpose(predicted_lcx, [1, 0])

    ground_rca = np.transpose(ground_rca, [1, 0])
    ground_lm = np.transpose(ground_lm, [1, 0])
    ground_lad = np.transpose(ground_lad, [1, 0])
    ground_lcx = np.transpose(ground_lcx, [1, 0])

    # if predicted_rca.shape[0] != 0:
    #     lengths_rca = np.zeros((predicted_rca.shape[0], ground_rca.shape[0]))
    #     min_avg_rca, hdd_rca = seed_point_min_avg_and_hdd(predicted_rca, ground_rca, lengths_rca)
    # else:
    #     min_avg_rca = None
    #     hdd_rca = None
    # if predicted_lm.shape[0] != 0:
    #     lengths_lm = np.zeros((predicted_lm.shape[0], ground_lm.shape[0]))
    #     min_avg_lm, hdd_lm = seed_point_min_avg_and_hdd(predicted_lm, ground_lm, lengths_lm)
    #     print(predicted_lm.shape[0])
    #     input('Enter')
    # else:
    #     min_avg_lm = None
    #     hdd_lm = None
    # if predicted_lad.shape[0] != 0:
    #     lengths_lad = np.zeros((predicted_lad.shape[0], ground_lad.shape[0]))
    #     min_avg_lad, hdd_lad = seed_point_min_avg_and_hdd(predicted_lad, ground_lad, lengths_lad)
    # else:
    #     min_avg_lad = None
    #     hdd_lad = None
    # if predicted_lcx.shape[0] != 0:
    #     lengths_lcx = np.zeros((predicted_lcx.shape[0], ground_lcx.shape[0]))
    #     min_avg_lcx, hdd_lcx = seed_point_min_avg_and_hdd(predicted_lcx, ground_lcx, lengths_lcx)
    # else:
    #     min_avg_lcx = None
    #     hdd_lcx = None

    # return {'seed_points_avg_rca': min_avg_rca,
    #         'seed_points_avg_lm': min_avg_lm,
    #         'seed_points_avg_lad': min_avg_lad,
    #         'seed_points_avg_lcx': min_avg_lcx,
    #         'seed_points_hdd_rca': hdd_rca,
    #         'seed_points_hdd_lm': hdd_lm,
    #         'seed_points_hdd_lad': hdd_lad,
    #         'seed_points_hdd_lcx': hdd_lcx
    #         }

    if predicted_rca.shape[0] == 0:
        predicted_rca = np.zeros((ground_rca.shape))
    if predicted_lm.shape[0] == 0:
        predicted_lm = np.zeros((ground_lm.shape))
    if predicted_lad.shape[0] == 0:
        predicted_lad = np.zeros((ground_lad.shape))
    if predicted_lcx.shape[0] == 0:
        predicted_lcx = np.zeros((ground_lcx.shape))

    lengths_rca = np.zeros((predicted_rca.shape[0], ground_rca.shape[0]))
    lengths_lm = np.zeros((predicted_lm.shape[0], ground_lm.shape[0]))
    lengths_lad = np.zeros((predicted_lad.shape[0], ground_lad.shape[0]))
    lengths_lcx = np.zeros((predicted_lcx.shape[0], ground_lcx.shape[0]))

    min_avg_rca, hdd_rca = seed_point_min_avg_and_hdd(predicted_rca, ground_rca, lengths_rca)
    min_avg_lm, hdd_lm = seed_point_min_avg_and_hdd(predicted_lm, ground_lm, lengths_lm)
    min_avg_lad, hdd_lad = seed_point_min_avg_and_hdd(predicted_lad, ground_lad, lengths_lad)
    min_avg_lcx, hdd_lcx = seed_point_min_avg_and_hdd(predicted_lcx, ground_lcx, lengths_lcx)

    return {'seed_points_avg_rca': min_avg_rca,
            'seed_points_avg_lm': min_avg_lm,
            'seed_points_avg_lad': min_avg_lad,
            'seed_points_avg_lcx': min_avg_lcx,
            'seed_points_hdd_rca': hdd_rca,
            'seed_points_hdd_lm': hdd_lm,
            'seed_points_hdd_lad': hdd_lad,
            'seed_points_hdd_lcx': hdd_lcx
            }


def unannotated_metrics(study_info):
    print('Starting unannotated metric!')

    cor_preds_dict = study_info.output
    cor_preds_lengths_dict = study_info.output_lengths

    output = {}
    for num_sp in cor_preds_dict:
        cor_preds = cor_preds_dict[num_sp]
        cor_preds_lengths = cor_preds_lengths_dict[num_sp]

        num_branches = 0
        for cor_name in cor_preds:
            in_str = cor_name.upper() + '_length (mm)'
            branch_length = 0
            if cor_preds[cor_name] is not None:
                num_branches += 1
                branch_length = cor_preds_lengths[cor_name]

            output[in_str] = [float(branch_length)] if in_str not in output else output[in_str] + [float(branch_length)]
        in_str = 'num_branches'
        output[in_str] = [num_branches] if in_str not in output else output[in_str] + [num_branches]

        in_str = 'num of seed points'
        output[in_str] = [num_sp*16] if in_str not in output else output[in_str] + [num_sp*16]
    return output


def unannotated_metrics_noise(study_info):
    print('Starting unannotated metric!')

    cor_preds_dict = study_info.output
    cor_preds_lengths_dict = study_info.output_lengths

    output = {}
    for noise_level in cor_preds_dict:
        cor_preds = cor_preds_dict[noise_level]
        cor_preds_lengths = cor_preds_lengths_dict[noise_level]

        num_branches = 0
        for cor_name in cor_preds:
            in_str = cor_name.upper() + '_length (mm)'
            branch_length = 0
            if cor_preds[cor_name] is not None:
                num_branches += 1
                branch_length = cor_preds_lengths[cor_name]

            output[in_str] = [float(branch_length)] if in_str not in output else output[in_str] + [float(branch_length)]
        in_str = 'num_branches'
        output[in_str] = [num_branches] if in_str not in output else output[in_str] + [num_branches]

        in_str = 'noise level'
        output[in_str] = [noise_level] if in_str not in output else output[in_str] + [noise_level]
    return output


if __name__ == '__main__':
    in_points = np.zeros((20, 3))
    in_points[:, 2] = np.array(range(20))

    in_points2 = np.zeros((10, 3))
    in_points2[:, 2] = np.array(range(10))

    s = np.array([[0, 0, 0]])

    print(centerline_avg_hdd(s, in_points2))
    exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    graph = sys.argv[1]
    data_set = np.load(sys.argv[2])

    for idx, path in enumerate(data_set):
        img, radii, dsvec, spacing, annot_dict = ns.get_data_numpy(data_set[0])
        break
    feed_dict = ns.get_feed_dict(img, spacing)
    compute_dict = ns.get_compute_dict()

    output = ns.run_network(feed_dict, compute_dict, graph)
    pred_seed = ns.get_seed_points(output[1][0], img.shape, spacing)
    gt_seed = np.array(np.where(radii > 0))  # ns.get_seed_points(radii,img.shape,spacing)
    imshow(np.array(radii*255, dtype=np.uint16))
    imshow(np.array(output[1][0]*255, dtype=np.uint16))
    print(seed_point_metric(pred_seed, gt_seed))
