import torch
import torchvision
import object_detection_dataset as odd
import time
import copy
from model.roi_align.modules.roi_align import RoIAlignAvg
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#dummy_time = 0
class FasterRCNN(torch.nn.Module):
    def __init__(self, base, base_stride, base_out_channels, rpn_channels, rpn_kernel_size, anchor_sizes, roi_grid_size, roi_align_kernel_size, roi_align_stride, head):
        super(FasterRCNN, self).__init__()
        self.base = base
        self.base_stride = base_stride
        self.head = head
        self.anchor_sizes = anchor_sizes
        self.roi_grid_size = roi_grid_size
        self.roi_pool = torch.nn.MaxPool2d(roi_align_kernel_size, stride = roi_align_stride)
        self.roi_align_layer = RoIAlignAvg(roi_grid_size / 2, roi_grid_size / 2, 1.0 / base_stride)
        self.rpn_conv = torch.nn.Conv2d(base_out_channels, rpn_channels, rpn_kernel_size, padding=1)
        self.rpn_conv_cls = torch.nn.Conv2d(rpn_channels, 2 * len(anchor_sizes), 1)
        self.rpn_conv_tfm = torch.nn.Conv2d(rpn_channels, 4 * len(anchor_sizes), 1)

    def roi_align(self, x, y, input_features):
        input_features = input_features.unsqueeze(0)
        x_0 = torch.floor(x)
        x_1 = torch.ceil(x)
        y_0 = torch.floor(y)
        y_1 = torch.ceil(y)
        weights_00 = (y_1 - y).unsqueeze(1).mm((x_1 - x).unsqueeze(0))
        weights_01 = (y - y_0).unsqueeze(1).mm((x_1 - x).unsqueeze(0))
        weights_10 = (y_1 - y).unsqueeze(1).mm((x - x_0).unsqueeze(0))
        weights_11 = (y - y_0).unsqueeze(1).mm((x - x_0).unsqueeze(0))
        features_00 = input_features.index_select(2, y_0.long()).index_select(3, x_0.long())
        features_01 = input_features.index_select(2, y_1.long()).index_select(3, x_0.long())
        features_10 = input_features.index_select(2, y_0.long()).index_select(3, x_1.long())
        features_11 = input_features.index_select(2, y_1.long()).index_select(3, x_1.long())

        roi_feature_map = torch.zeros(input_features.size(0), input_features.size(1), y.size(0), x.size(0), device = input_features.device)
        for i in range(roi_feature_map.size(0)):
            roi_feature_map[i] = features_00[i] * weights_00.unsqueeze(0).expand(input_features.size(1), -1, -1) + features_01[i] * weights_01.unsqueeze(0).expand(input_features.size(1), -1, -1) + features_10[i] * weights_10.unsqueeze(0).expand(input_features.size(1), -1, -1) + features_11[i] * weights_11.unsqueeze(0).expand(input_features.size(1), -1, -1)
        return self.roi_pool(roi_feature_map)

    def rpn(self, input_features):
        rpn_features = self.rpn_conv(input_features)
        rpn_cls = self.rpn_conv_cls(rpn_features)
        rpn_bbox_tfm = self.rpn_conv_tfm(rpn_features)
        return rpn_cls, rpn_bbox_tfm

    def forward(self, input_features, obj_score_threshold = 0.5, obj_nms_threshold = 0.7, obj_rank = 2000):
        feature_map = self.base(input_features)
        rpn_cls, rpn_bbox_tfm = self.rpn(feature_map)
        final_cls, final_bbox_tfm, final_bbox, proposal_bbox = [], [], [], []
        pos_index = torch.arange(0, rpn_cls.size(1), step = 2, dtype = torch.long, device = input_features.device)
        neg_index = torch.arange(1, rpn_cls.size(1), step = 2, dtype = torch.long, device = input_features.device)
        obj_score = torch.exp(rpn_cls.index_select(1, pos_index)) / (torch.exp(rpn_cls.index_select(1, pos_index)) + torch.exp(rpn_cls.index_select(1, neg_index)))
        obj_indices = obj_score.nonzero()

        # Calculate roi from anchor and bounding-box transformation parameters
        self.anchor_sizes = torch.tensor(self.anchor_sizes, device = input_features.device)
        roi_bbox_x = obj_indices[:, 3].float() + self.anchor_sizes[obj_indices[:, 1], 1] * (rpn_bbox_tfm[obj_indices[:, 0], obj_indices[:, 1] * 4 + 1, obj_indices[:, 2], obj_indices[:, 3]] - 0.5) / self.base_stride
        roi_bbox_y = obj_indices[:, 2].float() + self.anchor_sizes[obj_indices[:, 1], 0] * (rpn_bbox_tfm[obj_indices[:, 0], obj_indices[:, 1] * 4, obj_indices[:, 2], obj_indices[:, 3]] - 0.5) / self.base_stride
        roi_bbox_w = self.anchor_sizes[obj_indices[:, 1], 1] / self.base_stride * torch.exp(rpn_bbox_tfm[obj_indices[:, 0], obj_indices[:, 1] * 4 + 3, obj_indices[:, 2], obj_indices[:, 3]])
        roi_bbox_h = self.anchor_sizes[obj_indices[:, 1], 0] / self.base_stride * torch.exp(rpn_bbox_tfm[obj_indices[:, 0], obj_indices[:, 1] * 4 + 2, obj_indices[:, 2], obj_indices[:, 3]])
        obj_score_2 = obj_score[obj_indices[:, 0], obj_indices[:, 1], obj_indices[:, 2], obj_indices[:, 3]]

        # Ensure roi is within the image's bounds
        roi_bbox_x = torch.clamp(roi_bbox_x, 0, feature_map.size(3) - 1)
        roi_bbox_y = torch.clamp(roi_bbox_y, 0, feature_map.size(2) - 1)
        roi_bbox_w = torch.clamp(roi_bbox_x + roi_bbox_w, 0, feature_map.size(3) - 1) - roi_bbox_x
        roi_bbox_h = torch.clamp(roi_bbox_y + roi_bbox_h, 0, feature_map.size(2) - 1) - roi_bbox_y

        # Rank objects by confidence score
        obj_score_2, indices = obj_score_2.sort(0, descending = True)
        obj_indices = obj_indices[indices, :]
        roi_bbox_x = roi_bbox_x[indices]
        roi_bbox_y = roi_bbox_y[indices]
        roi_bbox_w = roi_bbox_w[indices]
        roi_bbox_h = roi_bbox_h[indices]

        # Filter region proposals and classify objects for each image individually
        roi_bbox_x_img, roi_bbox_y_img, roi_bbox_w_img, roi_bbox_h_img = [], [], [], []
        for i in range(input_features.size(0)):
            index_img = (obj_indices[:, 0] == i).nonzero().squeeze(1)
            # print(obj_score_2.size())
            # print(torch.cuda.memory_allocated())
            # a = time.time()
            with torch.set_grad_enabled(False):
                roi_bbox_img = torch.stack((roi_bbox_x[index_img], roi_bbox_y[index_img], roi_bbox_w[index_img], roi_bbox_h[index_img]), dim = 0)
                obj_score_2, roi_bbox_img = non_maximum_suppression(obj_score_2[index_img], roi_bbox_img, obj_nms_threshold)
            # a = time.time() - a
            # print('Tempo NMS: ' + str(a) + 's')
            # print(obj_score_2.size())
            # print(roi_bbox_img.size())
            # input('Press enter')
            #roi_bbox_img = [roi_bbox_x[index_img], roi_bbox_y[index_img], roi_bbox_w[index_img], roi_bbox_h[index_img]]
            #iou_img = intersection_over_union(roi_bbox_img, roi_bbox_img)

            # Compute Non-Maximum Suppression for the region proposals
            #j = 0
            #while (j < iou_img.size(0)):
            #    nms_aux = iou_img[j, :].le(obj_nms_threshold)
            #    nms_aux[j] = 1
            #    nms_indices = nms_aux.nonzero().squeeze(1)
            #    iou_img = iou_img.index_select(0, nms_indices).index_select(1, nms_indices)
            #    index_img = index_img[nms_indices]
            #    j += 1

            # Take only the top obj_rank proposals, ranked by confidence score
            if(obj_rank > 0):
                # index_img = index_img[0:min(obj_rank, index_img.size(0))]
                roi_bbox_img = roi_bbox_img[:, :min(roi_bbox_img.size(1), obj_rank)]

            # roi_bbox_x_img = roi_bbox_x[index_img] * self.base_stride
            # roi_bbox_y_img = roi_bbox_y[index_img] * self.base_stride
            # roi_bbox_w_img = roi_bbox_w[index_img] * self.base_stride
            # roi_bbox_h_img = roi_bbox_h[index_img] * self.base_stride
            roi_bbox_x_img = roi_bbox_img[0, :] * self.base_stride
            roi_bbox_y_img = roi_bbox_img[1, :] * self.base_stride
            roi_bbox_w_img = roi_bbox_img[2, :] * self.base_stride
            roi_bbox_h_img = roi_bbox_img[3, :] * self.base_stride

            # Extract the object's features from roi 
            i_2 = torch.tensor(i, device = input_features.device).float()
            roi = torch.stack((i_2.expand(roi_bbox_x_img.size(0)), roi_bbox_x_img, roi_bbox_y_img, roi_bbox_x_img + roi_bbox_w_img, roi_bbox_y_img + roi_bbox_h_img), dim = 1)
            object_features = self.roi_align_layer(feature_map, roi)

            # Classify object and compute bounding-box adjusting parameters
            output_head = self.head(object_features).squeeze()
            sep = output_head.size(1) // 5
            final_cls = output_head[:, :sep]
            det_cls = torch.argmax(final_cls, dim = 1)
            final_bbox_tfm = output_head[:, sep:]
            all_ind = torch.arange(final_bbox_tfm.size(0)).long()

            # Calculate final bounding-box coordinates
            bbox_x = roi_bbox_x_img + roi_bbox_w_img * final_bbox_tfm[all_ind, det_cls * 4 + 1]
            bbox_y = roi_bbox_y_img + roi_bbox_h_img * final_bbox_tfm[all_ind,  det_cls * 4]
            bbox_w = roi_bbox_w_img * torch.exp(final_bbox_tfm[all_ind, det_cls * 4 + 3])
            bbox_h = roi_bbox_h_img * torch.exp(final_bbox_tfm[all_ind, det_cls * 4 + 2])
            final_bbox = torch.stack((i_2.expand(roi_bbox_x_img.size(0)), bbox_x, bbox_y, bbox_w, bbox_h), dim = 1)
            proposal_bbox = torch.stack((i_2.expand(roi_bbox_x_img.size(0)), roi_bbox_x_img, roi_bbox_y_img, roi_bbox_w_img, roi_bbox_h_img), dim = 1)
            # print('Total memory allocated: ' + str(torch.cuda.memory_allocated()))
        return rpn_cls, rpn_bbox_tfm, proposal_bbox, final_bbox, final_cls, final_bbox_tfm

def non_maximum_suppression(score, bbox, iou_threshold):
    score, indices = score.sort(0, descending = True)
    bbox[:, indices] = bbox
    indices = torch.ones(score.size(0), dtype = torch.uint8, device = score.device)
    indices_nz = indices.nonzero()
    #for i in range(indices.size(0)):
#    j = 0
#    while (j < indices_nz.size(0) - 1):
#        i = indices_nz[j]
#    for i in indices_nz:
        # if (indices[i] != 0):
#        iou = intersection_over_union(bbox[:, i].view(-1, 1), bbox[:, (i + 1):]).view(-1)
            # print(indices[(i + 1):].size())
            # print(indices[(i + 1):
            # print((iou >= iou_threshold).size())
#        indices[(i + 1):][iou >= iou_threshold] = 0
#        indices_nz = indices.nonzero()
#        j += 1
    j = 0
    block_end = -1
    while (j < indices_nz.size(0) - 1):
        i = indices_nz[j]
        if (i > block_end):
            block_start = i
            block_ind = indices_nz[j:min((j + 5000), indices_nz.size(0))].view(-1)
            # print(block_ind)
            # print(bbox[:, block_ind].size())
            block_end = block_ind[-1]
#            a = torch.cuda.memory_allocated()
#            b = time.time()
            iou_block = intersection_over_union(bbox[:, block_ind], bbox[:, (i + 1):])
#            b = time.time() - b
#            a = torch.cuda.memory_allocated() - a
#            print(iou_block.dtype)
#            print('Expected memory spent: ' + str(iou_block.size(0) * iou_block.size(1) * 4) + 'B')
#            print('Actual memory spent: ' + str(a) + 'B')
#            print('Time spent: ' + str(b) + 's')
#            input('Press Enter')
            filter_ind = (iou_block < iou_threshold)
#        print(filter_ind.size())
#        print(iou_block.size())
#        print((block_ind == i).size())
        indices[(i + 1):] = indices[(i + 1):] * filter_ind[block_ind == i, (i - block_start):].view(-1)
        indices_nz = indices.nonzero()
        j += 1
    # bbox_cpu = bbox.to(torch.device("cpu"))
#    iou = intersection_over_union(bbox, bbox)
 #   iou = (iou < iou_threshold)
    # print(iou[:10, :10])
  #  aux = iou.cumsum(dim = 1).diag()
   # indices = (aux == torch.arange(score.size(0), dtype = torch.long, device = score.device))
    # print(torch.arange(score.size(0), dtype = torch.long))
    # print(indices)
    return score[indices], bbox[:, indices]

def intersection_over_union(bbox_1, bbox_2):
    bbox_1_mat, bbox_2_mat = [], []
    # for i in range(4):
    #    bbox_1_mat.append(bbox_1[i].unsqueeze(1).expand(-1, bbox_2[i].size(0)))
    #    bbox_2_mat.append(bbox_2[i].unsqueeze(0).expand(bbox_1[i].size(0), -1))
    bbox_1_x_min = bbox_1[0].unsqueeze(1).expand(-1, bbox_2[0].size(0))
    bbox_1_y_min = bbox_1[1].unsqueeze(1).expand(-1, bbox_2[1].size(0))
    bbox_1_x_max = (bbox_1[0] + bbox_1[2]).unsqueeze(1).expand(-1, bbox_2[2].size(0))
    bbox_1_y_max = (bbox_1[1] + bbox_1[3]).unsqueeze(1).expand(-1, bbox_2[3].size(0))
    bbox_1_area = (bbox_1[2] * bbox_1[3]).unsqueeze(1).expand(-1, bbox_2[0].size(0))
    bbox_2_x_min = bbox_2[0].unsqueeze(0).expand(bbox_1[0].size(0), -1)
    bbox_2_y_min = bbox_2[1].unsqueeze(0).expand(bbox_1[1].size(0), -1)
    bbox_2_x_max = (bbox_2[0] + bbox_2[2]).unsqueeze(0).expand(bbox_1[2].size(0), -1)
    bbox_2_y_max = (bbox_2[1] + bbox_2[3]).unsqueeze(0).expand(bbox_1[3].size(0), -1)
    bbox_2_area = (bbox_2[2] * bbox_2[3]).unsqueeze(0).expand(bbox_1[0].size(0), -1)
    bbox_intersection_x = torch.max(bbox_1_x_min, bbox_2_x_min)
    bbox_intersection_y = torch.max(bbox_1_y_min, bbox_2_y_min)
    bbox_intersection_w = (torch.min(bbox_1_x_max, bbox_2_x_max) - bbox_intersection_x).clamp(min = 0)
    bbox_intersection_h = (torch.min(bbox_1_y_max, bbox_2_y_max) - bbox_intersection_y).clamp(min = 0)
    intersection = bbox_intersection_w * bbox_intersection_h
    union = bbox_1_area + bbox_2_area - intersection
    return intersection / union

def intersection_over_union_2(bbox_1, bbox_2):
    a = torch.cuda.memory_allocated()
    bbox_1_mat, bbox_2_mat = [], []
    for i in range(4):
        bbox_1_mat.append(bbox_1[i].unsqueeze(1).expand(-1, bbox_2[i].size(0)))
        bbox_2_mat.append(bbox_2[i].unsqueeze(0).expand(bbox_1[i].size(0), -1))
    print(torch.cuda.memory_allocated() - a)
    bbox_intersection_w = torch.min(bbox_1_mat[0] + bbox_1_mat[2], bbox_2_mat[0] + bbox_2_mat[2]) - torch.max(bbox_1_mat[0], bbox_2_mat[0])
    print(torch.cuda.memory_allocated() - a)
    bbox_intersection_h = torch.min(bbox_1_mat[1] + bbox_1_mat[3], bbox_2_mat[1] + bbox_2_mat[3]) - torch.max(bbox_1_mat[1], bbox_2_mat[1])
    print(torch.cuda.memory_allocated() - a)
    # bbox_intersection_w[bbox_intersection_w < 0] = 0
    bbox_intersection_w.clamp(min = 0)
    print(torch.cuda.memory_allocated() - a)
    # bbox_intersection_h[bbox_intersection_h < 0] = 0
    bbox_intersection_h.clamp(min = 0)
    print(torch.cuda.memory_allocated() - a)
    intersection = bbox_intersection_w * bbox_intersection_h
    print(torch.cuda.memory_allocated() - a)
    return intersection / (bbox_1_mat[2] * bbox_1_mat[3] + bbox_2_mat[2] * bbox_2_mat[3] - intersection)

def average_precision(dataset, classifier, iou_threshold):
    classifier.eval()
    torch.set_grad_enabled(False)
    n_pos = torch.zeros(len(dataset.class_list), device = device)
    conf_list, tp_list = [], []
    for i in range(len(dataset.class_list)):
        conf_list.append(torch.tensor([], device = device))
        tp_list.append(torch.tensor([], device = device))
    for n in range(len(dataset)):
        sample = dataset[n]
        inputs, labels = sample['image'], sample['annotations']
        inputs = inputs.to(device).unsqueeze(0).float()
        labels = labels.to(device).float()
        rpn_cls, rpn_bbox_tfm, proposal_bbox, final_bbox, final_cls, final_bbox_tfm = classifier(inputs)

        # Count ground truths occurences for each class
        for gt in labels:
            n_pos[int(gt[0].item())] += 1

        # Find true positives and false positives for every class in this image
        # aux = [[0, -1]] * labels.size(0)
        # conf_list, tp_list = [], []
        # for i in range(len(dataset.class_list) + 1):
        #     conf_list.append([])
        #     tp_list.append([])
        
        confidence, prediction = torch.max(final_cls, dim = 1)
        iou = intersection_over_union(final_bbox[:, 1:].transpose(0, 1), labels[:, 1:].transpose(0, 1))
        matches = (prediction.view(-1, 1) == labels[:, 0].view(1, -1).long())
        iou = iou * matches.float()
        # print(iou)
        # print(matches)
        # print(labels)
        iou_max, j_max = torch.max(iou, dim = 1)
        j_max[iou_max < iou_threshold] = -1
        conf_list_2, tp_list_2 = [], []
        for i in range(len(dataset.class_list)):
            conf_list_2.append(confidence[prediction == i])
            tp_list_2.append(torch.zeros(conf_list_2[-1].size(), device = device))
        pos_conf = confidence - torch.min(confidence) + 1
        for j in range(labels.size(0)):
            i = int(labels[j, 0].item())
            if (conf_list_2[i].size(0) != 0):
                print(torch.max(iou[:, j]))
                pos_conf = conf_list_2[i] - torch.min(conf_list_2[i]) + 1
                pos_conf_max, arg_max = torch.max(pos_conf * (j_max[prediction == i] == j).float(), dim = 0)
                if (pos_conf_max > 0):
                    tp_list_2[i][arg_max] = 1
        
        for i in range(len(dataset.class_list)):
            tp_list[i] = torch.cat((tp_list[i], tp_list_2[i]))
            conf_list[i] = torch.cat((conf_list[i], conf_list_2[i]))
        # for i, detection in enumerate(final_cls):
        #     iou_max = 0
        #     confidence, prediction = torch.max(detection, dim = 0)
        #     for j, gt in enumerate(labels):
        #         if (gt[0] == prediction.float()):
        #             iou = intersection_over_union(final_bbox[i, 1:].unsqueeze(1), gt[1:].unsqueeze(1))
        #             if (iou > iou_max):
        #                 iou_max = iou
        #                 j_max = j
        #     if (iou_max > iou_threshold):
        #         if (confidence > aux[j_max][0]):
        #             aux[j_max][0] = confidence
        #             aux[j_max][1] = i
        #     conf_list[prediction].append(confidence)
        #     tp_list[prediction].append(0)
        # for j in range(len(labels)):
        #     if (aux[j][1] != -1):
        #         tp_list[int(labels[j, 0].item())][aux[j][1]] = 1
        # for i in range(len(conf_list_2)):
        #     aux = torch.tensor(conf_list[i], device = inputs.device).float()
        #     print((conf_list_2[i] - aux).sum())
        #     print((conf_list_2[i] - aux).nonzero().sum())
        #     aux = torch.tensor(tp_list[i], device = inputs.device).float()
        #     print((tp_list_2[i] - aux).sum())
        #     print((tp_list_2[i] - aux).nonzero().sum())
        #     input('Press Enter')

    # Calculate average precision for every class
    ap = torch.zeros(len(dataset.class_list), device = device)
    for i in range(len(dataset.class_list)):
        cls_conf_list = conf_list[i]
        cls_tp_list = tp_list[i]
        print(cls_tp_list.size())
        if (cls_tp_list.size(0) != 0 and n_pos[i] != 0):
            print(cls_tp_list.sum())
            # Sort detections by confidence
            cls_conf_list, indices = cls_conf_list.sort()
            cls_tp_list = cls_tp_list[indices]

            # Compute precision and recall
            tp = torch.cumsum(cls_tp_list, dim = 0)
            fp = torch.cumsum(1 - cls_tp_list, dim = 0)
            rec = tp / n_pos[i]
            prec = tp / (fp + tp)

            # Calculate average precision
            mrec = torch.zeros(rec.size(0) + 2, device = device)
            mrec[1:(rec.size(0) + 1)] = rec
            mrec[rec.size(0) + 1] = 1
            mpre = torch.zeros(prec.size(0) + 2, device = device)
            mpre[1:(prec.size(0) + 1)] = prec
            for j in range(mpre.size(0) - 2, -1, -1):
                mpre[j] = torch.max(mpre[j], mpre[j + 1])
            ap[i] = torch.sum((mrec[1:mrec.size(0)] - mrec[0:(mrec.size(0) - 1)]) * mpre[1:mpre.size(0)])
    print(ap)
    return ap

def criterion(outputs, labels, inputs, base_stride, anchor_sizes, phase):
    rpn_cls, rpn_bbox_tfm, proposal_bbox, final_bbox, final_cls, final_bbox_tfm = outputs
    # Ground-truth classes and bounding-boxes
    gt_bbox = labels[:, :, 1:].transpose(1, 2)
    gt_cls = labels[:, :, 0].long()
    cross_entropy = torch.nn.CrossEntropyLoss(size_average = False)
    smooth_l1_loss = torch.nn.SmoothL1Loss(size_average = False)

    anchor_sizes = torch.tensor(anchor_sizes, device = inputs.device)
    # Calculate anchor bounding-box coordinates
    x_grid = (torch.arange(rpn_cls.size(3), device = inputs.device) * base_stride).unsqueeze(0).unsqueeze(1).repeat(anchor_sizes.size(0), rpn_cls.size(2), 1)
    y_grid = (torch.arange(rpn_cls.size(2), device = inputs.device) * base_stride).unsqueeze(0).unsqueeze(2).repeat(anchor_sizes.size(0), 1, rpn_cls.size(3))
    anchor_bbox_x = x_grid - anchor_sizes[:, 1].view(-1, 1, 1).repeat(1, x_grid.size(1), x_grid.size(2)) * 0.5
    anchor_bbox_y = y_grid - anchor_sizes[:, 0].view(-1, 1, 1).repeat(1, y_grid.size(1), y_grid.size(2)) * 0.5
    anchor_bbox_w = anchor_sizes[:, 1].view(-1, 1, 1).repeat(1, x_grid.size(1), y_grid.size(2))
    anchor_bbox_h = anchor_sizes[:, 0].view(-1, 1, 1).repeat(1, y_grid.size(1), y_grid.size(2))
    if (phase == 'val'):
        # Ensure anchor is within the image's bounds
        delta = torch.clamp(anchor_bbox_x, 0, inputs.size(3) - 1) - anchor_bbox_x
        anchor_bbox_x = anchor_bbox_x + delta
        anchor_bbox_w = anchor_bbox_w - delta
        delta = torch.clamp(anchor_bbox_y, 0, inputs.size(2) - 1) - anchor_bbox_y
        anchor_bbox_y = anchor_bbox_y + delta
        anchor_bbox_h = anchor_bbox_h - delta
        anchor_bbox_w = torch.clamp(anchor_bbox_x + anchor_bbox_w, 0, inputs.size(3) - 1) - anchor_bbox_x
        anchor_bbox_h = torch.clamp(anchor_bbox_y + anchor_bbox_h, 0, inputs.size(2) - 1) - anchor_bbox_y
    else:
        # If anchor isn't within bounds, exclude it from training
        ind = ((anchor_bbox_x < 0) + (anchor_bbox_y < 0) + (anchor_bbox_x + anchor_bbox_w > inputs.size(3) - 1) + (anchor_bbox_y + anchor_bbox_h > inputs.size(2) - 1)).clamp(0, 1)
        anchor_bbox_x[ind] = -1
        anchor_bbox_y[ind] = -1
        anchor_bbox_w[ind] = 0
        anchor_bbox_h[ind] = 0
    for n in range(rpn_cls.size(0)):
        anchor_bbox_x_flat = anchor_bbox_x.permute(1, 2, 0).contiguous().view(-1)
        anchor_bbox_y_flat = anchor_bbox_y.permute(1, 2, 0).contiguous().view(-1)
        anchor_bbox_w_flat = anchor_bbox_w.permute(1, 2, 0).contiguous().view(-1)
        anchor_bbox_h_flat = anchor_bbox_h.permute(1, 2, 0).contiguous().view(-1)

        # Computes IoU between ground-truth objects and anchors
        gt_anchor_iou = intersection_over_union(gt_bbox[n], (anchor_bbox_x_flat, anchor_bbox_y_flat, anchor_bbox_w_flat, anchor_bbox_h_flat))
        gt_anchor_iou_argmax = torch.argmax(gt_anchor_iou, dim = 1)

        # Check which anchors have sufficient IoU with ground-truth objects
        prop_iou_max, prop_best_match = torch.max(gt_anchor_iou, dim = 0)
        pos_ind = (prop_iou_max >= 0.7)
        prop_best_match[1 - pos_ind] = -1
        gt_iou_max, gt_iou_argmax = torch.max(gt_anchor_iou, dim = 1)
        pos_ind[gt_iou_argmax[gt_iou_max > 0]] = 1
        neg_ind = (prop_iou_max < 0.3) * (anchor_bbox_w_flat != 0) * (1 - pos_ind)
        prop_best_match[gt_iou_argmax[gt_iou_max > 0]] = torch.arange(0, gt_iou_max.size(0), device = inputs.device).long()[gt_iou_max > 0]
        anchor_neg_indices = neg_ind.nonzero().view(-1)
        anchor_pos_indices = pos_ind.nonzero().view(-1)
        prop_best_match = prop_best_match[prop_best_match >= 0]
        # print('Object: ' + str(len(anchor_pos_indices)))
        # print('Not object: ' + str(len(anchor_neg_indices)))

        # Cross entropy loss and bounding-box transform loss for proposals
        rpn_cls_flat = rpn_cls.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_bbox_tfm_flat = rpn_bbox_tfm.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        if (len(anchor_pos_indices) != 0):
            if (anchor_pos_indices.size(0) > 128):  
                perm = torch.randperm(anchor_pos_indices.size(0))[0:128]
                anchor_pos_indices = anchor_pos_indices[perm]
                prop_best_match = prop_best_match[perm]
            gt_bbox_tfm = torch.zeros(anchor_pos_indices.size(0), 4, device = inputs.device)
            gt_bbox_tfm[:, 0] = (gt_bbox[n, 1, prop_best_match] - anchor_bbox_y_flat[anchor_pos_indices]) / anchor_bbox_h_flat[anchor_pos_indices]
            gt_bbox_tfm[:, 1] = (gt_bbox[n, 0, prop_best_match] - anchor_bbox_x_flat[anchor_pos_indices]) / anchor_bbox_w_flat[anchor_pos_indices]
            gt_bbox_tfm[:, 2] = torch.log(gt_bbox[n, 3, prop_best_match] / anchor_bbox_h_flat[anchor_pos_indices])
            gt_bbox_tfm[:, 3] = torch.log(gt_bbox[n, 2, prop_best_match] / anchor_bbox_w_flat[anchor_pos_indices])
            loss_obj = cross_entropy(rpn_cls_flat.index_select(0, anchor_pos_indices), torch.zeros(anchor_pos_indices.size(0), dtype = torch.long, device = inputs.device)) / 256
            loss_obj_tfm = smooth_l1_loss(rpn_bbox_tfm_flat.index_select(0, anchor_pos_indices), gt_bbox_tfm) / 256
        else:
            loss_obj = torch.zeros(1, device = inputs.device)
            loss_obj_tfm = torch.zeros(1, device = inputs.device)
        if (len(anchor_neg_indices) != 0):
            if (anchor_neg_indices.size(0) > 256 - len(anchor_pos_indices)):
                perm = torch.randperm(anchor_neg_indices.size(0))[0:(256 - anchor_pos_indices.size(0))]
                anchor_neg_indices = anchor_neg_indices[perm]
            loss_obj = loss_obj + (cross_entropy(rpn_cls_flat.index_select(0, anchor_neg_indices), torch.ones(anchor_neg_indices.size(0), dtype = torch.long, device = inputs.device))) / 256
        # print('Object detection loss: ' + str(loss_obj.item()))
        # print('Object bbox transformation loss: ' + str(loss_obj_tfm.item()))

        # Define targets for the classification layer
        proposal_bbox_n = proposal_bbox[proposal_bbox[:, 0] == n]
        gt_proposal_iou = intersection_over_union((proposal_bbox_n[:, 1], proposal_bbox_n[:, 2], proposal_bbox_n[:, 3], proposal_bbox_n[:, 4]), gt_bbox[n])
        j = torch.argmax(gt_proposal_iou, dim = 1)
        i = torch.arange(0, proposal_bbox_n.size(0), dtype = torch.long, device = inputs.device)
        cls_indices = torch.stack((i, j, gt_cls[n, j]), dim = 1)
        aux = ((gt_proposal_iou[i, j] < 0.5) * (gt_proposal_iou[i, j] >= 0.1)).nonzero().view(-1)
        if (len(aux) > 0):
            bg_indices = cls_indices[aux, 0].view(-1)
        else:
            bg_indices = torch.tensor([], dtype = torch.long, device = inputs.device)
        aux = (gt_proposal_iou[i, j] >= 0.5).nonzero().view(-1)
        if (len(aux) > 0):
            cls_indices = cls_indices[aux, :]
        else:
            cls_indices = torch.tensor([], dtype = torch.long, device = inputs.device)
        # print(cls_indices[:, 1].view(-1))
        # print((cls_indices[:, 1] == 12).sum())
        # print((cls_indices[:, 1] == 14).sum())
        # input('p')
        # print('Background proposals (<50% IoU): ' + str(len(bg_indices)))
        # print('Good proposals (>50% IoU): ' + str(len(cls_indices)))

        # Cross entropy loss and bounding-box transform loss for classification
        perm = torch.randperm(len(cls_indices))[0:min(len(cls_indices), 32)]
        loss_cls = torch.zeros(1, device = inputs.device)
        loss_bg = torch.zeros(1, device = inputs.device)
        loss_cls_tfm = torch.zeros(1, device = inputs.device)
        for i in perm:
            ind = cls_indices[i]
            loss_cls = loss_cls + cross_entropy(final_cls[ind[0]].squeeze().unsqueeze(0), ind[2].unsqueeze(0))
            gt_bbox_tfm = torch.zeros(4, device = inputs.device)
            gt_bbox_tfm[0] = (gt_bbox[n, 0, ind[1]] - proposal_bbox[ind[0]][1]) / proposal_bbox[ind[0]][3]
            gt_bbox_tfm[1] = (gt_bbox[n, 1, ind[1]] - proposal_bbox[ind[0]][2]) / proposal_bbox[ind[0]][4]
            gt_bbox_tfm[2] = torch.log(gt_bbox[n, 2, ind[1]] / proposal_bbox[ind[0]][3])
            gt_bbox_tfm[3] = torch.log(gt_bbox[n, 3, ind[1]] / proposal_bbox[ind[0]][4])
            gt_bbox_tfm_no_grad = gt_bbox_tfm.detach()
            loss_cls_tfm = loss_cls_tfm + smooth_l1_loss(final_bbox_tfm[ind[0]][(4 * ind[2]):(4 * (ind[2] + 1))], gt_bbox_tfm_no_grad)
        perm = torch.randperm(len(bg_indices))[0:min(len(bg_indices), 96)]
        for i in perm:
            ind = bg_indices[i]
            loss_bg = loss_bg + cross_entropy(final_cls[ind].squeeze().unsqueeze(0), 20 * torch.ones(1, dtype = torch.long, device = inputs.device))
            #print(20 * torch.ones(1, dtype = torch.long, device = inputs.device))
            #print(final_cls[ind])
            #print(cross_entropy(final_cls[ind].squeeze().unsqueeze(0), 20 * torch.ones(1, dtype = torch.long, device = inputs.device)))
            #print(loss)
            #input('Press enter')
        loss_cls = (loss_cls + loss_bg) / 128
        loss_cls_tfm = loss_cls_tfm / 128
        # loss_bg = loss_bg / 128
        # print('Classification loss: ' + str(loss_cls.item()))
        # print('Background loss: ' + str(loss_bg.item()))
        # loss = loss_obj + loss_obj_tfm + loss_cls + loss_bg
    return loss_obj, loss_obj_tfm, loss_cls, loss_cls_tfm

def train_FasterRCNN(model, criterion, optimizer, scheduler, num_epochs, dataset, base_stride, anchor_sizes):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_mAP_05 = -1
    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0

            dataset[phase].shuffle()
            # Iterate over data
            epoch_time = time.time()
            for samples in dataset[phase]:
                # samples = dataset[phase][0]
                inputs = samples['image'].to(device).unsqueeze(0)
                inputs = inputs.float()
                labels = samples['annotations'].to(device).unsqueeze(0)
                labels = labels.float()

                # Zero the parameter gradient
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                step_time = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    # forward_time = time.time()
                    # print(inputs.size())
                    # print('Initial memory allocated: ' + str(torch.cuda.memory_allocated()))
                    outputs = model(inputs)
                    # print('Total memory allocated: ' + str(torch.cuda.memory_allocated()))
                    # forward_time = time.time() - forward_time
                    # print('Forward complete in {:.4f}s'.format(forward_time))
                    # criterion_time = time.time()
                    loss_obj, loss_obj_tfm, loss_cls, loss_cls_tfm = criterion(outputs, labels, inputs, base_stride, anchor_sizes, phase)
                    loss = loss_obj + loss_obj_tfm + loss_cls + loss_cls_tfm
                    # criterion_time = time.time() - criterion_time
                    # print('Loss computation complete in {:.4f}s'.format(criterion_time))
                    # print('Class transform loss: ' + str(loss_cls_tfm.item()))
                    # print('Total loss: ' + str(loss.item()))

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        # back_time = time.time()
                        loss.backward()
                        optimizer.step()
                        # back_time = time.time() - back_time
                        # print('Backward complete in {:.4f}s'.format(back_time))
                step_time = time.time() - step_time
                # print('Step complete in {:.4f}s'.format(step_time))

                # Statistics
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataset[phase])
            epoch_time = time.time() - epoch_time
            print('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format(epoch_time // 3600, epoch_time // 60 % 60, epoch_time % 60))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # input('Press Enter')
            mAP_time = time.time()
            mAP_05 = average_precision(dataset[phase], model, 0.5).mean().item()
            mAP_time = time.time() - mAP_time
            print('{} mAP@0.5: {:.4f}'.format(phase, mAP_05))
            print('mAP computed in {:.0f}m {:.0f}s'.format(mAP_time // 60, mAP_time % 60))

            # If this is the best model, deep copy it
            if phase == 'val' and mAP_05 > best_mAP_05:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
    print('Best validation loss: {:.4f}'.format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.current_device())
#roi_layer_test = RoIAlignAvg(7, 7, 1.0/16)
#a = torch.rand(1, 256, 38, 50).to(device)
#b = torch.tensor([[0, 2, 2, 10, 10]]).to(device)
#roi_layer_test(a, b)
# Load dataset
annotation_dir = 'VOCdevkit/VOC2012/Annotations'
image_dir = 'VOCdevkit/VOC2012/JPEGImages'
file_list = dict()
with open('VOCdevkit/VOC2012/ImageSets/Main/train.txt') as f:
    file_list['train'] = f.read().splitlines()
with open('VOCdevkit/VOC2012/ImageSets/Main/val.txt') as f:
    file_list['val'] = f.read().splitlines()
class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
transform = torchvision.transforms.Compose([odd.Rescale(600, 1000), odd.ToTensor(), odd.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
dataset = {x: odd.ObjectDetectionDataset(annotation_dir, image_dir, file_list[x], class_list, transform) for x in ['train', 'val']}
# print('Train dataset size:' + str(len(dataset['train'])))
# print('Validation dataset size:' + str(len(dataset['val'])))
num_epochs = 20

# Load model
resnet = torchvision.models.resnet18(pretrained = True)
# resnet = torchvision.models.resnet101(pretrained = True)
print(sum(p.numel() for p in resnet.parameters()))
model_base = torch.nn.Sequential(*list(resnet.children())[:7])
# model_base = torch.nn.Sequential(*list(resnet.children())[:7])
base_stride = 16
base_out_channels = 256
# base_out_channels = 1024
#for a in resnet.children():
#    print(a)
model_head = torch.nn.Sequential(*list(resnet.children())[7:9], torch.nn.Conv2d(512, len(class_list) * 5 + 5, 1))
# model_head = torch.nn.Sequential(*list(resnet.children())[7:9], torch.nn.Conv2d(2048, len(class_list) * 5 + 5, 1))
rpn_channels = 256
# rpn_channels = 1024
rpn_kernel_size = 3
anchor_sizes = ((128, 128), (128 * 2 ** 0.5, 128 / 2 ** 0.5), (128 / 2 ** 0.5, 128 * 2 ** 0.5),
                (256, 256), (256 * 2 ** 0.5, 256 / 2 ** 0.5), (256 / 2 ** 0.5, 256 * 2 ** 0.5),
                (512, 512), (512 * 2 ** 0.5, 512 / 2 ** 0.5), (512 / 2 ** 0.5, 512 * 2 ** 0.5))
print(anchor_sizes)
roi_grid_size = 28
roi_align_kernel_size = 2
roi_align_stride = 2
faster_rcnn = FasterRCNN(model_base, base_stride, base_out_channels, rpn_channels, rpn_kernel_size, anchor_sizes, roi_grid_size, roi_align_kernel_size, roi_align_stride, model_head)
faster_rcnn.to(device)

# Train model
optimizer = torch.optim.SGD(faster_rcnn.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
model = train_FasterRCNN(faster_rcnn, criterion, optimizer, exp_lr_scheduler, num_epochs, dataset, base_stride, anchor_sizes)
torch.save(model.state_dict(), './fasterrcnn_resnet18.pth')
