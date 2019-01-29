import os
import json
import numpy as np


def calc_union(min_1, max_1, min_2, max_2):
    return np.max([max_1, max_2]) - np.min([min_1, min_2])


def calc_overlap(min_1, max_1, min_2, max_2):
    return np.max([0, np.min([max_1, max_2]) - np.max([min_1, min_2])])


def compute_iou(union, overlap):
    return overlap/union


def compute_ap(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)


def compute_map(clases_ap, params):
    return sum(clases_ap.values()) / params['classes_amount']


def compute_classes_ap(clases_true_positive, clases_false_positive):
    clases_ap = {}
    for label in clases_true_positive:
        clases_ap[label] = compute_ap(
            clases_true_positive[label],
            clases_false_positive[label])

    return clases_ap


def compute_threshold_map(prediction, ground_truth, metadata, params):
    threshold_map = dict()
    thresholds = np.linspace(0.5, 0.95, num=10)

    for threshold_iou in thresholds:
        clases_fp = dict()
        clases_tp = dict()
        for key, value in prediction['results'].items():
            
            ground_truth_key = ground_truth['database'][key]['annotations']
            for predicted_segment in prediction['results'][key]:

                true_positive = 0
                false_positive = 0
                                
                for ground_truth_key_item in ground_truth_key:
                    
                    predicted_label = predicted_segment['label']
                    truth_label = ground_truth_key_item['label']
                    
                    metadata_label = metadata[truth_label]['level_2']['name']
                    
                    if predicted_label == metadata_label:

                        min_1 = predicted_segment['segment'][0][0]
                        max_1 = predicted_segment['segment'][0][1]

                        min_2 = ground_truth_key_item['segment'][0]
                        max_2 = ground_truth_key_item['segment'][1]

                        union = calc_union(min_1, max_1, min_2, max_2)
                        overlap = calc_overlap(min_1, max_1, min_2, max_2)
                        iou = compute_iou(union, overlap)
                        
                        if iou >= threshold_iou:
                            true_positive = 1
                            false_positive = 0
                            break
                        else:
                            false_positive = 1

                    else:
                        
                        false_positive = 1           
                
                if predicted_label in clases_tp.keys():
                    clases_tp[predicted_label] += true_positive
                    clases_fp[predicted_label] += false_positive
                else:
                    clases_tp[predicted_label] = true_positive
                    clases_fp[predicted_label] = false_positive
                       
        clases_ap = compute_classes_ap(clases_tp, clases_fp)
        threshold_map[threshold_iou] = compute_map(clases_ap, params)

    return threshold_map


def compute_average_map(params):
    ground_truth_json = params['json_data_path']
    prediction_json = os.path.join(params['data_dir'], 'predicted_output.json')
    metadata_path = params['json_metadata_path']

    with open(ground_truth_json, 'r') as file:
        ground_truth = json.load(file)

    with open(prediction_json, 'r') as file:
        prediction = json.load(file)

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    threshold_map = compute_threshold_map(prediction, ground_truth, metadata, params)
    num_threshold = len(threshold_map)

    return sum(threshold_map.values()) / num_threshold
