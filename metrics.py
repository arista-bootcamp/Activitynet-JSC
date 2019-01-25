import json
import os

import numpy as np

def calc_union(min_1,max_1,min_2,max_2):
    return np.max([max_1,max_2]) - np.min([min_1,min_2])

def calc_overlap(min_1,max_1,min_2,max_2):
    return np.max([0,np.min([max_1,max_2])-np.max([min_1,min_2])])

def compute_iou(union,overlap):
	return round(union,2)/round(overlap,2)

def compute_AP(true_positive,false_positive):
    return true_positive/(true_positive+false_positive)

def compute_mAP(clases_ap,params):
    return sum(clases_ap.values())/params['classes_amount']

def compute_classes_ap(clases_true_positive,clases_false_positive):
    clases_ap = {}
    for label in clases_true_positive:
        clases_ap[label] = compute_AP(
            clases_true_positive[label],
            clases_false_positive[label])
        
    return clases_ap

def compute_threshold_mAP(prediction,ground_truth,params):

    threshold_mAP = dict()
    thresholds = np.linspace(0.5, 0.95, num=10)
    thresholds = [0.5]
    for threshold_iou in thresholds:
        clases_fp = dict()
        clases_tp = dict()
        for key,value in prediction['results'].items():
            ground_truth_key = ground_truth['database'][key]['annotations']
            for list_indice,predicted_segment in enumerate(prediction['results'][key]):

                true_positive = 0
                false_positive = 0
                predicted_label = predicted_segment['label']
                truth_label = ground_truth_key[list_indice]['label']

                if predicted_label == truth_label:

                    min_1 = predicted_segment['segment'][0]
                    max_1 = predicted_segment['segment'][1]

                    min_2 = ground_truth_key[list_indice]['segment'][0]
                    max_2 = ground_truth_key[list_indice]['segment'][1]

                    union = calc_union(min_1,max_1,min_2,max_2)
                    overlap = calc_overlap(min_1,max_1,min_2,max_2)
                    iou = compute_iou(union,overlap)

                    if iou >= threshold_iou:
                        true_positive = 1
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


        clases_ap = compute_classes_ap(clases_tp,clases_fp)
        threshold_mAP[threshold_iou] = compute_mAP(clases_ap,params)
        
    return threshold_mAP


def compute_average_mAP(params):

	ground_truth_json = os.path.join(params['data_dir'],'activity_net.v1-3.min.json')
	prediction_json = os.path.join(params['data_dir'], 'sample_detection_prediction.json')

	with open(ground_truth_json, 'r') as file:
	    ground_truth = json.load(file)
	    
	with open(prediction_json, 'r') as file:
	    prediction = json.load(file)

	threshold_mAP = compute_threshold_mAP(prediction,ground_truth,params)
	num_threshold = len(threshold_mAP)

	return sum(threshold_mAP.values())/num_threshold