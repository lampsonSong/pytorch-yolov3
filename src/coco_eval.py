# created by lampson.song @ 2020-4-9
# to evaluate coco dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)                                                                                             
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def convert_out_format(image_id, detection_outs, coco91cls, threshold):
    # detection_outs: [N,6] , 0:4 (x1,y1,x2,y2), 4 confidence, 5 class_ids
    
    image_result = []
    rois = detection_outs[:, :4].clone()
    scores = detection_outs[:, 4]
    class_ids = detection_outs[:, 5]

    if rois.shape[0] > 0:
        # x1,y1,x2,y2 -> x1,y1,w,h
        rois[:, 2] -= rois[:, 0]
        rois[:, 3] -= rois[:, 1]

        bbox_score = scores

        for roi_id in range(rois.shape[0]):
            score = float(bbox_score[roi_id])
            label = coco91cls[int(class_ids[roi_id])]
            box = rois[roi_id, :]

            if score < threshold:
                break
            image_result.append( {
                'image_id': image_id,
                'category_id': label,
                'score': float(score),
                'bbox': box.tolist(),
            } )

    return image_result


def save_json(file_name, results):
    if not len(results):
        return []

    # write output
    json.dump(results, open(file_name, 'w'), indent=4)


def get_coco_eval(val_file, pred_json_path, val_image_ids=None):
    coco_gt = COCO(val_file)
    MAX_NUM = 10000
    if val_image_ids == None:
        image_ids = coco_gt.getImgIds()[:MAX_NUM]
    else:
        image_ids = val_image_ids
    
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()    
    coco_eval.summarize()

    return coco_eval.stats

if __name__ == '__main__':
    set_name = 'val2017'
    val_file = '../data/coco/annotations/instances_{}.json'.format(set_name)

    coco_gt = COCO(val_file)
    MAX_NUM = 10000
    image_ids = coco_gt.getImgIds()[:MAX_NUM]
    print(image_ids)
