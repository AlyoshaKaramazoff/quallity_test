import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='manuscript_handtext_quallity_test',
                                     description="Manuscript quallity test tool",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i0', '--input_seg_file', nargs='?', required=True,
                        help='path to .seg file')
    parser.add_argument('-i1', '--input_result_file', nargs='?', required=True,
                        help='path to result file')
    parser.add_argument('-o', '--output_folder', nargs='?', required=True,
                        help='path to result directory')
    # parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug mode')
    # parser.add_argument('-in', '--inference', action='store_true', default = True, help='inference mode')
    # parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.3')
    return parser.parse_args()

args = get_args()
input_seg_file = args.input_seg_file
input_result_file = args.input_result_file
output_folder = args.output_folder

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou