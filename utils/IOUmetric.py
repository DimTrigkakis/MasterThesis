'''

import pickle

gt = pickle.load(open("./results/frame_labelling/2014091408/labels/gt_viewpoint_segments.pkl","rb"))
cl = pickle.load(open("./results/frame_labelling/2014091408/labels/viewpoint_segments.pkl","rb"))

cl_true = []
for idx,i in enumerate(cl):
    if idx>= len(gt):
        break
    else:
        a0,a1,b0,b1 = i[0][0],i[0][1],i[1][0],i[1][1]
        datum = ((a0,a1*2),(b0,b1*2))
        cl_true.append(datum)

print len(gt), len(cl_true)
print gt
print " \n\n"
print cl_true
'''

#print IoU_pairs(cl_true,gt)

def IoU_pairs(A,B,classes=[0,1,2], weighted = False):

    threshold = 0.95

    #A = (((1,0),(50,0)),((51,1),(127,1)),((128,2),(450,2)),((451,1),(600,1)))
    #B = (((1,0),(50,0)),((51,1),(127,1)),((128,2),(450,2)),((451,1),(600,1))) 

    iou_total = 0

    frames_included = 0
    found = 0
    b_length = 0
    tpfp = 0
    for pair_B in B:

        if pair_B[0][1] not in classes:
            continue

        b_length += 1
        # find matching pair in A based on IoU
        start_frame_B = pair_B[0][0]
        end_frame_B = pair_B[1][0]

        pairs = []
        for pair_A in A:
            start_frame_A = pair_A[0][0]
            end_frame_A = pair_A[1][0]
            label_A = pair_A[0][1]
        
            pairs.append((start_frame_A,end_frame_A,label_A))        
        
        # find best matching pair with same label
        # sort pairs based on intersection score, the biggest intersection wins
        b_was_found = False
        for pair in pairs:

            sfA, efA, lA = pair
            if efA <= start_frame_B or sfA >= end_frame_B:
                iou = 0
                # if you don't find anything, you lose the entire ground truth (IoU of 0)
                # otherwise, compute intersection over union from pair
            else:
                a1,a2,a3,a4 = max(start_frame_B,sfA), min(end_frame_B,efA), min(start_frame_B,sfA),max(end_frame_B,efA)
                intersection = a2-a1
                union = a4-a3
                iou = (1.0 * intersection)/union

            if iou > threshold:
                #print(pair_B, start_frame_B, end_frame_B, pair, sfA, efA, iou, intersection, union )
                if lA != pair_B[0][1]:
                    iou = 0
                iou_total += iou
                frames_included += 1
                #print(frames_included)
                b_was_found = True 

        if b_was_found:
            found += 1

        # multiply the iou score with the length of the sequence
        # divide with the total IoU possible in B (which should be the length of the frames)
    if frames_included != 0:
        return iou_total*1.0, frames_included,b_length, found, len(A)
    else:
        return -1,-1, b_length, found, len(A)

#print IoU_pairs(cl_true,gt)
