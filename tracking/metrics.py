"""
Annotations File format:

Give the names in order left-to-right. If there is a change in positioning, put a new ordering with the start frame of the change (indexed at 0). If unknown person name, then put "Unknown 1", "Unknown 2", etc. as their name.


frame_start1, Name1, Name2, Name3, ...
frame_start2, Name1, Name2, Name3, ...



Annotations format

{frame_id: [Name1, Name2, Name3, ...]}



"""

def load_annotations(file_name, n_frames=3000):
    with open(file_name) as f:
        lines = f.readlines()
    lines_st = []
    for line in lines:
        if len(line) > 2:
            lines_st.append(line)
    lines = lines_st
    
    annotations = {}
    for i in range(len(lines)):
        line = [x.strip() for x in lines[i].split(',')]
        if i == len(lines)-1:
            start_frame2 = n_frames
        else:
            start_frame2 = int(lines[i+1].split(',')[0].strip())

        start_frame = int(line[0].strip())
        names = line[1:]
        for k in range(start_frame, start_frame2):
            annotations[k] = names

    return annotations


def get_detection_accuracy(video_detections, annotations):
    """
    video_detections: output of detect_faces_video {frame_id: (boxes, labels), ...}
    annotations format: {frame_id: [Name1, Name2, Name3, ...], ...}
    
    """
    
    n_frames = len(video_detections)
    n_total = 0
    n_tp = 0
    n_fp = 0
    n_fn = 0
    
    for i in range(n_frames):
        #print(annotations[i])
        #print(video_detections[i])
        #print()
        n_gt = len(annotations[i])
        n_pr = len(video_detections[i][1])
        
        n_total += n_gt
        n_tp += min(n_pr, n_gt)
        n_fp += max(n_pr-n_gt, 0)
        n_fn += max(n_gt-n_pr, 0)
        
        
    print('# of GT Faces: %d. # of Detected Faces: %d.' % (n_total, n_tp))
    print('Detection Precision: %.2f. Detection Recall: %.2f.' % (n_tp/float(n_tp+n_fp), n_tp/float(n_tp+n_fn)))
    

def get_classification_accuracy(video_detections, annotations):
    """
    video_detections: output of detect_faces_video {frame_id: (boxes, labels), ...}  # bboxes are (top, right, bottom, left)
    annotations format: {frame_id: [Name1, Name2, Name3, ...], ...}
    
    """
    
    n_frames = len(video_detections)
    
    cor = {'Kevin': 0, 'Irene': 0, 'Danny': 0, 'Uknown': 0}
    inc = {'Kevin': 0, 'Irene': 0, 'Danny': 0, 'Uknown': 0}
    
    for i in range(n_frames):
        #print(annotations[i])
        #print(video_detections[i])
        #print()
        n_gt = len(annotations[i])
        n_pr = len(video_detections[i][1])
        
        if n_gt != n_pr:
            continue

        bboxes, labels, ids = video_detections[i]
        bbox_to_label = dict([(bboxes[b], labels[b]) for b in range(len(bboxes))]) 
        
        sorted_bboxes = sorted(bboxes, key = lambda y: (y[3], y[0]))
        #print([(bbox, bbox_to_label[bbox]) for bbox in sorted_bboxes])
        #print()
        
        for b, bbox in enumerate(sorted_bboxes):
            if bbox_to_label[bbox].lower() in annotations[i][b].lower():
                try:
                    cor[annotations[i][b]] += 1
                except:
                    cor['Uknown'] += 1
            else:
                try:
                    inc[annotations[i][b]] += 1
                except:
                    inc['Uknown'] += 1
            
    print('Kevin: # of Correct Classifications: %d. # of Incorrect Classifications: %d. Accuracy: %.2f' % (cor['Kevin'], inc['Kevin'], 100*cor['Kevin']/float(cor['Kevin'] + inc['Kevin'])))
    print('Irene: # of Correct Classifications: %d. # of Incorrect Classifications: %d. Accuracy: %.2f' % (cor['Irene'], inc['Irene'], 100*cor['Irene']/float(cor['Irene'] + inc['Irene'])))
    print('Danny: # of Correct Classifications: %d. # of Incorrect Classifications: %d. Accuracy: %.2f' % (cor['Danny'], inc['Danny'], 100*cor['Danny']/float(cor['Danny'] + inc['Danny'])))
    print('Unknown: # of Correct Classifications: %d. # of Incorrect Classifications: %d. Accuracy: %.2f' % (cor['Uknown'], inc['Uknown'], 100*cor['Uknown']/float(cor['Uknown'] + inc['Uknown'] + 1e-9)))
    
    
#anns = load_annotations('vid1_annotations.txt')
#dets = # TODO get detections for video
#get_detection_accuracy(dets, anns)
#get_classification_accuracy(dets, anns)
    

