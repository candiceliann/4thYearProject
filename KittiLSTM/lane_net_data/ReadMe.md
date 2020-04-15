##KITTI DATASET INFERENCE DATA##


Uses three subsets of the kitti training dataset:


* inf_data21.json 
	* This has the first 101 (0-100 inclusive) images from the sample in subdir 0020/ 
* inf_data22.json
	* This is 226 - 336 as a subset of the sample in subdir 0018/
* inf_data23.json
	* This is 240 - 389 as a subset of the sample in subdir 0008/

I put these images in a seperate sub directories (00021/ 0022/ 0023/ respectively as I thought it would be easier for viewing the images )


I also used had to resample this to use on the testing data - to do this change the fourth cell in kitti_lstm5 to the following:

```python

sequence_list = []
lane_test_sequence = []
lane_frames = []
i = 0
for sequence in data:
    frame_dict = {}
    for unique_object in sequence:
        if len(sequence[unique_object]) > 99:
            #if i == 7:
                #print(unique_object)
            for frame in sequence[unique_object]:
                if frame['frame'] in frame_dict.keys():
                    frame_dict[frame['frame']].append(frame['bbox'])
                else:
                    frame_dict[frame['frame']] = [frame['bbox']]
                #print(frame)
    frame_sample_dict = {}
    for frame in frame_dict:
        if len(frame_dict[frame]) > 2:
            if len(frame_sample_dict) < 150:
                frame_sample_dict[frame] = frame_dict[frame][:3]
    if len(frame_sample_dict) == 150:
        sequence_list.append(frame_sample_dict)
        print(i)
    lane_frame_sample_dict = {}
    if i == 9:
        for j in range(240, 400):
            if str(j) in frame_dict.keys():
                lane_frames.append(j)
                lane_frame_sample_dict[str(j)] = frame_dict[str(j)][:3]
        lane_test_sequence.append(lane_frame_sample_dict)
    i += 1               
np.array(sequence_list).shape
```



After this you need to re sample the new data - just use these two additional cells, they do the same thing as the others that look basically idenitcal so should just plug and play :)




```python

ylane = []
list_lengths = []
for sequence in lane_test_sequence:
    sequence_of_frames = []
    for frame in sequence:
        frame_list = []
        for tuples in sequence[frame]:
            frame_list.append(tuples[0])
            frame_list.append(tuples[1])
            frame_list.append(tuples[2])
            frame_list.append(tuples[3])
        sequence_of_frames.append(frame_list)
    ylane.append(sequence_of_frames)
ylane = np.array(ylane)
```



```python

ynormlane = np.zeros(ylane.shape)
sample_list_lane = []
for sample in range(0,ynormlane.shape[0]):
    list_0 = []
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    list_9 = []
    list_10 = []
    list_11 = []
    for clip in range(0, ynormlane.shape[1]):
        clip_list = []
        list_0.append(ylane[sample, clip, 0])
        list_1.append(ylane[sample, clip, 1])
        list_2.append(ylane[sample, clip, 2])
        list_3.append(ylane[sample, clip, 3])
        list_4.append(ylane[sample, clip, 4])
        list_5.append(ylane[sample, clip, 5])
        list_6.append(ylane[sample, clip, 6])
        list_7.append(ylane[sample, clip, 7])
        list_8.append(ylane[sample, clip, 8])
        list_9.append(ylane[sample, clip, 9])
        list_10.append(ylane[sample, clip, 10])
        list_11.append(ylane[sample, clip, 11])
    sample_list_lane.append([list_0, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11])
sample_list_lane = np.array(sample_list_lane).astype(np.float)
ynormlane = np.zeros(sample_list_lane.shape)
y_minmax = np.zeros((sample_list_lane.shape[0], sample_list_lane.shape[1], 2))
for sample in range(0, sample_list_lane.shape[0]):
    for clip in range(0, sample_list_lane.shape[1]):
        y_min = np.amin(sample_list_lane[sample, clip])
        y_max= np.amax(sample_list_lane[sample, clip])
        y_minmax[sample, clip, 0] = y_min
        y_minmax[sample, clip, 1] = y_max
        for value in range(0, sample_list_lane.shape[2]):
            ynormlane[sample, clip, value] = (float(sample_list_lane[sample, clip, value]) - y_min)/(y_max - y_min)
y_newlane = np.zeros(y.shape)
for sample in range(0, len(ynormlane)):
    for sets in  range(0, len(ynormlane[0])):
        for frame_value in range(0, len(ynormlane[0][0])):
            y_newlane[sample, frame_value, sets] = ynormlane[sample, sets, frame_value]
```



#Finally#

Change the testing sets to the following:

```python
yte = y_newlane[:1, 1:150, :]
Xte = y_newlane[:1, :149, :]
```

