import cv2
import math
from math import ceil
import os
import time
import random
import sqlite3
import numpy as np
from ultralytics import YOLO
from PIL import Image
try:
    from ultralytics.yolo.utils.plotting import Annotator
except:
    from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort
from classificationmodel import CowIdentificationModel
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
from database import Database
from utils import *



class Inference:
    def __init__(self, video_path, output_video_path, model_path='./models/yolov8/best.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.output_video_path = output_video_path
        self.object_tracker = DeepSort(max_age=4)
        self.video_name = os.path.basename(video_path).split('.mp4')[0]
        self.cow_activity_video_path = os.path.join('static','Activity-Videos',self.video_name,'Raw-Videos')
        self.fps = cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FPS) if cv2.VideoCapture(self.video_path).isOpened() else 30
        os.makedirs(self.cow_activity_video_path, exist_ok = True)

        self.db = Database()
        self.cow_images_path = os.path.join('static','Cow-Images', self.video_name.split('.mp4')[0])
        self.new_cows_path = os.path.join('static','Cow-Images','New')
        if not os.path.exists(self.new_cows_path):
            os.makedirs(self.new_cows_path)

        if not os.path.exists(self.cow_images_path):
            os.makedirs(self.cow_images_path)

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classifier_path = './models/classifier/best_model.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classifier = CowIdentificationModel().to(self.device)
        self.classifier.load_state_dict(torch.load(self.classifier_path, map_location=self.device))
        self.classifier.eval()
        self.minimum_threshold = 0.12
        self.cow_cluster_conf = {}

    def predict_cluster_id(self,image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #image = Image.open(cow_image_path).convert("RGB")
        image_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
        
        # Get the prediction
        with torch.no_grad():
            output = self.classifier(image_tensor)
            # _, predicted_class = torch.max(output, 1)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class = torch.max(probabilities, 1)
        classes = ['cluster_1', 'cluster_10', 'cluster_100', 'cluster_101', 'cluster_102', 'cluster_104', 'cluster_105', 'cluster_107', 'cluster_111', 'cluster_112', 'cluster_118', 'cluster_12', 'cluster_121', 'cluster_123', 'cluster_124', 'cluster_126', 'cluster_127', 'cluster_128', 'cluster_129', 'cluster_13', 'cluster_130', 'cluster_131', 'cluster_133', 'cluster_135', 'cluster_137', 'cluster_139', 'cluster_140', 'cluster_148', 'cluster_155', 'cluster_158', 'cluster_159', 'cluster_16', 'cluster_162', 'cluster_176', 'cluster_18', 'cluster_183', 'cluster_192', 'cluster_194', 'cluster_2', 'cluster_200', 'cluster_201', 'cluster_207', 'cluster_21', 'cluster_217', 'cluster_222', 'cluster_223', 'cluster_224', 'cluster_226', 'cluster_227', 'cluster_228', 'cluster_23', 'cluster_238', 'cluster_24', 'cluster_241', 'cluster_248', 'cluster_258', 'cluster_26', 'cluster_266', 'cluster_27', 'cluster_277', 'cluster_283', 'cluster_288', 'cluster_289', 'cluster_29', 'cluster_290', 'cluster_294', 'cluster_296', 'cluster_297', 'cluster_3', 'cluster_30', 'cluster_31', 'cluster_32', 'cluster_33', 'cluster_34', 'cluster_36', 'cluster_37', 'cluster_39', 'cluster_40', 'cluster_42', 'cluster_43', 'cluster_44', 'cluster_46', 'cluster_47', 'cluster_48', 'cluster_49', 'cluster_50', 'cluster_51', 'cluster_53', 'cluster_54', 'cluster_56', 'cluster_57', 'cluster_60', 'cluster_62', 'cluster_64', 'cluster_65', 'cluster_67', 'cluster_68', 'cluster_69', 'cluster_70', 'cluster_71', 'cluster_75', 'cluster_76', 'cluster_77', 'cluster_78', 'cluster_79', 'cluster_80', 'cluster_81', 'cluster_82', 'cluster_83', 'cluster_85', 'cluster_86', 'cluster_91', 'cluster_93', 'cluster_94', 'cluster_95', 'cluster_96', 'cluster_98']

        predicted_class_label = classes[predicted_class.item()]
        prob = probabilities[0, predicted_class.item()].item()
        return predicted_class_label, prob

    def inference(self):
        activity_frames = {}
        cap = cv2.VideoCapture(self.video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, frame_rate,(frame_width, frame_height))

        event_count = {}
        object_missing_count = {}

        frameNumber = 0
        while True:

            brushBusy = False
            watertubBusy = False
            cowsOccupancy = 0

            success, frame = cap.read()
            start = time.perf_counter()
            yolo_bbox , ids_missing_in_frame = [], []

            if success:
                frame_copy = frame.copy()
                activity_single_frame = frame.copy()
                results = self.model(frame, conf=0.6)
                detections = []

                for result in results:
                    annotator = Annotator(frame)
                    boxes = result.boxes

                    for box in boxes:
                        box_list = box.xywh.tolist()
                        conf = box.conf.tolist()
                        cls = box.cls.tolist()
                        box_ind = box_list[0]

                        left, top, width, height, r, b = convert_to_top_left_v1(box_ind[0], box_ind[1], box_ind[2], box_ind[3])
                        yolo_bbox.append([left, top, r, b])
                        detections.append(([left, top, width, height], conf[0], int(cls[0])))

                        if int(cls[0]) == 1:
                            cowsOccupancy += 1

                tracks = self.object_tracker.update_tracks(detections, frame=frame)

                id_class_mapping, del_tracks = {}, []

                for i, track in enumerate(tracks):
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    cls = track.det_class

                    ltrb = track.to_ltrb()
                    ltrb = ltrb.tolist()
                    x, y = calculate_centroid(ltrb)

                    if is_present(ltrb, yolo_bbox):
                        del_tracks.append(int(track_id))
                        if cls == 0:
                            id_class_mapping[cls] = [ltrb, track_id, [x, y]]
                        if cls == 2:
                            id_class_mapping[cls] = [ltrb, track_id, [x, y]]
                        if cls == 1:
                            if cls in id_class_mapping:
                                id_class_mapping[cls].append([ltrb, track_id, [x, y]])
                            else:
                                id_class_mapping[cls] = [[ltrb, track_id, [x, y]]]
                            cow_tracking_id = track_id
                            
                            cow_image_name = f"cow_{cow_tracking_id}_{str(frameNumber)}.png"
                            left, top, right, bottom = map(int, ltrb)
                            cow_image = frame_copy[top:bottom, left:right]
                            if frameNumber % 5 == 0:
                                imagePath = os.path.join(self.cow_images_path, cow_image_name)
                                try:
                                    cluster_id, prob = self.predict_cluster_id(cow_image)
                                    cluster_name = 0
                                    if prob > self.minimum_threshold:
                                        cv2.imwrite(imagePath, cow_image)
                                    else:
                                        cluster_name = 'New'
                                        image_name = f'cow_new_{cow_tracking_id}_{str(frameNumber)}.png'
                                        imagePath = os.path.join(self.new_cows_path, self.video_name+image_name)
                                        cv2.imwrite(imagePath , cow_image)
                                        

                                    if cow_tracking_id not in self.cow_cluster_conf:
                                        self.cow_cluster_conf[cow_tracking_id] = {
                                            'cluster_id':cluster_id,
                                            'prob':prob
                                        }
                                    else:
                                        previous_prob = self.cow_cluster_conf[cow_tracking_id]['prob']
                                        if prob > previous_prob:
                                            self.cow_cluster_conf[cow_tracking_id]['cluster_id'] = cluster_id
                                            self.cow_cluster_conf[cow_tracking_id]['prob'] = prob
                                        else :
                                            cluster_id = self.cow_cluster_conf[cow_tracking_id]['cluster_id']
                                    # print("=========",cow_tracking_id,":",self.cow_cluster_conf[cow_tracking_id]['cluster_id'],":",self.cow_cluster_conf[cow_tracking_id]['prob'],"=========")
                                    if cluster_name == 'New':
                                        cluster_id = 'New'
                                        cow_tracking_id = -1
                                    self.db.insert_cow_Images_data(cow_tracking_id, imagePath, self.video_name, cluster_id)
                                except:
                                    pass
                        bbox = ltrb
                        start_range, end_range = 0, 255

                        random_number = random.randint(start_range, end_range)

                        #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (random_number, random_number, 255), 2)
                        cv2.putText(frame, "id:" + str(track_id) + str(self.model.names[int(cls)]), (int(bbox[0]) - 110, int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), 15)

                        if track_id in object_missing_count:
                            object_missing_count[track_id] = 0

                    else:
                        if track_id not in object_missing_count:
                            object_missing_count[track_id] = 0
                        else:
                            object_missing_count[track_id] += 1

                        if object_missing_count[track_id] >= 4:
                            if track_id in event_count:
                                del event_count[track_id]

                even_list = list(event_count.keys())

                for delt in even_list:
                    if delt in del_tracks:
                        pass
                    else:
                        try:
                            del event_count[delt]
                        except:
                            pass

                if 0 in id_class_mapping:
                    brush_data = id_class_mapping[0]
                    bbox_brush = brush_data[0]
                    track_id_brush = brush_data[1]
                    centroid_brush = brush_data[2]

                    if 1 in id_class_mapping:
                        cows_data = id_class_mapping[1]
                        for cow in cows_data:
                            bbox_cow = cow[0]
                            track_id_cow = cow[1]
                            centroid_cow = cow[2]

                            distance = calculate_centroid_distance(
                                centroid_brush, centroid_cow)
                            overlap_status, overlap_area = are_boxes_overlapping(
                                bbox_brush, bbox_cow)

                            if overlap_status and distance < 200:

                                brushBusy = True

                                if int(track_id_cow) in event_count:
                                    event_count[int(track_id_cow)] = event_count[int(
                                        track_id_cow)] + 1
                                else:
                                    event_count[int(track_id_cow)] = 0

                                frame_rate_seconds = float(
                                    event_count[int(track_id_cow)]) / float(frame_rate)

                                '''activity_video_code'''
                                start_range, end_range = 0, 255
                                random_number = random.randint(start_range, end_range)
                                new_frame_copy = activity_single_frame.copy()
                                cv2.rectangle(new_frame_copy, (int(bbox_cow[0]), int(bbox_cow[1])), (int(bbox_cow[2]), int(bbox_cow[3])), (random_number, random_number, 255), 2)
                                
                                cv2.putText(frame, f'brushing-time:{str(round(frame_rate_seconds, 3))} s', (int(bbox_brush[0]) + 100, int(bbox_brush[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
                                tracking_id = track_id_cow +"Brushing"
                                if tracking_id in activity_frames:
                                    activity_frames[tracking_id].append(new_frame_copy)
                                else:
                                    activity_frames[tracking_id] = [new_frame_copy]
                                #print(activity_single_frame.shape)
                                # Insert data into the SQLite table
                                self.db.insert_cow_events_data(
                                    track_id_cow, 'Brushing', (round(frame_rate_seconds, 3)), self.video_name)

                if 2 in id_class_mapping:
                    watertub_data = id_class_mapping[2]
                    bbox_watertub = watertub_data[0]
                    track_id_watertub = watertub_data[1]
                    centroid_watertub = watertub_data[2]

                    if 1 in id_class_mapping:  # Cow data
                        cows_data = id_class_mapping[1]
                        for cow in cows_data:
                            bbox_cow = cow[0]
                            track_id_cow = cow[1]
                            centroid_cow = cow[2]

                            distance = calculate_centroid_distance(
                                centroid_watertub, centroid_cow)
                            overlap_status, overlap_area = are_boxes_overlapping(
                                bbox_watertub, bbox_cow)

                            if overlap_status and distance < 200:

                                watertubBusy = True
                                if int(track_id_cow) in event_count:
                                    event_count[int(track_id_cow)] = event_count[int(
                                        track_id_cow)] + 1
                                else:
                                    event_count[int(track_id_cow)] = 0

                                frame_rate_seconds = float(
                                    event_count[int(track_id_cow)]) / float(frame_rate)
                                '''activity_video_code'''
                                start_range, end_range = 0, 255
                                random_number = random.randint(start_range, end_range)
                                new_frame_copy = activity_single_frame.copy()
                                cv2.rectangle(new_frame_copy, (int(bbox_cow[0]), int(bbox_cow[1])), (int(bbox_cow[2]), int(bbox_cow[3])), (random_number, random_number, 255), 2)

                                cv2.putText(frame, f'drinking-time:{str(round(frame_rate_seconds, 3))} s', (int(bbox_watertub[0]) + 100, int(bbox_watertub[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
                                tracking_id = track_id_cow+ 'Drinking'
                                if tracking_id in activity_frames:
                                    activity_frames[tracking_id].append(new_frame_copy)
                                else:
                                    activity_frames[tracking_id] = [new_frame_copy]
                                
                                self.db.insert_cow_events_data(track_id_cow, 'Drinking', (round(frame_rate_seconds, 3)),self.video_name)

                end = time.perf_counter()
                totalTime = end - start
                fps = 1 / totalTime

                frameNumber += 1

                if frameNumber % 50 == 0: 
                    brushBusy = 'yes' if brushBusy else 'no'
                    watertubBusy = 'yes' if watertubBusy else 'no'
                    self.db.insert_cow_occupancy_data(frameNumber, cowsOccupancy, brushBusy, watertubBusy, self.video_name)
                out.write(frame)
                frame = cv2.resize(frame, (640, 480))
                # cv2.imshow('frame', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break

        for track_id_cow, frames in activity_frames.items():
            video_path = os.path.join(self.cow_activity_video_path, track_id_cow+'.mp4')
            try:
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
                activity_video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
                for frame in frames:
                    activity_video.write(frame)
                activity_video.release()
            except:
                pass     
        # print(self.cow_cluster_conf)
        out.release()
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    inp = "static/input_video/7eaf3c45-74d5-4858-b184-ff379b560850output_video_compressed.mp4"
    out = "video.mp4"

    inf = Inference(inp, out)
    inf.inference()