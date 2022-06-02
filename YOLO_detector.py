import cv2
import numpy as np
import keras.models as models


class Detector:

    def __init__(self, cfg):

        # Load models
        self.yolo_net = cv2.dnn.readNet(cfg.modelConfig, cfg.modelWeights)
        self.output_layers = [self.yolo_net.getLayerNames()[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

        self.colors = np.random.uniform(0, 255, size=(len(self.clNames), 3))

        self.dog_breed_net = models.load_model(cfg.dog_breed_net)

        # Load list of classes
        with open(cfg.classes, 'rt') as f:
            self.clNames = f.read().rstrip().split('\n')
        with open(cfg.dog_breed_classes, 'rt') as f:
            self.dog_breeds = f.read().rstrip().split('\n')

        # Thresholds
        self.confThreshold = cfg.confThreshold
        self.nmsThreshold = cfg.nmsThreshold
        self.needClasses = cfg.needClasses


    def find_objects(self, source_img):

        # Detecting objects
        img = source_img.copy()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(self.output_layers)

        # Process photo
        ht, wt, _ = img.shape
        bbox = []
        class_ids = []
        confs = []
        obj = []

        for output in outs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    w, h = int(det[2] * wt), int(det[3] * ht)
                    x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        # NMS Process
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)
        bbox = np.array(bbox)[indices]
        class_ids = np.array(class_ids)[indices]

        # Setup result
        for i in range(len(indices)):
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            obj.append(self.clNames[class_ids[i]].capitalize())
            cv2.rectangle(img, (x, y), (x + w, y + h), self.colors[class_ids[i]], 5)
            cv2.putText(img, f'{self.clNames[class_ids[i]].upper()}',
                        (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 4)
        return img, obj, bbox

    def find_needed_classes(self, img, objs, bboxes):
        breeds = []
        images = []
        for index, obj in enumerate(objs):

            # Only for needed classes
            if obj in self.needClasses:
                box = bboxes[index]
                x, y, w, h = box[0], box[1], box[2], box[3]
                new_img = img[y:y+h, x:x+w]
                images.append(new_img)

                # Prepare img to NN
                new_img = cv2.resize(new_img, (299, 299))
                new_img = np.expand_dims(new_img, axis=0)

                # Append predictions
                predictions = self.dog_breed_net.predict(new_img)
                index = np.argmax(predictions)
                breeds.append(str(f'{self.dog_breeds[index]} breed - I\'m {round(predictions[0][index]*100, 2)}% sure'))

        return images, breeds
