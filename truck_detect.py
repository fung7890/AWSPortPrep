import panoramasdk
import cv2
import numpy as np
import time
import boto3


# Global Variables

HEIGHT = 512
WIDTH = 512


class TruckDetection(panoramasdk.base):
    def interface(self):
        return {
            "parameters":
            (
                ("float", "threshold", "Detection threshold", 0.50),
                ("model", "truck_test_model",
                 "Model for testing truck detection, weights are currently for multiple COCO objects", "model"),
                ("int", "batch_size", "Model batch size", 1),
                ("float", "truck_index",
                 "truck index based on pretrained model's dataset, COCO", 8),
            ),
                "inputs":
            (
                ("media[]", "video_in", "Camera input stream"),
            ),
                "outputs":
            (
                ("media[video_in]", "video_out", "Camera output stream"),
            )
        }

    """init() function is called once by the Lambda runtime.  It gives Lambda a chance to perform 
    any necessary initialization before entering the main loop."""

    def init(self, parameters, inputs, outputs):
        try:
            # Frame Number Initialization
            self.frame_num = 0
            # # Index for truck from parameters
            self.index = parameters.truck_index
            # Set threshold for model from parameters
            self.threshold = parameters.threshold
            # set number of trucks
            self.number_trucks = 0

            # Load model from the specified directory.
            print("loading the model...")
            self.model = panoramasdk.model()
            self.model.open(parameters.truck_test_model, 1)
            print("model loaded")

            # Create input and output arrays.
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.class_array = np.empty(
                class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(
                prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(
                rect_info.get_dims(), dtype=rect_info.get_type())

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def preprocess(self, img):
        resized = cv2.resize(img, (HEIGHT, WIDTH))

        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]  # RGB

        img = resized.astype(np.float32) / \
            255.  # converting array of ints to floats
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]

        # Extracting single channels from 3 channel image
        # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)

        # normalizing per channel data:
        img_a = (img_a - mean[0]) / std[0]
        img_b = (img_b - mean[1]) / std[1]
        img_c = (img_c - mean[2]) / std[2]

        # putting the 3 channels back together:
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c

        # x1 = mx.nd.array(np.asarray(x1))
        x1 = np.asarray(x1)
        return x1

    def get_number_trucks(self, class_data, prob_data):
        # get indices of truck detections in class data
        truck_indices = [i for i in range(
            len(class_data)) if int(class_data[i]) == self.index]
        # use these indices to filter out anything that is less than 95% threshold from prob_data
        prob_truck_indices = [
            i for i in truck_indices if prob_data[i] >= self.threshold]
        return prob_truck_indices

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            truck_image = stream.image

            stream.add_label('Number of Trucks : {}'.format(
                self.number_trucks), 0.6, 0.05)

            x1 = self.preprocess(truck_image)

            # Do inference on the new frame.
            self.model.batch(0, x1)
            self.model.flush()

            # Get the results.
            resultBatchSet = self.model.get_result()

            class_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            rect_batch = resultBatchSet.get(2)

            class_batch.get(0, self.class_array)
            prob_batch.get(0, self.prob_array)
            rect_batch.get(0, self.rect_array)

            class_data = self.class_array[0]
            prob_data = self.prob_array[0]
            rect_data = self.rect_array[0]

            # Get Indices of classes that correspond to truck
            truck_indices = self.get_number_trucks(class_data, prob_data)

            print('Truck indices is {}'.format(truck_indices))

            try:
                self.number_trucks = len(truck_indices)
            except:
                self.number_trucks = 0

            # Draw Bounding Boxes on HDMI Output
            if self.number_trucks > 0:
                for index in truck_indices:

                    left = np.clip(rect_data[index]
                                   [0] / np.float(HEIGHT), 0, 1)
                    top = np.clip(rect_data[index][1] / np.float(HEIGHT), 0, 1)
                    right = np.clip(
                        rect_data[index][2] / np.float(HEIGHT), 0, 1)
                    bottom = np.clip(
                        rect_data[index][3] / np.float(HEIGHT), 0, 1)

                    stream.add_rect(left, top, right, bottom)
                    stream.add_label(str(prob_data[index][0]), right, bottom)

            stream.add_label('Number of Truck : {}'.format(
                self.number_truck), 0.6, 0.05)

            self.model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True


def main():
    TruckDetection().run()

main()