import os
import gradio as gr  # type: ignore
from paddleocr import PaddleOCR  # type: ignore
from ultralytics import YOLO  # type: ignore
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
import cv2  # type: ignore
import numpy as np
import re
from internetarchive import download  # type: ignore
from tqdm import trange

download("anpr_weights", files=["anpr.pt"], verbose=True)  # type: ignore

download(
    "anpr_examples_202208",
    files=["test_image_1.jpg", "test_image_2.jpg", "test_image_3.jpeg", "test_video_1.mp4"],  # type: ignore
    verbose=True,
)

paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)

model = YOLO(model="./anpr_weights/anpr.pt", task="detect")


def detect_plates(src):
    predictions = model.predict(src, verbose=False)

    results = []

    for prediction in predictions:
        for box in prediction.boxes:
            det_confidence = box.conf.item()
            if det_confidence < 0.6:
                continue
            coords = [int(position) for position in (box.xyxy.view(1, 4)).tolist()[0]]
            results.append({"coords": coords, "det_conf": det_confidence})

    return results


def crop(img, coords):
    cropped = img[coords[1] : coords[3], coords[0] : coords[2]]
    return cropped


def preprocess_image(src):
    normalize = cv2.normalize(
        src, np.zeros((src.shape[0], src.shape[1])), 0, 255, cv2.NORM_MINMAX
    )
    denoise = cv2.fastNlMeansDenoisingColored(
        normalize, h=10, hColor=10, templateWindowSize=7, searchWindowSize=15
    )
    grayscale = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold


def ocr_plate(src):
    # Preprocess the image for better OCR results
    preprocessed = preprocess_image(src)

    # OCR the preprocessed image
    results = paddle.ocr(preprocessed, det=False, cls=True)

    # Get the best OCR result
    plate_text, ocr_confidence = max(
        results,
        key=lambda ocr_prediction: max(
            ocr_prediction,
            key=lambda ocr_prediction_result: ocr_prediction_result[1],
        ),
    )[0]

    # Filter out anything but uppercase letters, digits, hypens and whitespace.
    # Also, remove hypens and whitespaces at the first and last positions
    plate_text_filtered = re.sub(r"[^A-Z0-9- ]", "", plate_text).strip("- ")

    return {"plate": plate_text_filtered, "ocr_conf": ocr_confidence}


def ocr_plates(src, det_predictions):
    results = []

    for det_prediction in det_predictions:
        plate_region = crop(src, det_prediction["coords"])
        ocr_prediction = ocr_plate(plate_region)
        results.append(ocr_prediction)

    return results


def plot_box(img, coords, label=None, color=[0, 150, 255], line_thickness=3):
    # Plots box on image
    c1, c2 = (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    # Plots label on image, if exists
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[
            0
        ]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            line_thickness / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def get_plates(src):
    det_predictions = detect_plates(src)
    ocr_predictions = ocr_plates(src, det_predictions)

    for det_prediction, ocr_prediction in zip(det_predictions, ocr_predictions):
        plot_box(src, det_prediction["coords"], ocr_prediction["plate"])

    return src, det_predictions, ocr_predictions


def predict_image(src):
    detected_image, det_predictions, ocr_predictions = get_plates(src)
    return detected_image


def predict_image_api(src):
    detected_image, det_predictions, ocr_predictions = get_plates(src)
    return ocr_predictions[0]["plate"]


def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]


def get_best_ocr(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info["track_id"] == track_id:
            # Check if the ocr confidence is maximum or not
            if info["ocr_conf"] < rec_conf:
                info["ocr_conf"] = rec_conf
                info["ocr_txt"] = ocr_res
            else:
                rec_conf = info["ocr_conf"]
                ocr_res = info["ocr_txt"]
            break
    return preds, rec_conf, ocr_res


def predict_video(src):
    output = f"{Path(src).stem}_detected{Path(src).suffix}"

    # Create a VideoCapture object
    video = cv2.VideoCapture(src)

    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object.
    temp = f"{Path(output).stem}_temp{Path(output).suffix}"
    export = cv2.VideoWriter(
        temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Intializing tracker
    tracker = DeepSort()

    # Initializing some helper variables.
    preds = []
    total_obj = 0

    for i in trange(frames_total):
        ret, frame = video.read()
        if ret is True:
            # Run the ANPR algorithm
            det_predictions = detect_plates(frame)
            # Convert Pascal VOC detections to COCO
            bboxes = list(
                map(
                    lambda bbox: pascal_voc_to_coco(bbox),
                    [det_prediction["coords"] for det_prediction in det_predictions],
                )
            )

            if len(bboxes) > 0:
                # Storing all the required info in a list.
                detections = [
                    (bbox, score, "number_plate")
                    for bbox, score in zip(
                        bboxes,
                        [
                            det_prediction["det_conf"]
                            for det_prediction in det_predictions
                        ],
                    )
                ]

                # Applying tracker.
                # The tracker code flow: kalman filter -> target association(using hungarian algorithm) and appearance descriptor.
                tracks = tracker.update_tracks(detections, frame=frame)

                # Checking if tracks exist.
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # Changing track bbox to top left, bottom right coordinates
                    bbox = [int(position) for position in list(track.to_tlbr())]

                    for i in range(len(bbox)):
                        if bbox[i] < 0:
                            bbox[i] = 0

                    # Cropping the license plate and applying the OCR.
                    plate_region = crop(frame, bbox)
                    ocr_prediction = ocr_plate(plate_region)
                    plate_text, ocr_confidence = (
                        ocr_prediction["plate"],
                        ocr_prediction["ocr_conf"],
                    )

                    # Storing the ocr output for corresponding track id.
                    output_frame = {
                        "track_id": track.track_id,
                        "ocr_txt": plate_text,
                        "ocr_conf": ocr_confidence,
                    }

                    # Appending track_id to list only if it does not exist in the list
                    # else looking for the current track in the list and updating the highest confidence of it.
                    if track.track_id not in list(
                        set(pred["track_id"] for pred in preds)
                    ):
                        total_obj += 1
                        preds.append(output_frame)
                    else:
                        preds, ocr_confidence, plate_text = get_best_ocr(
                            preds,
                            ocr_confidence,
                            plate_text,
                            track.track_id,
                        )

                    # Plotting the prediction.
                    plot_box(
                        frame,
                        bbox,
                        f"{str(track.track_id)}. {plate_text}",
                        color=[255, 150, 0],
                    )

            # Write the frame into the output file
            export.write(frame)
        else:
            break

    # When everything done, release the video capture and video write objects
    video.release()
    export.release()

    # Compressing the video for smaller size and web compatibility.
    os.system(
        f"ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}"
    )
    os.system(f"rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree")
    return output


with gr.Blocks() as demo:
    gr.Markdown('### <h3 align="center">Automatic Number Plate Recognition</h3>')
    gr.Markdown(
        "This AI was trained to detect and recognize number plates on vehicles."
    )
    with gr.Tabs():
        with gr.TabItem("Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
                image_input.upload(
                    predict_image,
                    inputs=[image_input],
                    outputs=[image_output],
                )
            with gr.Row(visible=False):  # Prediction API
                api_image_input = gr.Image()
                api_prediction_output = gr.Textbox()
                api_image_input.upload(
                    predict_image_api,
                    inputs=[api_image_input],
                    outputs=[api_prediction_output],
                    api_name="predict",
                )
            gr.Examples(
                [
                    ["./anpr_examples_202208/test_image_1.jpg"],
                    ["./anpr_examples_202208/test_image_2.jpg"],
                    ["./anpr_examples_202208/test_image_3.jpeg"],
                ],
                [image_input],
                [image_output],
                predict_image,
                cache_examples=True,
            )
        with gr.TabItem("Video"):
            with gr.Row():
                video_input = gr.Video(format="mp4")
                video_output = gr.Video(format="mp4")
                video_input.upload(
                    predict_video, inputs=[video_input], outputs=[video_output]
                )
            gr.Examples(
                [["./anpr_examples_202208/test_video_1.mp4"]],
                [video_input],
                [video_output],
                predict_video,
                cache_examples=True,
            )
    gr.Markdown("[@itsyoboieltr](https://github.com/itsyoboieltr)")

demo.launch()
