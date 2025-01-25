import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import os

# Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Vibrant green
ALL_COORDINATES = []
np.random.seed(43)

# Functions

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and annotate handedness on the image."""
    global ALL_COORDINATES
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = handedness_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Get image dimensions
        height, width, _ = annotated_image.shape

        # Store hand coordinates
        hand_coordinates = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand_landmarks
        ]
        ALL_COORDINATES.append({
            "handedness": handedness[0].category_name,
            "coordinates": hand_coordinates
        })

        # Annotate handedness on the image
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image

def show_mask(mask, ax, obj_id=None, random_color=False):
    """Visualize a mask on the image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Visualize points with labels."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Visualize a bounding box on the image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def process_image(image_path, model_path):
    """Detect hand landmarks and visualize them on an image."""
    # Create HandLandmarker object
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Load input image
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    # Annotate image
    rgb_image = image.numpy_view()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    annotated_image = draw_landmarks_on_image(bgr_image, detection_result)

    # Display annotated image
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_video(video_dir, model_cfg, sam2_checkpoint):
    """Segment objects in video frames using SAM2."""
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cpu")

    # Scan all JPEG frames in the directory
    frame_names = [p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    # Interact with a specific frame
    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[585, 322], [762, 377]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )

    # Display results on the frame
    plt.figure(figsize=(9, 6))
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.axis('off')
    plt.show()

    # Propagate segmentation throughout the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Visualize results every few frames
    for out_frame_idx in range(0, len(frame_names), 10):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        plt.show()

# Main Execution
if __name__ == "__main__":
    IMAGE_PATH = "images/1.jpg"
    MODEL_PATH = "hand_landmarker.task"
    VIDEO_DIR = "images"
    SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # Process image
    process_image(IMAGE_PATH, MODEL_PATH)

    # Process video
    process_video(VIDEO_DIR, MODEL_CFG, SAM2_CHECKPOINT)
