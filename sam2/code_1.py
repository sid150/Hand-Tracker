#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
ALL_COORDINATES=[]
np.random.seed(43)
def draw_landmarks_on_image(rgb_image, detection_result):
    global ALL_COORDINATES  # Declare global to store the coordinates across function calls
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
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

        # Extract and store (x, y) pixel coordinates for this hand
        hand_coordinates = [
            (int(landmark.x * width), int(landmark.y * height))  # Convert normalized to pixel coordinates
            for landmark in hand_landmarks
        ]
        ALL_COORDINATES.append({
            "handedness": handedness[0].category_name,
            "coordinates": hand_coordinates
        })

        # Draw handedness (left or right hand) on the image.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.show()


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))  

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# STEP 1: Import the necessary modules.

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("images/1.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
# Convert the input image from RGB to BGR before using OpenCV
rgb_image = image.numpy_view()
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Annotate the image and visualize it
annotated_image = draw_landmarks_on_image(bgr_image, detection_result)
# print(detection_result)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
  # To hide axis labels for better visualization
# plt.show()
all_coordinates = np.array([])  # Start with an empty array

for hand in ALL_COORDINATES:
    coordinates = np.array(hand['coordinates'])  # Convert each list of coordinates to numpy array
    all_coordinates = np.vstack([all_coordinates, coordinates]) if all_coordinates.size else coordinates  # Stack the arrays
# print(all_coordinates)

def show_points(coords, labels, ax, marker_size=100):
    # Plot each point based on its label
    unique_labels = np.unique(labels)
    
    # Plot the points with different colors for each unique label
    for label in unique_labels:
        label_points = coords[labels == label]
        
        # You can use different colors or markers for each label
        ax.scatter(label_points[:, 0], label_points[:, 1], marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
    # Optionally, annotate each point with its label index
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), color='black', fontsize=12, ha='center', va='center')

image=Image.open('images/1.jpg')
image = np.array(image.convert("RGB"))
input_label = np.arange(0,42)
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(all_coordinates, input_label, plt.gca())
plt.axis('on')
# plt.show()  


# if rgb_image.shape[-1] == 4:  # If the image has 4 channels (RGBA)
#     rgb_image = rgb_image[:, :, :3]



from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import os

sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cpu")

video_dir = "images"


# 585 322
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# plt.show()

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


points = np.array([[585, 322],[762,377]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1,1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[585, 322], [667, 187],[762,377],[801,463]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1,1,1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

plt.show()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 10
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.show()
# image = Image.open('images/1.jpg')
# predictor.set_image(image)
# input_point=all_coordinates[:43]
# # input_label=np.array([1]*42)


# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]

# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

# # mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
# # sam2_result = mask_generator.generate(rgb_image)
# # print(sam2_result)

# # def show_output(result_dict,axes=None):
# #      if axes:
# #         ax = axes
# #      else:
# #         ax = plt.gca()
# #         ax.set_autoscale_on(False)
# #      sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
# #      # Plot for each segment area
# #      for val in sorted_result:
# #         mask = val['segmentation']
# #         img = np.ones((mask.shape[0], mask.shape[1], 3))
# #         color_mask = np.random.random((1, 3)).tolist()[0]
# #         for i in range(3):
# #             img[:,:,i] = color_mask[i]
# #             ax.imshow(np.dstack((img, mask*0.5)))
# #             plt.show()

# # _,axes = plt.subplots(1,2, figsize=(16,16))
# # axes[0].imshow(rgb_image)
# # show_output(sam2_result, axes[1])

