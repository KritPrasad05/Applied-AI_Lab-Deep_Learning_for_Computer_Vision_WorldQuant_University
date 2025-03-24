# **AI Lab: Deep Learning for Computer Vision**
# **WorldQuant University**
#
#

# **Usage Guidelines**
#
# This file is licensed under Creative Commons Attribution-NonCommercial-
# NoDerivatives 4.0 International.
#
# You **can** :
#
#   * ✓ Download this file
#   * ✓ Post this file in public repositories
#
# You **must always** :
#
#   * ✓ Give credit to WorldQuant University for the creation of this file
#   * ✓ Provide a link to the license
#
# You **cannot** :
#
#   * ✗ Create derivatives or adaptations of this file
#   * ✗ Use this file for commercial purposes
#
# Failure to follow these guidelines is a violation of your terms of service and
# could lead to your expulsion from WorldQuant University and the revocation
# your certificate.
#
#

# Import needed libraries
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load MTCNN, Resnet, and the embedding data

mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)

resnet = InceptionResnetV1(pretrained="vggface2")
embedding_data = torch.load("embeddings.pt")

resnet = resnet.eval()


# Fill in the locate_face function
def locate_faces(image):
    cropped_images, probs = mtcnn(image,return_prob=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))    


# Fill in the determine_name_dist function
def determine_name_dist(cropped_image, threshold=0.9):
    # Use `resnet` on `cropped_image` to get the embedding.
    # Don't forget to unsqueeze!
    emb = resnet(cropped_image.unsqueeze(0))

    # We'll compute the distance to each known embedding
    distances = []
    for known_emb, name in embedding_data:
        # Use torch.dist to compute the distance between
        # `emb` and the known embedding `known_emb`
        dist = torch.dist(emb, known_emb)
        distances.append((dist, name))

    # Find the name corresponding to the smallest distance
    dist, closest = min(distances)

    # If the distance is less than the threshold, set name to closest
    # otherwise set name to "Undetected"
    if dist < threshold:
        name= closest
    else:
        name="Undetected"

    return name, dist   


# Fill in the label_face function
def label_face(name, dist, box, axis):
    """Adds a box and a label to the axis from matplotlib
    - name and dist are combined to make a label
    - box is the four corners of the bounding box for the face
    - axis is the return from fig.subplots()
    Call this in the same cell as the figure is created"""

    # Add the code to generate a Rectangle for the bounding box
    # set the color to "blue" and fill to False
    rect = plt.Rectangle(
        (box[0],box[1]),
        box[2]-box[0],
        box[3]-box[1],
        fill = False,
        color="blue",
    )
    axis.add_patch(rect)

    # Set color to be red if the name is "Undetected"
    # otherwise set it to be blue
    if name == "Undetected":
        color = "red"
    else:
        color = "blue"
    
    label = f"{name} {dist:.2f}"
    axis.text(box[0], box[1], label, fontsize="large", color=color) 


# Fill in the add_labels_to_image function
def add_labels_to_image(image):
    # This sets the image size
    # and draws the original image
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    # Use the function locate_faces to get the individual face info
    faces = locate_faces(image)

    for box, prob, cropped in faces:
        # If the probability is less than 0.90,
        # It's not a face, skip this run of the loop with continue
        if prob < 0.90:
            continue        
        # Call determine_name_dist to get the name and distance
        name, dist = determine_name_dist(cropped)

        # Use label_face to draw the box and label on this face
        label_face(name, dist, box, axis)

    return fig


# This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.
