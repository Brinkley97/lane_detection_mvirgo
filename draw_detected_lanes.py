import os
import cv2
import numpy as np


# from scipy.misc import imresize
from IPython.display import HTML
from keras.models import load_model
from skimage.transform import resize
from moviepy.editor import VideoFileClip


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def arithm_op(arr1, arr2, operation, dtype=None):
    if dtype is None:
        dtype = np.result_type(arr1, arr2)
    
    arr1 = arr1.astype(dtype)
    arr2 = arr2.astype(dtype)
    
    if operation == 'add':
        return np.add(arr1, arr2)
    elif operation == 'subtract':
        return np.subtract(arr1, arr2)
    elif operation == 'multiply':
        return np.multiply(arr1, arr2)
    elif operation == 'divide':
        return np.divide(arr1, arr2)
    else:
        raise ValueError("Unsupported operation")

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = resize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = resize(lane_drawn, (720, 1280, 3))

    # Ensure both images have the same type
    lane_image = lane_image.astype(image.dtype)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    # Use arithm_op to set the final result
    final = arithm_op(result, result, 'add', dtype=result.dtype)

    return final

def save_video():
    
    find_videos = os.listdir("videos_to_process")
    print("find_videos: ", find_videos)

    for video in find_videos:

        video_dir = "videos_to_process/" + video
        print("video_dir: ", video_dir)
        
        # Location of the input video
        clip1 = VideoFileClip(video_dir)
        # print("clip1 type : ", type(clip1), clip1, "\n")

        # Ensure fps is a real number
        fps = clip1.fps if clip1.fps is not None else 24  # Set a default fps if clip1.fps is None
    
        # Create the clip
        vid_clip = clip1.fl_image(road_lines)
        
        # # Where to save the output video
        save_video_as = "output/" + video
        final_video = vid_clip.write_videofile(save_video_as, fps=fps, audio=False)

    return final_video

if __name__ == '__main__':
    # Load Keras model
    model = load_model('full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()

    save_video()
