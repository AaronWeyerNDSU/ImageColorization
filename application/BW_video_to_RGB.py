import cv2
import numpy as np
import torch
from torchvision.transforms import *
from tqdm.auto import tqdm

from GrayNet import *

@torch.no_grad()
def processFrame(model, frame, device, videoSize, showFrames=False):   
    # Resize input frame to the desired output size.
    inp = np.clip(cv2.resize(frame, videoSize),0,255)

    # Convert input frame to tensor.
    cv2tensor = transforms.ToTensor()
    inp = cv2tensor(inp).to(device).reshape(1,3,videoSize[1],videoSize[0])

    # Run input frame through neural network.
    output = model(inp)

    # Convert neural network output to numpy image.
    tmp = output.reshape(3,videoSize[1],videoSize[0]).cpu().numpy()
    tmp = np.transpose(tmp, (1,2,0))

    # Convert numpy image to cv2 image.
    frameOutput = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    # Display frames as they are processed if desired.
    if showFrames:
        cv2.imshow("original", frame)
        cv2.imshow("output", frameOutput)
        cv2.waitKey(1)

    # Return processed frame to write to file.
    return np.clip((frameOutput*255).astype(np.uint8), 0, 255)

# Test if GPU is available.
print("Testing if GPU is available.")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, Falling back to CPU.")

# Load input Black and white video to cv2 object.
input_video_path = 'AndygriffithLake.mp4'#'Depeche Mode - Wagging Tongue (Official Video).mp4'#
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open the video file.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
VIDEOSIZE = (256,256)
#VIDEOSIZE = (width, height)

# Setup video output.
output_video_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, VIDEOSIZE)
if not out.isOpened():
    print("Error: Could not open the output video file.")


# Create Neural Network and load weights from file.
model = GrayNet().to(device)
loadPath = "480pweights3.pt"#"largeimagetrainedweights.pt"#"480pweights2.pt"#f"finaltrainedweights.pt"
model.load_state_dict(torch.load(loadPath))

# Iterate over each frame
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=totalFrames, desc='Processing Frames')
while True:
    # get next frame from video.
    ret, frame = cap.read()

    # Check if end of file has been reached, if so, exit.
    if not ret:
        print("End of video.")
        break
    
    # Process frame and write it to the output video file.
    out.write(processFrame(model, frame, device, VIDEOSIZE))
    
    # Update the progress bar.
    progress_bar.update(1)

# Release all resources
cap.release()
out.release()
cv2.destroyAllWindows()
progress_bar.close()
