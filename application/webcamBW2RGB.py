import cv2
import numpy as np
import torch
from torchvision.transforms import *

from GrayNet import *

IMAGESIZE = 512

@torch.no_grad()
def processFrame(model, cam, device):
    # Capture image from camera.
    ret, frame = cam.read()

    # Test if image was successfully captured. otherwise raise exception.
    if not ret:
        cam.release()
        cv2.destroyAllWindows()
        raise Exception("uh huh huh no camera frame")
    
    # Resize captured image to desired size.
    frameOriginal = cv2.resize(frame, (IMAGESIZE,IMAGESIZE))
    
    # Convert captured image to grayscale.
    frameBlackWhite = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2GRAY)
    
    # Convert captured image to tensor.
    cv2tensor = transforms.ToTensor()
    inp = cv2tensor(frameBlackWhite).to(device).reshape(1,1,IMAGESIZE,IMAGESIZE)
    inp = torch.cat((inp,inp,inp),1)

    # Feed captured image to neural network.
    output = model(inp)

    # Resize neural network output and convert to numpy image.
    tmp = output.reshape(3,IMAGESIZE,IMAGESIZE).cpu().numpy()
    tmp = np.transpose(tmp, (1,2,0))

    # Convert numpy image to normal color image.
    frameOutput = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    # Display images at various stages.
    cv2.imshow("original", frameOriginal)
    cv2.imshow("BW", frameBlackWhite)
    cv2.imshow("output", frameOutput)

# Test if GPU is available.
print("Testing if GPU is available.")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, Falling back to CPU.")


# Declare camera object.
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create Neural Network and load weights from file.
model = GrayNet().to(device)
loadPath = "largeimagetrainedweights.pt"#"480pweights2.pt"#f"finaltrainedweights.pt"
model.load_state_dict(torch.load(loadPath))

# Main loop.
while True:
    # Capture, process, and display images from webcam.
    processFrame(model, cam, device)

    # Look for escape key hit.
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

# Release resources.           
cam.release()
cv2.destroyAllWindows()

