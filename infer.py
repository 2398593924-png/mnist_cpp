import torch
import torch.nn as nn

import numpy as np
import cv2
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim = 0)
        x = self.net(x)

        return x
    
# 全局状态
drawing = False
prev_pt = None
color = (255, 255, 255)
thickness = 10

def mouse_callback(event, x, y, flags, param):
    global drawing, prev_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, prev_pt, (x, y), color, thickness)
            prev_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_pt = None

def showpred(pred):
    img = np.zeros((400, 450), dtype=np.uint8)
    count = 0
    for i in range(20, 400, 40):
        cv2.putText(img, f"{count}", (i, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.line(img, (i-1, 320), (i+13, 320), (255, 255, 255), 2)
        cv2.rectangle(img, (i-1, int(320 - 270*pred[count])), (i+13, 320), (255, 255, 255), -1)
        count += 1
    
    cv2.imshow('2', img)

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load("./weights/mnist_9.pth", map_location="cpu"))
    model.eval()

    canvas = np.zeros((300, 300))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        cv2.imshow("image", canvas)
        key = cv2.waitKey(1) & 0xFF

        image = cv2.resize(canvas, (28, 28))
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        pred = model(image)

        pred = pred.detach()
        pred = torch.softmax(pred, dim = 0)

        pred = pred.numpy()

        showpred(pred)

        if key == ord('s'):
            canvas[:, :] = 0
