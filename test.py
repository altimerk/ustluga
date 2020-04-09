from collections import deque
from PIL import Image
import cv2
import torch
from torchvision import transforms, models

cap = cv2.VideoCapture("d:\\ustluga\\video\\short.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = torch.load("ustluga_cam6_resnet18_1346-04092020.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x1 = 640
x2 = x1 + 448
y1 = 360
y2 = y1 + 448
start_point = (x1, y1)
end_point = (x2, y2)
color = (255, 255, 0)
thickness = 5
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
font = cv2.FONT_HERSHEY_SIMPLEX
org = (150, 50)
org1 = (150, 150)
fontScale = 2
green = (0, 255, 0)
red = (0, 0, 255)
fn = 0
d = deque([],5)
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
while(True):
    ret, frame = cap.read()
    if not ret:
        break
    crop = frame[y1:y2, x1:x2]
    pil_im = Image.fromarray(crop)

    inp = val_transforms(pil_im).view(1, 3, 224, 224)
    inp = inp.to(device)
    preds = model(inp)
    probability = torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy().item()
    d.appendleft(probability)
    moving_average = sum(d) / len(d)
    if(moving_average>0.5):
        cv2.putText(frame,"wheels out of stend", org, font,
                   fontScale, red, thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, "wheels on stend", org, font,
                   fontScale, green, thickness, cv2.LINE_AA)
    cv2.putText(frame, "out of stend probability: {:.2f}".format(moving_average), org1, font,
                fontScale, (255,255,255), thickness, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if  key & 0xFF == ord('s'):
        cv2.imwrite("frame{}.png".format(fn),crop)
    if  key & 0xFF == ord('q'):
        break
    fn +=1
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()