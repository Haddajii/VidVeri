import torch 
import torchvision.transforms as transforms 
from torchvision import models
from PIL import Image
import os 
import urllib.request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = urllib.request.urlopen(url)
labels = [line.strip() for line in response.readlines()]

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(f"{labels[top5_catid[i]]}: {top5_prob[i].item() * 100:.2f}%")


if __name__ == "__main__":
    test_image = "output_frames/frame_0.jpg"
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print(f"Image not found: {test_image}")
