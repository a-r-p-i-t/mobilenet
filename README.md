## MOBILENETV3
This repository contains an implementation of MobileNetV3, a lightweight deep learning architecture for mobile and embedded devices. The MobileNetV3 model achieves high accuracy with low computational cost, making it suitable for real-time applications on resource-constrained devices.
## Installation
1. Clone the repository:
 git clone https://github.com/a-r-p-i-t/mobilenet
2. Install the required dependencies:

   pip install -r requirements.txt
## Usage
Load a pretrained MobileNetV3:

from torchvision.models import mobilenet_v3_small,mobilenet_v3_large

model = mobilenet_v3_small(pretrained=True)

Data Loading:

The folder structure looks as :
Main Directory
- Train
  - Sub1
    - img1
    - img2
  - Sub2
    - img3
    - img4
- Val
  - Sub1
    - img5
    - img6
  - Sub2
    - img7
    - img8


            
   
root = "path/of/root/directory"

train_folder = os.path.join(root, "train")

val_folder = os.path.join(root, "val")

Training Parameters:Model,Batch Size,num_epochs,weight_decay,learning_rate,data_loader

Validation Parameters: Model,Batch Size,num_epochs,data_loader


Run:

python main.py

## Inference Results

# Without ONNX Export:

Model size: 29.35 MB

Accuracy: 0.9767827529021559

Average inference time per image: 0.0097 seconds

Throughput: 103.01 images per second

Total testing time is 5.853941440582275 



# With ONNX Export:

ONNX model file size: 9.713944435119629 MB

Accuracy of the model is 96.01990049751244%

Inference time per image is 0.010009029809119887

Total Testing Time is 6.035444974899292






