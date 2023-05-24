# MOBILENETV3
This repository contains an implementation of MobileNetV3, a lightweight deep learning architecture for mobile and embedded devices. The MobileNetV3 model achieves high accuracy with low computational cost, making it suitable for real-time applications on resource-constrained devices.
## Installation
1. Clone the repository:
 git clone https://github.com/xiaolai-sqlai/mobilenetv3.git
2. Install the required dependencies:
pip install -r requirements.txt
# Usage
Load a pretrained MobileNetV3:

from mobilenetv3 import MobileNetV3_Small,MobileNetV3_Large
model = MobileNetV3_Small()

