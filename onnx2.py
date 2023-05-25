import onnxruntime
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
import torchvision.models as models


class MobileNetV3:
    def __init__(self):
        
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = models.mobilenet_v3_small().to("cpu")
        checkpoint = torch.load('C:\\Users\\Arpit Mohanty\\mobnet\\data\\output\\checkpoint-best.pth')
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        
        

        self.model.load_state_dict(state_dict=state_dict)

        self.model.eval()

        # Convert the PyTorch model to ONNX
        self.to_onnx('mobilenet_v3_small.onnx')
        onnx_file_size = os.path.getsize("mobilenet_v3_small.onnx")
        onnx_file_size_mb = onnx_file_size / (1024 * 1024)
        print(f"ONNX model file size: {onnx_file_size_mb} MB")


        # Load ONNX model
        self.ort_session = onnxruntime.InferenceSession('mobilenet_v3_small.onnx')

    def to_onnx(self, filename):
        # Create example input tensor
        input_shape = (1, 3, 480, 480)
        example_input = torch.randn(input_shape)
        

        # Convert PyTorch model to ONNX format
        torch.onnx.export(self.model, example_input, filename,
                          opset_version=11, do_constant_folding=True,
                          input_names=['input'], output_names=['output'])

    def infer(self,test_folder_path):
        test_dir = "C:\\Users\\Arpit Mohanty\\mobilenetv3\\data\\test"
        empty_dir = os.path.join(test_dir, 'empty')
        filled_dir = os.path.join(test_dir, 'filled')
        empty_imgs = [os.path.join(empty_dir, img) for img in os.listdir(empty_dir)]
        filled_imgs = [os.path.join(filled_dir, img) for img in os.listdir(filled_dir)]
        all_imgs = empty_imgs + filled_imgs

        # Preprocess image
        results = []
        count=0
        correct_predictions=0
        timer=0
        for folder_name in ['empty', 'filled']:
            folder_path = os.path.join(test_folder_path, folder_name)
            for filename in os.listdir(folder_path):
                test_image_path = os.path.join(folder_path, filename)
                count+=1
            # Preprocess image
                start_time = time.time()
                image = Image.open(test_image_path)
                image1 = self.transform(image).unsqueeze(0).to("cpu")
                image1 = image1.to("cpu")


                # Run inference on ONNX model
                ort_inputs = {self.ort_session.get_inputs()[0].name: image1.cpu().numpy().astype(np.float32)}
                ort_outputs = self.ort_session.run([self.ort_session.get_outputs()[0].name], ort_inputs)
                output_tensor = torch.from_numpy(ort_outputs[0]).to("cpu")


                # Get predicted class label
                _, pred_class = torch.max(output_tensor, 1)
                end_time = time.time()
                timer+=end_time-start_time
                true_class = 1 if folder_name == 'filled' else 0

                results.append(pred_class.item())
                if pred_class.item() == true_class:
                        correct_predictions += 1


                

        inference_time = (timer) / len(all_imgs)

       
        accuracy = correct_predictions /603

        return results, accuracy, inference_time
model = MobileNetV3()



results,accuracy,inf_time = model.infer("C:\\Users\\Arpit Mohanty\\mobilenetv3\\data\\test")
print(f'Accuracy of the model is {accuracy *100}%')
print(f"Inference time per image is {inf_time}")
print(f"Total Testing Time is {inf_time*len(results)}")



