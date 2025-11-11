from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

pred = ImageClassification()
pred.setModelTypeAsMobileNetV2()
pred.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth')) # Sets the model path combined with the current directory
pred.loadModel() # Loads the model, that is already pre-built


preds, probs = pred.classifyImage(os.path.join(exec_path, 'house.jpg'), result_count = 5)
for each_pred, each_prob in zip(preds, probs):
    print(f'{each_pred}: {each_prob}')