import torch

from model import *
from torchvision import transforms
from PIL import Image
from convert_rg_blindness import *
from pretrained_model import *

#모델 로드
def load_ensemble_models(ensemble_model, device):
    #models = [model1(), model2(), model3(), model4(), model5()]
    models = ensemble_model
    checkpoint_dir = './loss_and_accuracy/0726 - 2 pretrained model synthesis + real data(modified, mean) - best'
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model_{i+1}.pt')))
        model.to(device)
        model.eval()
    return models


#이미지 예측
def predict_image(image_path, models, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = [model(image) for model in models]
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        probability = torch.sigmoid(ensemble_output)
        predicted = probability > 0.5

    return predicted.item(), probability.item()

def predicting(img_path, device):
    # 모델 로드 및 예측 예시
    loaded_models = load_ensemble_models(generate_model(), device)

    # 예측을 위한 transform (학습 시 사용한 것과 동일해야 함)
    predict_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(size=(240, 240)),
        transforms.RandomResizedCrop(240, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 예측 실행
    is_soldier, prob = predict_image(img_path, loaded_models, predict_transform, device)

    if not is_soldier:
        prob = 1 - prob

    return is_soldier, prob