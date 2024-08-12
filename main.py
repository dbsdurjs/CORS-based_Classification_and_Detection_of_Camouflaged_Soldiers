from train_model import training
from test import *
from predict_image import *
from yolo_excute import *

if __name__ == "__main__":
    if torch.cuda.is_available():
        print('cuda is available. working on gpu')
        device = torch.device('cuda')
    else:
        print('cuda is not available. working on gpu')
        device = torch.device('cpu')

    # 데이터 셋 생성 및 모델 훈련, 평가
    #num_epochs = training(device)

    # 테스트
    #testing(device, num_epochs=num_epochs)

    # 특정 이미지 테스트
    predict_img_path = '/home/yoonyeogeon/test13.jpeg'
    is_soldier, prob = predicting(predict_img_path, device)

    conf = 0.8
    while True:
        # yolo detection
        box_prob = detecting(predict_img_path, conf=conf)

        print(f'is soldier {is_soldier} | prob {prob:.2f} | box_prob {box_prob} | conf {conf}')

        if (box_prob is None and is_soldier == False) or (box_prob and is_soldier == True): # 둘 다 True, False
            break

        if (is_soldier == False and box_prob):  # classification 인식 못함, detection 인식함
            prob = round(prob + 0.1, 2)
            break

        # classification 인식함, detection 인식 못함
        if conf > 0.1:
            conf = round(conf - 0.1, 2)
        elif conf <= 0.01:
            break
        else:
            conf = round(conf - 0.01, 2)

    print(f'classifier : {is_soldier}')
    print(f'classified with {prob:.2f} and detected with {box_prob}')
