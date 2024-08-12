from model import *
from convert_rg_blindness import *
from pretrained_model import *
import csv
import pandas as pd
from show_graph import show_graphs
from dataset import make_dataset

#모델 로드
def load_ensemble_models(ensemble_model, device):
    models = ensemble_model
    checkpoint_dir = 'loss_and_accuracy/0726 - 2 pretrained model synthesis + real data(modified, mean) - best'
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model_{i+1}.pt')))
        model.to(device)
        model.eval()
    return models

#이미지 예측
def predict_image(test_loader, models, num_epochs, device):
    ensemble_test_accuracy = []
    ensemble_test_losses = []
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(num_epochs):

        test_total = 0
        test_correct = 0
        test_loss = 0.0

        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                targets = labels.float().unsqueeze(1)

                ensemble_output = torch.stack([model(inputs) for model in models], dim=0)
                ensemble_output = torch.mean(ensemble_output, dim=0)

                loss = criterion(ensemble_output, targets)
                test_loss += loss.item()

                predicted = (torch.sigmoid(ensemble_output) > 0.5).float()
                test_total += len(targets)
                test_correct += (predicted == targets).float().sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_loss /= len(test_loader)  # 평균 손실 계산

        ensemble_test_accuracy.append(test_accuracy)
        ensemble_test_losses.append(test_loss)

        print(f'test accuracy : {test_accuracy:.2f}')
        print(f'test loss : {test_loss:.4f}')

    return ensemble_test_accuracy, ensemble_test_losses

def testing(device, num_epochs=20):
    non_soldier_path = "./Landscape Classification/Landscape Classification/Validation Data"
    real_soldier_path = "./camouflage_soldier_dataset/Testing"
    train_root_path = "./test_dataset"

    n_target_path = './test_dataset/non_soldier'
    s_target_path = './test_dataset/camouflage_soldier'

    path = [train_root_path, non_soldier_path, real_soldier_path, n_target_path, s_target_path]

    test_loader, _ = make_dataset(path, method="test")

    # 모델 로드 및 예측
    loaded_models = load_ensemble_models(generate_model(), device)
    ensemble_test_accuracy, ensemble_test_loss = predict_image(test_loader, loaded_models, num_epochs, device)

    max_accuracy = max(ensemble_test_accuracy)
    max_epoch = ensemble_test_accuracy.index(max_accuracy)

    # 80~100 범위의 그래프 그리기
    truncated_data = ensemble_test_accuracy[:max_epoch+1]

    filename = './loss_and_accuracy/0726 - 2 pretrained model synthesis + real data(modified, mean) - best/test accuracy.csv'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Accuracy'])  # 헤더 작성
        for epoch, accuracy in enumerate(truncated_data):
            csvwriter.writerow([epoch, accuracy])

    print(f"Data has been saved to {filename}")
    # df = pd.read_csv(filename)

    # # 그래프 그리기
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Epoch'], df['Accuracy'])
    # plt.title('Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.ylim(80, 100)
    # max_value = max(df['Accuracy'])
    # max_index = df['Accuracy'].idxmax()

    # 최대값에 텍스트 추가
    # plt.text(df['Epoch'][max_index], max_value, f'Max: {max_value:.2f}', ha='center', color='red', va='bottom')
    #
    # plt.grid(True)
    # plt.show()

    metrics = [
        ('Test Loss', ensemble_test_loss),
        ('Test Accuracy', ensemble_test_accuracy)
    ]
    show_graphs(metrics, num_epochs)
