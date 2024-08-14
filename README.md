# CORS-based Classification and Detection of Camouflaged Soldiers
Classification and Detection of Camouflaged Soldiers Using CORS-Based Ensemble Models with Augmented Data

## Goal
- 300장의 위장 군인 데이터로 1500장의 합성 데이터 생성
- 생성 데이터를 포함한 데이터셋으로 Classification ensemble model을 학습
- Classification과 Object Detection의 상호보완(CORS)을 제안
- 제안한 알고리즘으로 위장 군인 탐지 및 식별


### Class Image
<p align="center">
  <img src="./class_dir/105.png" alt="image1" width="200"/>
  <img src="./class_dir/106.png" alt="image2" width="200"/>
  <img src="./class_dir/109.png" alt="image3" width="200"/>
  <img src="./class_dir/111.png" alt="image4" width="200"/>
</p>

### Data Image
<p align="center">
  <img src="./data_dir/train1/image183.jpg" alt="image1" width="200"/>
  <img src="./data_dir/train1/image28.jpg" alt="image2" width="200"/>
  <img src="./data_dir/train1/image49.jpg" alt="image3" width="200"/>
  <img src="./data_dir/train1/image54.jpg" alt="image4" width="200"/>
</p>

### Sampling Image
<p align="center">
  <img src="./sampling/sample_2_97.png" alt="image1" width="200"/>
  <img src="./sampling/sample_5_109.png" alt="image2" width="200"/>
  <img src="./sampling/sample_5_57.png" alt="image3" width="200"/>
  <img src="./sampling/sample_6_109.png" alt="image4" width="200"/>
</p>


### Sampling with rg blindness
<p align="center">
  <img src="./sampling_with_rg_blindness/sample_8_102.png" alt="image1" width="200"/>
  <img src="./sampling_with_rg_blindness/sample_8_107.png" alt="image2" width="200"/>
  <img src="./sampling_with_rg_blindness/sample_8_46.png" alt="image3" width="200"/>
  <img src="./sampling_with_rg_blindness/sample_8_48.png" alt="image4" width="200"/>
</p>

### Result loss and accuracy
| ![image1](./result/Train_Loss_with_bagging.png) | ![image2](./result/Train_Accuracy_with_bagging.png) |
|:----------------------------------------------:|:--------------------------------------------------:|
| Train Loss                                      | Train Accuracy                                     |

| ![image3](./result/Validation_Loss_with_bagging.png) | ![image4](./result/Validation_Accuracy_with_bagging.png) |
|:----------------------------------------------------:|:---------------------------------------------------------:|
| Validation Loss                                      | Validation Accuracy                                        |

<img src="./result/Test_Accuracy.png" alt="image5"/>


