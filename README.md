# CORS-based Classification and Detection of Camouflaged Soldiers
Classification and Detection of Camouflaged Soldiers Using CORS-Based Ensemble Models with Augmented Data

## Goal
- 300장의 위장 군인 데이터로 1500장의 합성 데이터 생성
- 생성 데이터를 포함한 데이터셋으로 Classification ensemble model을 학습
- Classification과 Object Detection의 상호보완(CORS)을 제안
- 제안한 알고리즘으로 위장 군인 탐지 및 식별

## Problem & Solution
- 현실에서는 국방 데이터가 많이 부족
- 생성 모델을 통해서 부족한 데이터에 대한 문제를 해결
- 적은 데이터로 많은 데이터를 만들 수 있는 모델 탐색


## Task
- 약 300장의 실제 이미지로 약 1500장의 생성 이미지 생성
- 생성 이미지 절반은 적록 색맹 적용(750장)
- 생성된 이미지로 Classification ensemble model 학습
- 제안한 __상호보완 객체 인식 시스템(Complementary Object Recognition System, CORS)__으로 결과 향상

## Utilization Strategies
- 비무장 지대 혹은 전쟁 중 산악 지형에서 유용하게 사용될 수 있을 것
- 경계 감시 군인들의 피로를 줄일 수 있음
- 전차 카메라에 활용하여 매복병 탐지
- 다른 위장 및 매복 데이터 셋에서 활용 가능


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


