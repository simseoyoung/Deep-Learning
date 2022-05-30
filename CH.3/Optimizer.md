# Optimzer

: Train data set을 이용해 모델을 학습 할 때 데이터의 실제 결과와 모델이 예측한 결과를 기반으로 오류를 잘 줄일 수 있게 만들어주는 역할을 함


## Optimizer의 종류

1) SGD <br>
    : full-batch가 아닌 mini batch로 학습을 진행하는 것 <br>
    ->  full-batch로 epoch마다 weight를 수정하지 않고 빠르게 mini-batch로 weight를 수정하면서 학습 가능 <br>
  +) Momentum <br>
  : SGD의 높은 편차를 줄이고 수렴을 부드럽게 하기 위해 고안된 optimizer로 관련된 방향으로의 수렴을 가속화시키고 관련이 없는 방향으로의 변동을 막음    

2) AdaGrad <br>
    :학습을 통해 크게 변동이 있었던 가중치에 대해서는 학습률을 감소시키고 학습을 통해 가중치의 변동이 별로 없었던 가중치는 학습률을 증가시켜서 학습시킴
    * SGD의 개념에서 h(가중치 기울기 제곱들을 더해감)를 추가하여 가중치에 따라 학습률을 다르게 함 <br>
    -> 학습을 무한히 하다보면 학습이 아예 안될 수도 있음 (RMS에서 개선)

3) RMSprop <br>
    :가중치 기울기를 단순 누적시키는 것이 아니라 최신 기울기들이 더 반영되도록 하여 무한히 학습기키면 학습이 아예 안되는 AdaGrad의 단점을 보완함 <br>
    -> hyper parameter p를 추가하여 h가 무한히 커지지 않게 함

4) Adam <br>
    : Momentum과 RMSProp를 융합한 방법 (각 파라미터마다 다른 크기의 업데이트를 진행) <br>
    *가장 많이 사용되는 방법

