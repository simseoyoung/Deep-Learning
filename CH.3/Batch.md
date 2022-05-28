# Batch의 정의
모델을 학습할 때 한 iteration(반복) 당 사용되는 example의 set의 모임

- 한 번의 epoch를 위해 여러 번의 iteration 필요

## Batch size를 선택하는 방법

1) Batch
  : 여러 개의 sample들이 한번에 영향을 주기 때문에 smooth하게 수렴하지만 sample의 개수 전체를 계산해야하므로 시간이 많이 소요됨 

  (한 step을 처리하는 데에 모든 데이터를 계산해야함 -> 모든 training data set 사용)

  *실질적으로는 메모리에 모든 데이터를 한 번에 올리기 어렵기 때문에 잘 사용 못함

2) Stochastic
  : 데이터를 하나씩 뽑아 처리해보고 이 과정을 모든 데이터에 반복하는 과정으로 수렴속도는 빠를 수 있으나 오차율이 큼

3) Mini Batch
  : 전체 학습 데이터를 배치 사이즈로 등분하여 각 배치 set을 순차적으로 진행기 떄문에 batch보다 빠르고 SGD보다 오차가 적음 

=> Mini batch의 크기가 전체 training data의 크기와 같다면 Batch Gradient Descent과 같고, Mini batch의 크기가 1이면 Stochastic Gradient Descent와 같음
