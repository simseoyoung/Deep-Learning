# Hyperparameter Optimization

1) Manual Search <br>
 : 사용자가 경험적으로 괜찮다고 생각하는 hyperparameter를 선정하여 tuning <br>
  -> 경험이 부족하다면 사용하기 어렵고, 사용자의 선택에 의해 결정되므로 시간이 많이 소요될 수 있음 <br>
  
2) Grid Search <br>
: 각각의 hyperparmeter 구간을 정해놓고 일정 구간으로 잘라 모든 조합을 search함

3) Random Search <br>
: random하게 hyperparameter를 뽑아 search함 <br>
-> 구간이 적으면 성능이 좋으나 구간이 넓으면 성능이 저하됨

4) Bayesian Optimization <br>
: 가장 많이 사용되는 방법으로 실제로 훈련을 시키고 학습을 시켜 최적의 hyperparameter를 찾아감
