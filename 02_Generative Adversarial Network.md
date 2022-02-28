# **Generative Adversarial Network**
> 적대적 생성 모델
> https://arxiv.org/abs/1406.2661 

<br/>
<br/>


## **0. Abstract**
- Generative model과 Discriminative model 총 2개의 모델 학습.
- "경쟁" 과정을 통해 Generative model을 추정하는 새로운 프레임워크 제안.
- Neural Network로 이루어진 생성자(Generator)와 판별자(Discriminator)가 서로 겨루며 훈련. <br/>


<p align="center"><img src=https://media.vlpt.us/images/tobigs-gm1/post/b6751877-1293-4be7-b2b1-c31e4d013000/image.png width="500px"></p>
<center> [그림.1] </center> <br/>


- Generative model and Discriminative model
  - **Generative model(생성 모델, G)** <br/>
    : Discriminative model이 구별하지 못하도록 Training data의 분포를 모사함.<br/>
    : Unsupervied Learning(비지도 학습)
  
  - **Discriminative model(판별 모델, D)** <br/>
    : Sample data가 G(Generative model)로부터 나온 데이터가 아닌, 실제 Training data로부터 나온 데이터일 확률 추정.<br/>
    : Supervied Learning(지도 학습)
  
  - Generative model을 학습하는 과정은 Discriminative model이
  Smaple 데이터가 G로부터 나온 가짜 데이터와 실제 Training 데이터를 잘 못 판별할 확률을 최대화 하는 것. <br/>

<p align="center"><img src=https://t1.daumcdn.net/cfile/tistory/9928E6375B75872D17 width="500px"></p>
<center> [그림.2] </center> <br/>


- 이 논문에서는 GAN이라는 새로운 프레임워크를 제안했다.
  - generative model과 Discriminative model 두가지 모델을 학습하며,
  - G는 실제 Trianing data의 분포를 모사 하며 그와 비슷한 데이터를 생성 하고자 한다.
  - D는 실제 데이터와 G가 생성한 데이터를 정확하게 구별하고자 한다.

<br/>
<br/>

## **1. Introduction**
- Generative model과 Discriminative model 프레임을 새로이 제안한 것이 아님 유의.<br/> 
  위 두 모델은 기존에 활발히 개발되고 있었으며, <br/>
  GAN의 저자는 *경쟁구도를 제안하여 두 모델이 모두 각각의 목적을 달성시키기 위하여 스스로를 개선하도록 이끄는 프레임*을 제안한 것.

- Discriminative model(판별 모델)
  - 지금까지(~2014) 딥러닝에서 가장 훌륭한 성공은 고차원의 풍부한 입력 데이터를 범주형 class label로 매핑하는 discriminatice model(판별 모델)이었다.
  - 위 성공은 주로 역전파, 드롭아웃, 구간별 선형 활성화 유닛(piecewise linear unints)등에 기반을 두고 있다.
    > **Piecewise linear ?** <br/>
    > - 비선형 회귀에 주로 이용.
    > - 부분적 선형. 
    >> https://ichi.pro/ko/jogag-byeol-seonhyeong-hoegwi-model-geugeos-eun-mueos-imyeo-eonje-sayonghal-su-issseubnikka-161801836618834
  - classifier, regressor etc.

<br/>

- Deep Generative Model(심층 생성모델)
  - 생성모델은 최대 가능도 추정에 있어서 수많은 다루기 힘든 확률적 계산을 근사하는 것에 어려움이 있었다.
    > **최대 가능도(Maximum Likeihood)** <br/>
    > : 셀 수 있는 사건에서는 특정 사건이 발생한 확률이고, 셀수 없는 연속적인 사건의 경우에는 특정 사건 영역이 발생할 확률.
    >> https://rpubs.com/Statdoc/204928
  - piecewise linear unint의 이점을 생성적 관점에서 잘 이용하는 데에 어려움이 있었다.
  - 이러한 이유로 생성모델은 그동안 판별모델에 비해 적은 영향력(발전)을 보여주고 있었다.
  - 그래서 *저자들은 위와 같은 어려움을 회피하는 새로운 생성모델 추정절차를 제안.*
    > 제안된 적대적 신경망(adversarial nets) 프레임워크에서 생성모델은 상대편(판별모델)을 속이도록 세팅되고 판별모델 D은 어떤 샘플이 생성모델 G가 모델링한 분포로부터 나온것인지 실제 데이터 분포로부터 나온것인지를 결정하는법을 배운다. 이러한 경쟁구도는 두 모델이 모두 각각의 목적을 달성시키기 위해 스스로를 개선하도록 이끈다. (G는 D를 더 잘 속이도록 원본 데이터를 더 잘 모방한 분포를 학습하게되며, D는 진짜/가짜 데이터를 더 잘 간파하도록 데이터의 특징을 더잘 파악하도록 학습된다.)
- 적대적 프레임워크는 다양한 모델, 최적화 알고리즘들을 위한 훈련알고리즘을 유도하게 확장될 수 있다.
- 본 연구는 이 적대적 프레임워크를 이용한 하나의 case로써 MLP(다층퍼셉트론)로 구성된 생성모델에 Random noise를 흘려보내는 것으로 데이터를 생성하며, 판별을 위한 모델또한 MLP로 구성된다.
- 이 case를 본 연구에서 adversarial nets이라고 명명.

<br/>
<br/>

## **2. Related work**
- 본 연구 이전까지 심층생성모델의 연구는 주로 확률분포함수의 parametric specification을 발견하는 것에 집중되어왔다. 그렇게 분포함수가 구체화 되면 최대가능도 추정을 하는식으로 모델이 얻어질 수 있다(such as restricted Boltzmann machines (RBMs), deep Boltzmann machines(DBMs)). 이런 모델은 일반적으로 다루기 힘든 가능도 함수를 가지고 있기 때문에 gradient에 대한 수치적인 근사가 필요하다.
  > Restricted Boltzmann machines (RBMs)은 심층신뢰신경망의 기본 뼈대가 된다.  
  > 1] (Boltzmann machines) https://nlp.jbnu.ac.kr/AI2019/slides/ch05-3.pdf <br/>
  > 2] (Deep belief network ) http://tcpschool.com/deep2018/deep2018_deeplearning_algorithm

<br/>

- 몇몇의 테크닉은 확률 분포를 명식적으로 정의하지 않고, Generative model을 특정 분포의 데이터를 이용해 학습을 시킨 후에 원하는 분포의 데이터를 뽑아내게 한다. 이러한 접근법은 Model들이 Back-ropagation에 의해 학습되는 것을 기본으로 설계되었다는 점에서 이점이 있다. <br/>
  대표적으로 Generative Stochastic Network(GSN)이 있다. GSN은 일반화 된 잡음을 제거하는 denoising auto-encoder들의 확장이며, 매개변수화 된 Markov chain을 정의한다.
  > (GSN Paper) https://arxiv.org/abs/1503.05571 //29p <br/>
  > (Auto encoder) https://hyen4110.tistory.com/37 <br/>
  > **Denosing Autoencoder(DAE)** <br/>
     : Training data에 noise를 추가하여 인코더에 넣어 학습된 결과와 노잉즈를 붙이기 전 데이터의 error를 최소화하는 목적을 가지는 Autoencoder.

<br/>

- GSN과 비교할 때, adversarial net은 샘플링을 위해 Markov chain을 필요로 하지 않는다. adversarial net들은 generation 과정에서 피드백 루프를 필요로 하지 않기 때문이다. 본 연구는 GSN에서 사용되는 Markov Chains을 제거함으로써 generative machine의 아이디어를 확장한다.
  > **Markov chain** <br/>
  > : Markov 성질을 지닌 이산 확률 과정을 의미. <br/>
  > : Markov 성질이란 '특정 상태의 확률은 오직 과거의 상태에 의존한다'는 것. <br/> 
  > (참고) https://www.puzzledata.com/blog190423/ <br/>
  > (참고) https://velog.io/@sapphire317/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Markov-Chain <br/>

<br/>

- back-propagating을 이용한 generative machine을 학습 시키는 방법들은 auto-encoding variational Bayes와 stochastic back-propagation 등이 있다.
  > (auto-encoding variational Bayes, VAE) https://arxiv.org/abs/1312.6114 <br/>
  
  > **stochastic back-propagation** <br/>
  > : We now develop the key identities that are used to allow for efficient inference by exploiting specific properties of the problem of computing gradients through random variables. We refer to this computational strategy as stochastic backpropagation(논문 인용).

<br/>

> (참고)
> - GAN은 종종 **Adversarial Examples**와 혼동되기도 한다.
>  > **Adverarial Examples** <br/>
>  > Original input에 매우 작은 noise를 더하여(perturbation) 사람의 눈에는 차이가 없어 보이나 분류기는 잘못 분류 하도록 변형된 input data. <br/>
>  > 모델이 오분류를 하게끔 noise를 더한 이미지를 adversarial example. <br/>
>  > (참고) https://noru-jumping-in-the-mountains.tistory.com/16?category=1218655 
> -  Adversarial examples는 실제 데이터와 유사하지만 오분류되는 예시들을 찾기위해 그레디언트 기반 최적화를 입력 데이터에 직접적으로 적용함으로써 찾아지는 examples이다. <br/>
  생성모델의 훈련을 위한 메커니즘이 아니기 때문에 GAN과는 전혀다르다.
> - Adversarial examples는 인간이 보기에는 차이가 거의 없어보이는 관측을 매우 강한 확인을 갖고 서로 다른 클래스로 분류하는 신경망의 흥미로운 결정방식을 분석하는 툴로써 주로 사용된다.
> - Adversarial examples의 존재는 GAN의 훈련이 비효율적일 수 있음을 시사한다. 적대적 사례들은 판별모델이 인간이 인식하는 데이터에 대한 속성을 모방하지 않고 확신에 찬 인식을 하게끔 만들어버릴 수 있음을 보이기 때문이다. 
> - 이는 GAN 뿐만 아니라 다른 모든 discriminative network가 해결해야할 문제.

<br/>


#### **!! 정리하고 가기 !!** <br/>
Point 1. Generative model
1. Deep generative model들은 maximum likelihood estimation과 관련된 전략들에서 발생하는 많은 확률 연산들을 근사하는 데 발생하는 어려움과 generative context에서는, 앞서 모델 사용의 큰 성공을 이끌었던 선형 활성화 함수들의 이점들을 가져오는 것의 어려움이 있기 때문에 크게 임팩트 있지 않았음.
2. 논문에서 소개될 새로운 generative model은 이러한 어려움들을 회피한다.<br/>

Point 2. Adversarial nets <br/>
1. 이 논문에서 소개되는 adversarial nets 프레임워크의 컨셉은 ‘경쟁’으로, discriminative model은 sample data가 G model이 생성해낸 sample data인지, 실제 training data distribution인지 판별하는 것을 학습함.
2. 이 프레임워크는 많은 특별한 학습 알고리즘들과 optimization 알고리즘을 사용할 수 있음.
3. (abstract)에서 나왔듯이 이 논문에서는 multi-layer perceptron을 사용하면 multilayer perceptron을 쓰면 다른 복잡한 네트워크 필요 없이 오직 forward propagation/ back propagation / dropout algorithm으로 학습 가능.


<br/>
<br/>

## **3. Adversarial nets**
- adversarial modeling 프레임워크는 모델이 둘다 MLP일때 가장 간단히 적용할 수 있다.
- 학습 초반에는 G가 생성해내는 이미지는 D가 G가 생성해낸 가짜 샘플인지 실제 데이터의 샘플인지 바로 구별할 수 있을 만큼 형편없어, D(G(z))의 결과가 0에 가깝다. <br/>
  즉, z로 부터 G가 생성해낸 이미지가 D가 판별하였을 때 바로 가짜라고 판별할 수 있다고 하는 것을 수식으로 표현한 것이다. <br/>
  그리고 학습이 진행될수록, G는 실제 데이터의 분포를 모사하면서 D(G(z))의 값이 1이 되도록 발전한다. <br/>
  이는 G가 생성해낸 이미지가 D가 판별하였을 때 진짜라고 판별해버리는 것을 표현한 것이다. <br/>
  -> 위 내용을 수식으로 나타낸 것이 아래 수식 = **Loss Function**.

<p align="center"><img src=https://t1.daumcdn.net/cfile/tistory/993D64395C8E306F22 width="700px"></p>
<center> [그림.3] </center> <br/>

- 첫번째 항 : real data x를 discriminator 에 넣었을 때 나오는 결과를 log취했을 때 얻는 기댓값.
- 두번째 항 : fake data z를 generator에 넣었을 때 나오는 결과를 discriminator에 넣었을 때 그 결과를 log(1-결과)했을 때 얻는 기댓값.

- 이 방정식을 D의 입장, G의 입장에서 각각 이해해보면, <br/>
  먼저 D의 입장에서 이 value function V(D,G)의 이상적인 결과를 생각해보면, D가 매우 뛰어난 성능으로 판별을 잘 해낸다고 했을 때, D가 판별하려는 데이터가 실제 데이터에서 온 샘플일 경우에는 D(x)가 1이 되어 첫번째 항은 0이 되어 사라지고 G(z)가 생성해낸 가짜 이미지를 구별해낼 수 있으므로 D(G(z))는 0이 되어 두번째 항은 log(1-0)=log1=0이 되어 전체 식 V(D,G) = 0이 된다. 즉 D의 입장에서 얻을 수 있는 이상적인 결과, '최댓값'은 0임을 확인 할 수 있다.
- G의 입장에서 이 value function V(D,G)의 이상적인 결과를 생각해보면, G가 D가 구별못할만큼 진짜와 같은 데이터를 잘 생성해낸다고 했을 때, 첫번째 항은 D가 구별해내는 것에 대한 항으로 G의 성능에 의해 결정될 수 있는 항이 아니므로 패스하고 두번째 항을 살펴보면 G가 생성해낸 데이터는 D를 속일 수 있는 성능이라 가정했기 때문에 D가 G가 생성해낸 이미지를 가짜라고 인식하지 못하고 진짜라고 결정내버린다. 그러므로 D(G(z)) =1이 되고 log(1-1)=log0=마이너스무한대가 된다. 즉, G의 입장에서 얻을 수 있는 이상적인 결과, '최솟값'은 '마이너스무한대'임을 확인할 수 있다.
- 다시말해, D는 training data의 sample과 G의 sampl에 진짜인지 가짜인지 올바른 라벨을 지정할 확률을 최대화하기 위해 학습하고, G는 log(1-D(G(z))를 최소화(D(G(z))를 최대화)하기 위해 학습되는 것!
- D입장에서는 V(D,G)를 최대화시키려고, G입장에서는 V(D,G)를 최소화시키려고 하고, 논문에서는 D와 G를 V(G,D)를 갖는 two-player minmax game으로 표현했다.
- 또 학습시킬때, inner loop에서 D를 최적화하는 것은 많은 계산들을 필요로 하고 유한한 데이터셋에서는 overfitting을 초래함 -> 그래서 k step만큼 D를 최적화하고 G는 1 step 만큼 최적화하도록 한다.

<p align="center"><img src=https://media.vlpt.us/images/changdaeoh/post/6a546a06-547d-4073-97a9-53586ffaf44a/image.png width="500px"></p>
<center> [그림.4] </center> <br/>
> (파란색 점선: discriminative distribution, 검은색 점선: data generating distribution(real, 실제 데이터 분포), 녹색 실선: generative distribution(fake, 생성기(G)가 모델링하는 데이터 분포 Pg))

- GAN의 학습과정을 이 그림을 통해 확인해보면, (a): 학습초기에는 real과 fake의 분포가 전혀 다름. D의 성능도 썩 좋지 않음 (b): D가 (a)처럼 들쑥날쑥하게 확률을 판단하지 않고, 흔들리지 않고 real과 fake를 분명하게 판별해내고 있음을 확인할 수 있다. 이는 D가 성능이 올라갔음을 확인가능 (c): 어느정도 D가 학습이 이루어지면, G는 실제 데이터의 분포를 모사하며 D가 구별하기 힘든 방향으로 학습을 함 (d): 이 과정의 반복의 결과로 real과 fake의 분포가 거의 비슷해져 구분할 수 없을 만큼 G가 학습을 하게되고 결국, D가 이 둘을 구분할 수 없게 되어 확률을 1/2로 계산하게 된다.
- 이 과정으로 진짜와 가짜 이미지를 구별할 수 없을 만한 데이터를 G가 생성해내고 이게 GAN의 최종 결과라고 볼 수 있다.

<br/>
<br/>

## **4. Theoretical Results**
- 앞서 제시되었던 GAN의 minmax problem이 제대로 동작 한다면, <br/>
  minmax problem이 global minimum에서 unique solution을 가지고 어떠한 조건에 만족하면 그 solution으로 수렴한다는 사실이 증명되어야 한다.
- 논문에서는 이부분에 관하여 2가지(Global Optimality of Pdata, Convergence of Algorithm)를 증명하였다.
- 본 자료에서는 생략한다.
> 증명에 대한 자세한 내용이 궁금하다면 아래 자료와 영상 참고 바람.
> (자료) https://deeplearning.cs.cmu.edu/F20/document/slides/bstriner_gans_part_1.pdf 
> (영상) https://www.youtube.com/watch?v=cBgX-8IxRUM 

- 알고리즘 관련하여서만 잠깐 살펴보자면,<br/>
  <p align="center"><img src=https://media.vlpt.us/images/changdaeoh/post/462027a5-49a5-4b46-a362-1a211e275e24/image.png width="600px"></p>
  <center> [그림.5] </center> <br/>

<br/>
<br/>

## **5. Experiments**
- MNIST, TFD(Toronto Face Database), CIFAR-10에 대해 훈련.
- generator에서는 ReLU, sigmoid activation을 섞어 사용.
- discriminator에서는 maxout activation만을 사용.
- discriminator 훈련시 드랍아웃 사용.
- 저자들이 제안하는 프레임워크는 generator의 중간 레이어들에 dropout과 noise 추가를 이론적으로   허용하지만, 오직 generator의 최하단 레이어에만 노이즈를 추가했다고 함.

- 실험에서는 Gㄹ 생성된 sample에 Gaussian Parzen window맞추고, 해당 분포에 따른 log-likelihood를 알려줌으로써 Pg에 따른 test set 데이터의 확률을 추정하였다.

<p align="center"><img src=https://media.vlpt.us/images/changdaeoh/post/c728e429-2caa-4bb1-9e2e-5bede4ee0b08/image.png width="500px"></p>
<center> [그림.6] </center> <br/>

<p align="center"><img src=https://media.vlpt.us/images/changdaeoh/post/6d8d032c-c0c5-4f58-a39c-8b1d1427638a/image.png width="500px"></p>
<center> [그림.7] </center> <br/>

- GAN 모델은 일반적인 머신 러닝, 혹은 딥 러닝 모델과는 달리 명확한 평가의 기준이 없다. Loss는 단지 학습을 위한 오토 파라미터의 구실을 하는 셈이고, 실제적인 Loss를 나타내거나 Accuracy와 같은 기준이 되는 명확한 평가지표가 존재하지 않는다. -> 즉, 정량적 지표가 현재까지 없다(지속적인 연구 필요).

- 논문에서는 실험 결과에 대하여 G가 생성해낸 sample이 기존 존재 방법으로 만든 sample보다 좋다고 주장할 수 는 없지만, 더 나은 생성모델과 경쟁할 수 있다 생각하며, adversarial framework의 잠재력 강조하였다.

<br/>
<br/>

## **6. Advantages and disadvantages**

**장점**

- Markov chain 불필요.
- 학습단계에서 inference 필요없음. 
- 다양한 함수들이 모델이 접목될 수 있음.
- Markov chains을 쓸 때보다 훨씬 선명한 이미지를 얻을 수 있음.

**단점**
- Pg(x)에 대한 명시적인 표현이 없음.
- 훈련동안 D와 G가 반드시 균형을 잘 맞춰 학습되어야 함.
- 최적해로의 수렴에 있어 이론적 보장의 부족.

<br/>
<br/>

!! 간략 정리 !!
- Adversarial Learning이라는 개념을 생성모델에 처음 적용한 새로운 프레임워크
- generator와 discriminator 두 모델을 경쟁적으로 학습시켜 둘 모두를 동시에 최적화함.
- 데이터의 분포에 대한 명시적인 가정없이 데이터 생성이 가능함.
- 그 당시 생성모델연구에서 주로 사용되던 MC method나 Approximate Inference등에 의존하지 않음.
- VAE와 비슷하게 훈련과정에서 latent space를 학습.
  학습 완료 후 학습된 latent space에서 임의로 샘플링하여 generator에 전달함으로써 새로운 데이터 생성.
- 단일모델 훈련시키는거에 비해서 난이도가 매우 높은편, generator와 discriminator 균형 맞추는게 실제로 엄청 어려움.
- 매우 많은 후속연구들을 유발한 멋들어지고 우수한 모델. 많은 변형모델들을 통해 이미지 생성 task에서 SOTA로 군림.

<br/>
<br/>
<br/>
<br/>
<br/>

### [ Reference ]
- [논문 설명] https://tobigs.gitbook.io/tobigs/deep-learning/computer-vision/gan-generative-adversarial-network
- [논문 설명] https://velog.io/@tobigs-gm1/basicofgan
- [논문 설명] https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI-2.html
- [논문 설명] https://m.blog.naver.com/euleekwon/221557899873
- [논문 설명] https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN
- [논문 설명] https://dhhwang89.tistory.com/27
- [논문 설명] https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/
- [논문 설명] https://yamalab.tistory.com/98 
- [그림.1] https://velog.io/@tobigs-gm1/basicofgan
- [그림.2] https://sites.google.com/site/aidysft/generativeadversialnetwork  
- [그림.3] https://dongsarchive.tistory.com/31 
- [그림.4] https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN 
- [그림.5] https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN 
- [그림.6] https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN 
- [그림.7] https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN 
- [최대 가능도] https://greedywyatt.tistory.com/114
- [Restricted Boltzmann machines] http://wiki.hash.kr/index.php/%EC%A0%9C%ED%95%9C_%EB%B3%BC%EC%B8%A0%EB%A7%8C_%EB%A8%B8%EC%8B%A0
- [Denoising Autoencoder] https://blogyong.tistory.com/31 
- [Adverarial Examples] https://lepoeme20.github.io/archive/FGSM 
