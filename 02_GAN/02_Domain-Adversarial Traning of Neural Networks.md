# **Domain-Adversarial Training of Neural Networks(DANN)**
> (paper) https://arxiv.org/abs/1505.07818 

<br/>

- 2016년 JMLR(Journal of Machine Learning Research)발표 논문. 
- 이 논문은 Train과 test의 data distribution이 다른 경우, Domain Adaptation을 효과적으로 할 수 있는 새로운 접근방법을 제시. <br/>

<br/>
<br/>

## **01. Domain Adaptation**
- training distribution과 test distribution 간에 차이가 있을 때 classifier 또는 predictor의 학습 문제를 다루는 연구 분야.
- train과 test분포가 약간 다른 환경에서 효율적인 학습을 진행하려는 과정이 **Domain Adaptation(DA)**.
- source(training time)과 target(test time) 사이의 mapping을 통해 source domain에서 학습한 classifier가 target domain에서도 효과적으로 동작하는 것을 목표.
- 두 가지의 domain(source domain and target domain)에 대하여 한 쪽의 domain을 다른 쪽으로 조정/맞추려(adapt)하는 것.

<p align="center"><img src=https://blog.kakaocdn.net/dn/BqW0U/btqF1TGQDkA/5FOcuI0Sw9cQzdaLNGe9PK/img.png width="300px"></p>
<center> [그림.1] </center> 

- [그림.1]를 참고할 때, SVHN숫자 dataset에서 training시킨 네트워크로 전혀 다른 style을 갖는 MNIST 숫자 dataset을 분류하는 데에 쓰고 싶을 때 *두 domain의 간격을 줄여주는 방법(=domain adaptation)*.
> 용어 정리<br/>
> -train dataset : source dataset<br/>
> -test dataset : target dataset<br/>
> -train domain : source domain<br/>
> -test domain : target domain<br/>
> -Multi source domain adaptation : 다양한 domain이 source domain으로 주어짐<br/>
> -Multi target domain adaptation : 다양한 domain이 target domain으로 주어짐

<br/> 

- Domain Adaptation은 Transfer learning에 속하며, source domain에서만 label data가 존재하는 경우를 다룬다.
<p align="center"><img src=https://jamiekang.github.io/media/2017-06-05-domain-adversarial-training-of-neural-networks-taxonomy.jpg width="600px"></p>
<center> [그림.2] </center>

<br/>
<br/>
<br/>

## **02. Domain Adaptation(DA) 이론적 배경**
- DA의 이론적 배경은 논문 “Analysis of Representations for Domain Adaptation”에 기반한다.
> Analysis of Representations for Domain Adaptation, S.Ben-David, 2006 <br/>
> (paper) https://webee.technion.ac.il/people/koby/publications/nips06.pdf
<br/> 

- 본 논문에서 풀고자하는 문제는, input space *X*에서 label의 집합인 *Y={0, ... , L-1}* 로의 classification task이다.

$$
Source Domain:D_s,
$$

$$
Target Domain : D_t, 
$$

$$
Target Risk : R_{D_T}
$$


<p align="center"><img src=https://ifh.cc/g/11mdNP.jpg width="500px"></p>

- 제안하는 알고리즘의 목표는 target domain의 label에 대한 정보가 없더라도 target risk가 낮도록 classifier *η* 를 만드는 것이다.
<p align="center"><img src=https://latex.codecogs.com/svg.image?R_%7B%5Cmathcal%7BD%7D_T%7D(%5Ceta)=%5CPr_%7B(x,y)%5Csim%5Cmathbb%7BD%7D_T%7D%5Cleft(%20%5Ceta(x)%20%5Cneq%20y%20%5Cright) width="250px"></p>

- 목적은 target domain error를 줄이는 것이다. <br/> 
  target domain error는 source domain error + domain divergence로 upper bound를 정할 수 있다.
  즉, <u>**source domain에서 classify를 잘하고, source domain과 target domain의 거리가 가깝다면 target error가 작을 것**</u> 이다.

- 두 domain간의 거리는 아래 식과 같이 *H-divergence*로 계산할 수 있다.<br/>
  (H-Divergence는 Domain Divergence를 정의하기위해 나온 개녕 중 하나)
<p align="center"><img src=https://latex.codecogs.com/svg.image?%20d_%7B%5Cmathcal%7BH%7D%7D(%5Cmathcal%7BD%7D_S%5EX,%5Cmathcal%7BD%7D_T%5EX)%20=%202%5Csup_%7B%5Ceta%5Cin%5Cmathcal%7BH%7D%7D%5Cleft%7C%20%5CPr_%7B%5Cmathbf%7Bx%7D%5Csim%20%5Cmathcal%7BD%7D_S%5EX%7D%5Cleft%5B%20%5Ceta(%5Cmathbf%7Bx%7D)=1%5Cright%5D%20-%20%5CPr_%7B%5Cmathbf%7Bx%7D%5Csim%20%5Cmathcal%7BD%7D_T%5EX%7D%5Cleft%5B%20%5Ceta(%5Cmathbf%7Bx%7D)=1%5Cright%5D%5Cright%7C width="450px"></p>

- S.Ben-David는 H-divergence를 다음과 같이 정의했다.
> **H-divergence** <br/>
> : 여러 domain classifier를 원소로하는 classifer *η* 집합을 hypothesis class *H*라고 정의했을 때, H-divergence란 두 도메인을 잘 구분하는 classifier *η* 를 얼마나 담을 수 있는지는 뜻하기에 domain을 구분하는 능력을 의미한다.<br/>
>  classifier *η*가 H 안에 존재하려면(H는 classifier *η*의 집합이라고 위에 언급) H의 dimension은 충분히 커야한다. 그러나, complexity term이 존재하기에 H의 dimension은 한없이 커져서만도 안된다. 즉, 잘 구분하는 classifier *η*를 H에 포함할 수 있을 만큼 H의 complexity는 커야하지만, complexity가 너무 크다면, constant complexity의 값이 커져 결국 target error의 upper bound는 커져버리게 된다. 따라서 충분한 성능을 내는 classifier가 있어야 하나 이 classifier가 너무 복잡해서는 안 되겠다. ~그래서 간단한 MLP나 SVM을 주로 쓰나 보다.~

- *H*가 symmetric할 때 empirical H-divergence를 계산할 수 있고, empirical H-divergence는 아래식과 같다.
<p align="center"><img src=https://latex.codecogs.com/svg.image?%5Chat%7Bd%7D_%7B%5Cmathcal%7BH%7D%7D(S,T)%20=%202%5Cleft(1-%5Cmin_%7B%5Ceta%5Cin%5Cmathcal%7BH%7D%7D%20%5Cleft%5B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi=1%7D%5En%20I%20%5Cleft%5B%20%5Ceta(x_i)=1%5Cright%5D%20&plus;%20%5Cfrac%7B1%7D%7Bn'%7D%5Csum_%7Bi=n&plus;1%7D%5EN%20I%20%5Cleft%5B%20%5Ceta(x_i)=0%5Cright%5D%5Cright%5D%5Cright) width="450px"></p>

>*여기서 I[a]는 a가 true면 1 아니면 0인 indicator function). 
> *empirical H-divergence 수식은 논문 Appendix에 아주 상세히 증명이 나와있음. <br/>
> *논문에서 empirical H-divergence수식이 위에 기재된거랑 다른데 typo 추정.<br/>

- 그러나, 일반적으로 empirical H-divergence 값을 정확하게 계산하는 것이 어렵기 때문에 아래의 식으로 근사하고, 근사한 식을 **Proxy A Distance(PAD)** 라고 부른다.<br/>
  (이후 이 논문의 실험들에서는 이 PAD값을 사용함)
<p align="center"><img src=https://latex.codecogs.com/svg.image?%5Chat%7Bd%7D_%7B%5Cmathcal%7BA%7D%7D%20=%202%5Cleft(1-2%5Cepsilon%20%5Cright) width="130px"></p>

- 여기서 ϵ은 classification error입니다. 즉, sample의 출처가 source domain인지 target domain인지 classifier가 정확히 구분할 수 있으면 ϵ=0 이다.

<br/>
<u>!! 정리하자면 !!</u> <br/>
- 도메인이 달라지더라도 충분히 일반화할 수 있도록 모델을 학습하려면, source domain에서의 classifier 성능을 높이면서 한편 domain을 구분하는 성능은 낮아지게 훈련해야한다. <br/>
- 즉, 다른 말로 하면 label classifier의 loss를 minimize하면서 동시에 domain classifier의 loss를 maximize하도록 optimize하는 문제를 푸는 것이 되기 때문에 이 논문에서 adversarial이라고 표현!

<br/>
<br/>
<br/>

## **03. Domain-Adversarial neural networks(DANN) 구조**
<br/>
<p align="center"><img src=https://jamiekang.github.io/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig1.jpg width="600px"></p>
<center> [그림.3] </center>
<br/>

- 크게 feature extractor(초록), label predictor(파랑), domain classifier(빨강)로 구성되어 있다. 
- 앞에서 설명한 것처럼 domain을 구분하는 성능을 낮추기 위해 추가된 부분이 domain classifier인데, 앞 단의 feature extractor와 gradient reversal layer (black)를 통해 연결되어있다.
- 논문의 목표는 앞의 feature extractor *Gf* 가 최대한 source와 target에 동시에 포함되는, domain의 특성을 지우고 class 분류에만 쓰인 특징을 뽑게 하는 것이다.<br/>
   이를 위하여 back propogation시 domain label을 구분하는 분홍색 네트워크에서 뽑힌 loss에 -람다를 곱해 feature extractor weight를 업데이트한다.  
  이렇게 되면 아래 두개의 목적을 동시에 달성할 수 있게 된다.  <br/>
    1. MInimize Training classification error
    2. Maximize Training & test domain classification error <br/>
<br/>

- 즉, 일반적인 neural network에서는 backpropagation을 통해 prediction loss를 줄이는 방향으로 gradient를 계산하는데, DANN에서는 domain classifier가 prediction을 더 못하게 하려는 것이 목적이므로 gradient reversal layer에서 negative constant를 곱해 부호를 바꿔 전달하는 것이다.
<br/>
<p align="center"><img src=https://jamiekang.github.io/media/2017-06-05-domain-adversarial-training-of-neural-networks-arch2.jpg width="500px"></p>
<center> [그림.4] </center>

<br/>
<br/>
<br/>

## **04. Experiments**
<br/>
<p align="center"><img src=https://jamiekang.github.io/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig2.jpg width="500px"></p>
<center> [그림.5] </center>

- 이 논문에서는 앞에서 보인 알고리즘을 inter-twinning moons 2D problem라고 하는 초승달 모양의 distribution을 가지는 dataset에 적용했다.

- 아래 그림에서 red 색의 upper moon이 source distribution의 label 1이고, green 색의 lower moon이 source distribution의 label 0입니다. black 색의 target distribution은 source distribution을 35도 회전시키고 label을 제거해서 만들었다.


<br/>
<br/>
<p align="center"><img src=https://2.bp.blogspot.com/-zq-STtvcjYY/WI27ecjp7bI/AAAAAAAABO8/WnSD7Vb1wp4mX_zPtTWFC8x2sIi6t9PGQCK4B/s1600/predicted-sNN-vs-sDANN-v2.png width="500px"></p>
<center> [그림.6] </center>

- 빨간색이 'o' 파란색이 '+'로 분류된 점들을 보여줍니다. 두 번째 줄과 세 번째 줄의 결과는 각각 'o'의 ground truth와 '+'의 ground truth를 초록색으로 겹쳐서 각각의 class label에 대해 결과가 어떻게 나왔는지 보여준다.


<br/>
<br/>
<br/>
<br/>
<br/>

### [ Reference ]
- [논문 설명] https://jamiekang.github.io/2017/06/05/domain-adversarial-training-of-neural-networks/
- [논문 설명] https://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html
- [논문 설명] https://jayeon8282.tistory.com/7
- [논문 설명] https://blog.naver.com/PostView.nhn?blogId=dmsquf3015&logNo=222029881504&categoryNo=6&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
- [논문 설명] https://yjs-program.tistory.com/168
- [논문 설명] https://www.youtube.com/watch?v=GmqW_v_bXiM
- [Transduction] https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d
- [그림.1] https://jayeon8282.tistory.com/7
- [그림.2] https://jamiekang.github.io/2017/06/05/domain-adversarial-training-of-neural-networks/
- https://www.codecogs.com/latex/eqneditor.php  
- https://ifh.cc/
- [그림.3] https://jamiekang.github.io/2017/06/05/domain-adversarial-training-of-neural-networks/
- [그림.4] https://jamiekang.github.io/2017/06/05/domain-adversarial-training-of-neural-networks/
- [그림.5] https://jamiekang.github.io/2017/06/05/domain-adversarial-training-of-neural-networks/
- [그림.6]https://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural-3.html 