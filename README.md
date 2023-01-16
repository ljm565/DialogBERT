# DialogBERT
## 설명
본 코드는 [DialogBERT](https://arxiv.org/pdf/2012.01775.pdf) 모델을 구현합니다.
DialogBERT는 multi-turn 대화 생성을 위한 모델입니다.
본 코드는 [원본 코드](https://github.com/guxd/DialogBERT)를 따르되 좀 더 보기 쉽게 리팩토링을 진행하였습니다.
다만 코드가 'greedy'로 문장을 생성하게 하면 거의 비슷한 문장만 생성하는 이슈가 있었으며, 원본 코드를 돌려봐도 마찬가지였습니다.
또한 결과 재현이 안되었으며, 이와 관련된 이슈는 [원본 코드 이슈](https://github.com/guxd/DialogBERT/issues)를 참고하시기 바랍니다.
DialogBERT 대한 설명은 [DialogBERT 설명글](https://ljm565.github.io/contents/dialogbert1.html)을 참고하시기 바랍니다.
<br><br><br>

## 모델 종류
* ### DialogBERT
    Multi-turn 대화 생성 모델을 학습하기 위해 DialogBERT를 학습합니다.
<br><br><br>


## 토크나이저 종류
* ### Wordpiece Tokenizer
    Likelihood 기반으로 BPE를 수행한 subword 토크나이저를 사용합니다.
<br><br><br>


## 사용 데이터
* 실험으로 사용하는 데이터는 [DailyDialog](http://yanran.li/dailydialog) 데이터셋입니다.
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, inference}, **필수**: 학습을 시작하려면 train, 학습된 모델의 문장 생성, BLEU 등 결과를 보고싶은 경우에는 inference로 설정해야합니다.
    inference 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m {inference} 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, inference를 할 경우에도 실험할 모델의 이름을 입력해주어야 합니다(최초 학습시 src/config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 src/main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 src/main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 BLEU 등의 결과 등을 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 src/main.py -d cpu -m inference -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * pretrained_model: "bert-base-uncased", "bert-base-cased" 등 pre-trained 토크나이저 선택을 위한 모델.
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/src/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/src/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * max_multiturn: 최대 multi-turn 수.
    * max_utterance: 최대 문장 길이.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: learning rate 지정.
    * early_stop_criterion: Validation set의 최대 BLEU-4를 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.
    * result_num: 모델 테스트 시, 결과를 보여주는 sample 개수.
    * temperature: 문장 생성 시 temperature parameter, mode가 'sampling'일 때 적용.
    * mode: {'greedy', 'sampling'} 중 선택, sampling은 multinomial sampling을 수행.
    <br><br><br>


## 기타
* "Greedy"로 생성한 문장은 서로 다른 source에 비해 거의 비슷하거나 같은 문장만을 생성합니다.
이는 DialogBERT의 원본 코드도 마찬가지이며, 이에 대한 원인은 DialogBERT의 context encoder BERT의 [CLS] 토큰의 결과만을 사용하기 때문입니다.
이 값을 확인해보았을 때 서로 다른 source에 대해 조금씩 다른 representation을 가지기는 하나, 그 차이가 미미하였기에 발생하는 현상이라고 생각합니다.
아마도 저자는 source에 대해 서로 다른 문장 생성을 위해서 multinomial sampling을 하여 생성을 하지 않았나 싶고, 이는 본 코드와 [원본 코드](https://github.com/guxd/DialogBERT)에도 구현 되어있습니다.
다만 좋은 모델이라면 'greedy' 생성에 대해 좋은 결과를 내어줘야한다고 생각하지만, 이렇게 하는 경우 성능이 잘 나오지 않는 것 같습니다.


<br><br><br>
