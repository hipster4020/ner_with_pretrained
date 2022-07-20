# 📰 News Company NER
기업명을 추출하는 NER Model train<br>
<blockquote></blockquote><br><br>



## 🤖 Model
KLUE bert<br><br>
## 📰 Klue NER Dataset
기업명만 ORG tagging, 나머지 O tagging<br>
<blockquote>'LC', 'DT', 'OG', 'TI', 'QT', 'O', 'PS'<br>
</blockquote>LC : location 지역 명칭과 행정구역 명칭 등<br>
DT : DATE 날짜<br>
OG : ORGANIZATION 기관 및 단체와 회의/회담을 모두 포함<br>
TI : TIME 시간<br>
O : 기타<br>
PS : PERSON 인물명<br>


## tree
* [src]
  * [config.yml]
  * [data.py]
  * [dataloader.py]
  * [generic.py]
  * [ner_pipeline.py]
  * [train.py]
* [.gitignore]
* [README.md]
* [requirements.txt]