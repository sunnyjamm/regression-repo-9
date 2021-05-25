# 파테크 Green Onion Investments 

## 개요: 

<img src = "./img/news.jpeg">


1년 사이에 다섯 배로 가격이 뛰었다는 농작물 파에 대한 뉴스를 접하고 파를 집에서 키워 먹는 것이 유행이 되면서, 파값의 가격을 예측해보고 싶었습니다.

"""
통계청 소비자물가동향을 살펴보면 지난 1월 농산물 가격은 지난해보다 11.2% 상승했다. 전체 소비자물가 상승률이 4개월 연속 0%대에 머물러 있는 것과 대조적이다. 특히, 대파·양파 등 채소를 중심으로 가격이 크게 오르며 밥상물가 상승을 부채질했다. 서민층을 중심으로 파, 고추, 상추 등 손쉽게 재배할 수 있는 채소를 직접 키워 먹는 사람이 늘어나는 이유다. 한국농수산식품유통공사(aT) KAMIS 농산물 유통정보에 따르면 2일 기준 대파 1㎏ 소매 가격은 7399원으로 한 달 전(5500원)보다 33.3%, 1년 전(2197원)보다는 무려 236.8%나 급등했다. 도매 가격은 1년 새 408.1%(1140원→5792원)나 오르면서 ‘금(金)파’가 됐다.

양파도 비슷하다. 양파 1㎏ 소매 가격은 3459원으로 한 달 전(3314원)보다 4.4%, 1년 전(2296원)보다는 50.7%나 올랐다. 20㎏ 기준 도매 가격(4만2920원)은 1개월 전(3만7220원)보다 15.3%, 1년 전(2만7730원)보다 54.8% 올랐다.

유통 과정을 거친 채소를 구입하려면 꽤 부담스러운 비용을 지불해야 하지만, 집에서 재배하면 물값과 ‘나의 부지런함’만 있으면 된다.

이투데이 뉴스발전소 김재영 기자
"""

## Getting Started 
Weather data = "https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158"

Price data = "https://www.kamis.or.kr/customer/main/main.do"

agriculture data = "http://www.yongin.go.kr/home/atc/cityConsum/cityConsum05/cityConsum05_02/cityConsum05_02_20.jsp"

### Prerequisites / 선행 조건

Python 3.8.5
Selenium 


### Installing / 설치
```
Python Client
    pip install -U selenium
    
Java Server
  Download the server from http://selenium.googlecode.com/files/selenium-server-standalone-2.28.0.jar
  java -jar selenium-server-standalone-2.28.0.jar
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

## Built With / 누구랑 만들었나요?

* [Sunny Kim](https://github.com/sunnyjamm) - Making dataset using Selenium, working on EDA, Regression analysis and README.md on ./working notebooks/crawling.ipynb, ./working notebooks/main.ipynb, ./working notebooks/Paa.ipynb
* [Chaebeen Seo](https://github.com/chaebeen) - Project idea to planning, working on Regression analysis on ./working notebooks/04.ipynb, ./working notebooks/06.ipynb

## Contributiong / 기여

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.


## Must Keep in Mind that 
This project is using a time series data that is analyzed using Regression methods (Random Forest, OLS, etc.)  
Keep in mind that this project is just for learning, and the result data is not significant. 
Try using time series analysis for this datasets. 
