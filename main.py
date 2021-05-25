#!/usr/bin/env python
# coding: utf-8

# # 파테크

# <img src = "./img/파테크.jpeg">

# ## 개요: 
# 1년 사이에 다섯 배로 가격이 뛰었다는 농작물 파에 대한 뉴스를 접하고 파를 집에서 키워 먹는 것이 유행이 되면서, 파값의 가격을 예측해보고 싶었습니다.
# 
# """
# 통계청 소비자물가동향을 살펴보면 지난 1월 농산물 가격은 지난해보다 11.2% 상승했다. 전체 소비자물가 상승률이 4개월 연속 0%대에 머물러 있는 것과 대조적이다. 특히, 대파·양파 등 채소를 중심으로 가격이 크게 오르며 밥상물가 상승을 부채질했다. 서민층을 중심으로 파, 고추, 상추 등 손쉽게 재배할 수 있는 채소를 직접 키워 먹는 사람이 늘어나는 이유다. 한국농수산식품유통공사(aT) KAMIS 농산물 유통정보에 따르면 2일 기준 대파 1㎏ 소매 가격은 7399원으로 한 달 전(5500원)보다 33.3%, 1년 전(2197원)보다는 무려 236.8%나 급등했다. 도매 가격은 1년 새 408.1%(1140원→5792원)나 오르면서 ‘금(金)파’가 됐다.
# 
# 양파도 비슷하다. 양파 1㎏ 소매 가격은 3459원으로 한 달 전(3314원)보다 4.4%, 1년 전(2296원)보다는 50.7%나 올랐다. 20㎏ 기준 도매 가격(4만2920원)은 1개월 전(3만7220원)보다 15.3%, 1년 전(2만7730원)보다 54.8% 올랐다.
# 
# 유통 과정을 거친 채소를 구입하려면 꽤 부담스러운 비용을 지불해야 하지만, 집에서 재배하면 물값과 ‘나의 부지런함’만 있으면 된다.
# 
# 이투데이 뉴스발전소 김재영 기자
# """
# 
# <img src = "./img/news.jpeg">

# ### 크롤링 
# -  https://www.kamis.or.kr/customer/main/main.do
# - 농산물유통정보 사이트에서 날자별로 파값 (대파) 가격 크롤링. 
#     - 주말과 공휴일에는 데이터가 없어서 데이터가 없는 날은 전 날의 가격으로 처리. 
# 
# - https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158
# - Selenium을 이용해 기상자료개방포털에서 매일 가격 크롤링. 
#     - 기간을 하루 단위로 하지 않으면 sum 값이 나오기 때문에 기간을 하루 단위로 설정해서 크롤링.

# In[ ]:


# 농산물 유통정보 사이트 
url = "https://www.kamis.or.kr/customer/price/product/item.do?action=priceinfo&regday={}&itemcategorycode=200&itemcode=246&kindcode=&productrankcode=&convert_kg_yn=N"

# 1년 단위로 크롤링을 해서 합쳐주었다. 
period = pd.date_range('2015.01.01', '2016.01.01', freq='D').strftime('%Y-%m-%d') 

price_df = pd.DataFrame()

# 하루 단위로 파가격 크롤링 해서 데이터프레임에 어팬드. 
for date in period: 
    urls = url.format(date)
    df = pd.read_html(urls, header=0)[3].iloc[:3,1:2]
    price_df = price_df.append(df.T)
    
price_df.reset_index(drop=True, inplace=True)
date_df = pd.DataFrame(period)
price_df= pd.concat([date_df, price_df], axis=1)

# 컬럼명 바꾸기 
price_df.columns = ["date", "avg_price", "max_price", "min_price"]

price_df


# In[ ]:


from selenium import webdriver
from selenium.webdriver.support.select import Select
import time 


# In[ ]:


# 기상자료개방포털에서 하루 기온, 강수량 크롤링 

def weather(): 
    
    driver = webdriver.Chrome('./chromedriver')
    driver.get("https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158")
    
    # 지역 클릭 
    region = """//*[@id="btnStn"]"""
    driver.find_element_by_xpath(region).click()

    time.sleep(2)
    
    # 지역을 전체로 선택 
    choose_all = """//*[@id="ztree_1_check"]"""
    driver.find_element_by_id("ztree_1_check").click()
    
    # 선택하기 클릭
    finish = """//*[@id="sidetreecontrol"]/a"""
    driver.find_element_by_xpath(finish).click()

    # 일별 선택 
    day = """//*[@id="dataFormCd"]/option[1]"""
    driver.find_element_by_xpath(day).click()

    df = pd.DataFrame()
    
    years = range(2018, 2022)
    months = range(1, 13)
    dates = range(1, 32)
    
    for year in years:   
        
        # 시작하는 년도: 끝나는 년도와 동일
        startYear = Select(driver.find_element_by_xpath('//*[@id="startYear"]'))
        startYear.select_by_value(f'{year}')

        # 끝나는 년도: 시작하는 년도랑 동일 
        endYear = Select(driver.find_element_by_xpath('//*[@id="endYear"]'))
        endYear.select_by_value(f'{year}')
        
        for month in months: 
            
           # try:
                time.sleep(0.2)
            
                # 시작하는 월 = 끝나는 월
                startMonth = """//*[@id="startMonth"]/option[{}]"""
                driver.find_element_by_xpath(startMonth.format(month)).click()

                time.sleep(0.2)
                
                # 끝나는 월 = 시작하는 월 
                endMonth = """//*[@id="endMonth"]/option[{}]"""
                driver.find_element_by_xpath(endMonth.format(month)).click()

                for date in dates:
                    
                    time.sleep(0.2)
                    
                    # 시작하는 날짜 = 끝나는 날짜 
                    startDate= """//*[@id="startDay"]/option[{}]"""
                    driver.find_element_by_xpath(startDate.format(date)).click()

                    time.sleep(0.2)
                    # 끝나는 날짜 = 시작하는 날짜 
                    endDate= """//*[@id="endDay"]/option[{}]"""
                    driver.find_element_by_xpath(endDate.format(date)).click()

                    time.sleep(0.2)
                    # 서치 클릭
                    search = """//*[@id="schForm"]/div[3]/button"""
                    driver.find_element_by_xpath(search).click()
                    
                    # 테이블 가져와서 데이터프레임에 어펜드 
                    soup = BeautifulSoup(driver.page_source, "lxml")
                    table = soup.find_all('table')[1]
                    df1 = pd.read_html(str(table),header=0)
                    df = df.append(df1)

                    if date is None: 
                        print("day is none")

            #except: 
             #   df = df.append(np.nan)

        return df


# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[97]:


datas = pd.read_excel('./datas/dataPA.xlsx', index_col='date', parse_dates=True)
datas = datas.sort_values(by="date", ascending=True)
datas.head(3)


# In[70]:


# 컬럼 설명
# avg_price = 평균 파 가격
# max_price = 당일 파 최고값 
# min_price = 당일 파 최저값 
# avg_temp = 평균 온도 
# max_temp = 당일 최고온도
# min_temp = 당일 최저온도
# rain_fall = 평균 강수량 
# paddy = 논 
# field = 밭 
# total_field = 합계
# cons_price = (월별) 소비자물가상승률/ 전년비, 전년동월비
# agr_price = (월별) 농축수산물 물가 상승률 


# In[98]:


# 상관관계를 보고 싶어서 heatmap을 그려보았다. 
plt.figure(figsize=(12,10))
sns.heatmap(datas.corr(), cmap="Purples", annot=True)
plt.show()


# <img src = './heatmap.png' width="700">

# In[6]:


# MinMax scaler로 스케일을 해준 뒤 
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
datas["mms_price"] = mms.fit_transform(datas[["avg_price"]])
datas["mms_temp"] = mms.fit_transform(datas[["avg_temp"]])
datas["mms_field"] = mms.fit_transform(datas[["field"]])
datas["mms_agr"] = mms.fit_transform(datas[["agr_price"]])


# In[7]:


# 파 가격이 2020년 초반부터 가파르게 오르는 것을 확인할 수 있었다. 빨리 원인을 찾고 싶었다. 
fig, axs = plt.subplots(2, 1, figsize=(18, 12))
axs[0].plot("avg_price", data=datas)
axs[0].set(xlabel="years", ylabel="price per pa (KRW)", title="Price over years")
axs[1].plot("mms_price", label='price', data=datas)
axs[1].plot("mms_temp", label='temperature', data=datas)
axs[1].set(ylabel="scaled", xlabel="years", title="Price and weather")
axs[1].legend()
plt.show()


# In[8]:


# NaN 값이 없는 기간 동안의 가격, 온도, 논밭, 농산물 물가를 한번에 plot 해보았다. 
period = datas['2017-01-01':'2021-01-01']

fig, axs = plt.subplots(figsize=(18, 10))
plt.plot("mms_price", label='price', data=period)
plt.plot("mms_field", label='field', data=period)
plt.plot("mms_agr", label='agriculture', data=period)
plt.ylabel("scaled")
plt.xlabel("years")
plt.title("Green onion price")
plt.legend()
plt.show()


# In[9]:


# price에 대한 나머지 컬럼들의 상관관계를 보았다. 
# 물가, 논밭, 기온, 강수량 순서였다. 

corr_matrix = datas.corr()
print(corr_matrix["avg_price"].sort_values(ascending=False))


# In[10]:


# 상관관계는 논, 밭, 물가가 높게 나왔지만 
# info를 보았다. 논, 밭, 물가는 최근 데이터가 없었다.

datas.info()


# In[11]:


import missingno as msno
msno.matrix(datas)
plt.show()


# In[12]:


# 그래서 NaN값이 너무 많고 하루가 아닌 한 달이나 일년 단위로 나오는 논밭과 소비자 물가 데이터를 없애주었다. 

datas = datas.iloc[:,:7]
datas.head()


# In[13]:


# 이제 데이터의 분포도를 확인해보았다. 

import matplotlib.pyplot as plt 
datas.hist(bins=10, figsize=(15,10), color='b')
plt.show()


# In[14]:


# rain_fall에 0이 가장 많았기 때문에 rain_Fall을 0으로 replace 해주었다. 
datas['rain_fall'] = datas['rain_fall'].replace(np.nan, 0)


# In[15]:


# 결측치는 이제 없어졌다 
import missingno as msno
msno.matrix(datas)
plt.show()


# ### Moving Average 구하기
#     - 당일의 날씨와 온도가 당일의 가격을 결정짓는다는 오류를 해결하기 위해 Moving average를 구해서 
#     - 파종 기르기, 생육기, 수확, 그리고 유통 기간까지 합쳐서 5개월 단위로 rolling average를 잡고
#     - 5개월 동안의 mean 날씨가 당일의 가격을 결정한다고 가정을 했습니다.
#     - 가격을 뺀 모든 컬럼에 Moving Average를 구해 컬럼에 더해주었습니다.
#     - 정보: http://www.yongin.go.kr/home/atc/cityConsum/cityConsum05/cityConsum05_02/cityConsum05_02_20.jsp

# <img src = "./pa_time.png">

# In[16]:


# Moving average를 mean 값으로 잡았을때 


# In[18]:


datas.head(1)


# In[45]:


data = pd.DataFrame()
datas['ma_temp'] = datas['avg_temp'].rolling(window=152).mean()
datas['ma_min_temp'] = datas['min_temp'].rolling(window=152).mean()
datas['ma_temp'] = datas['max_temp'].rolling(window=152).mean()
datas['ma_max_temp'] = datas['avg_temp'].rolling(window=152).mean()
datas['ma_field'] = datas['total_field'].rolling(window=152).mean()
datas['ma_cons'] = datas['cons_price'].rolling(window=152).mean()


# In[59]:


d = pd.read_csv('moving_data.csv')


# In[30]:


# temperature가 세 개가 있는데 무엇을 쓰는게 더 가격과의 상관관계가 잘 나올지? 
# ma_max_temperature의 corr coef 값이 가장 잘 x나왔다. 
corr_matrix = data[["avg_price","ma_temp", "ma_min_temp", "ma_max_temp"]].corr()
print(corr_matrix["avg_price"].sort_values(ascending=False))


# In[32]:


# Moving Average maximum temperature과 maximum temperature의 시각화 
plt.figure(figsize=(16,4))
plt.plot(datas['max_temp'])
plt.plot(data['ma_max_temp'])
plt.ylabel("price")
plt.xlabel("years")
plt.title("moving average temperature")
plt.show()


# In[ ]:


# Moving average 컬럼들을 다시 plot 해보기.


# In[65]:


d["mms_temp_ma"] = mms.fit_transform(d[["ma_temp"]])
d["mms_rain_ma"] = mms.fit_transform(d[["ma_rain"]])
d["mms_price"] = mms.fit_transform(d[['avg_price']])


# In[66]:


fig, axs = plt.subplots(figsize=(18, 6))
plt.plot("mms_price", label='price', data=d)
plt.plot("mms_temp_ma", label='temperature', data=d)
plt.plot("mms_rain_ma", label='rainfall', data=d)
plt.ylabel("scaled")
plt.xlabel("years")
plt.title("Moving average and price")
plt.legend()
plt.show()


# In[97]:


# moving average를 구하느라 nan이 나온 5개월을 잘라주었고 필요한 컬럼들만 추출해 데이터 프레임을 만들었다.


# In[154]:


#datas = datas['2015-06-01':]
ma_data = datas[['avg_price', 'ma_max_temp', 'ma_rain']]


# In[136]:


ma = ma_data['2015-06-01':]


# In[155]:


ma.head()


# In[137]:


# 저장하기 
ma.to_csv('./datas/moving_data_final.csv')


# In[158]:


ma_data = pd.read_csv('./datas/moving_data_final.csv', index_col = "date")
ma_data.head()


# In[ ]:


# Random Forest Importance로 features의 중요도 보기 


# In[213]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[248]:


# 데이터를 살짝 바꿔주고 
d = datas.reset_index()
d = d[["avg_price","ma_max_temp", "ma_rain", "ma_min_temp", "ma_temp", "ma_field", "agr_price", "cons_price"]]
d.head()


# In[173]:


rf = RandomForestRegressor()


# In[249]:


X = d.drop("avg_price", axis='columns')
y = d[["avg_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=13)


# In[244]:


rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[245]:


rf.feature_importances_ 


# In[246]:


print(rf.score(X_train, y_train))


# In[255]:


feature_importances = pd.DataFrame(rf.feature_importances_, index = X_train.columns, 
                                  columns = ["importance"]).sort_values("importance", ascending=False)


# In[256]:


feature_importances


# In[276]:


plt.figure(figsize=(16,8))
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.title("feature importances")


# In[ ]:


# 농산 소비자 물가, 논밭이 가장 중요한 컬럼이라고 한다. 


# ### OLS 분석. 

# #### Minmax scaler, Standard scaler 비교하기

# In[365]:


d.head(1)


# In[514]:


# 스케일 하지 않은 데이터 
X = d.drop("avg_price", axis='columns')
y = d["avg_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)


# In[506]:


import statsmodels.api as sm
lm = sm.OLS(y_train, X_train).fit()


# In[498]:


plt.figure(figsize=(16,6))
sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value")
plt.legend()
plt.xlabel('test')
plt.ylabel('predicted')
plt.title('Non-scaled')
plt.show()


# In[499]:


# Minmax scaler, Standard scaler 

from sklearn.preprocessing import MinMaxScaler, StandardScaler

MMS = MinMaxScaler()
SS = StandardScaler()

X_ss = SS.fit_transform(X)
X_mms = MMS.fit_transform(X)


# In[500]:


# Standard Scaler로 핏 시킨 데이터로 학습 
X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)


# In[501]:


# OLS Regression statistics를 보았다. 
import statsmodels.api as sm
lm = sm.OLS(y_train, X_train).fit()


# In[502]:


# 그래프로 찍어보기. 
plt.figure(figsize=(16,6))
sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value")
plt.legend()
plt.xlabel('test value')
plt.ylabel('predicted value')
plt.title('SS_ test, prediction')
plt.show()


# In[503]:


from sklearn import metrics


# In[504]:


print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# #### MinMax

# In[505]:


# Minmax scaler로 fit 시킨 데이터
X_train, X_test, y_train, y_test = train_test_split(X_mms, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)


# In[506]:


import statsmodels.api as sm
lm = sm.OLS(y_train, X_train).fit()


# In[587]:


plt.figure(figsize=(16,6))
sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value")
plt.legend()
plt.xlabel('test value')
plt.ylabel('predicted value')
plt.title('MM_ test, prediction')
plt.show()


# In[508]:


print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[509]:


from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
model.score(X, y)


# In[510]:


# Minmax scaler를 쓰니까 MSE와 RMSE가 더 낮아졌다. MinMax scaler를 쓰는게 좋을 것 같다. 


# In[528]:


# 한 번에 보기 편하게 정리 

fig, ax = plt.subplots(3, 1, figsize=(18, 20))

#Normal graph 
X = d.drop("avg_price", axis='columns')
y = d["avg_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)

sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value", ax=ax[0])
ax[0].set(xlabel='test', ylabel='predicted', title='Non-scaled')
ax[0].text(3400,3000000, "MAE: {}\nMSE: {}\nRMSE: {}\nR2: {}".format(
                                       round(metrics.mean_absolute_error(y_test, predictions), 3),
                                       round(metrics.mean_squared_error(y_test, predictions),3),
                                       round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),3),
                                       round(r2_score(y_test, predictions),3)))

#Standard Scaler
X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)

sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value", ax=ax[1])
ax[1].set(xlabel='test', ylabel='predicted', title='Standard scaler')
ax[1].text(3400, -500, "MAE: {}\nMSE: {}\nRMSE: {}\nR2: {}".format(
                                       round(metrics.mean_absolute_error(y_test, predictions), 3),
                                       round(metrics.mean_squared_error(y_test, predictions),3),
                                       round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),3),
                                       round(r2_score(y_test, predictions),3)))

# MinMax Scaler
X_train, X_test, y_train, y_test = train_test_split(X_mms, y, test_size = 0.2, random_state=13)
predictions = lm.predict(X_test)

sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value", ax=ax[2])
ax[2].set(xlabel='test', ylabel='predicted', title='MinMax scaler')
ax[2].text(3400, 1500, "MAE: {}\nMSE: {}\nRMSE: {}\nR2: {}".format(
                                       round(metrics.mean_absolute_error(y_test, predictions), 3),
                                       round(metrics.mean_squared_error(y_test, predictions),3),
                                       round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),3),
                                       round(r2_score(y_test, predictions),3)))
plt.show()


# In[554]:


X = d.drop("avg_price", axis='columns')
y = d["avg_price"]

MMS = MinMaxScaler()

X = MMS.fit_transform(X)


# ### Linear Regression

# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


# 일단 OLS 모델을 보았다.
# avgPrice = βagrPrice + βmaField + βmaTemp + βmaRain 


# In[555]:


X_constant = sm.add_constant(X)


# In[556]:


pd.DataFrame(X_constant)
model = sm.OLS(y, X_constant)


# In[557]:


lr = model.fit()


# In[558]:


lr.summary()


# In[ ]:


# 어떤 컬럼을 써야 r2_adj가 잘 나올지 보았다 


# In[539]:


lm1 = smf.ols(formula='avg_price ~ ma_max_temp + ma_min_temp + ma_rain + ma_temp + ma_field + agr_price + cons_price',
              data=d).fit()
lm1.rsquared_adj


# In[540]:


lm1 = smf.ols(formula='avg_price ~ ma_max_temp + ma_rain', data=d).fit()
lm1.rsquared_adj


# In[541]:


lm1 = smf.ols(formula='avg_price ~ ma_min_temp + ma_rain', data=d).fit()
lm1.rsquared_adj


# In[475]:


# 과적합이 아닌지 판단하기 위해 train과 test의 MSE와 R2값을 구했습니다.
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

y_pred = model.predict(X_test)
print('R2: ', r2_score(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))


# In[479]:


# 과적합이 아닌지 판단하기 위해 train과 test의 MSE와 R2값을 구했습니다.


ols = linear_model.LinearRegression()
model = ols.fit(X, y)

y_pred = model.predict(X_train)
print('R2: ', r2_score(y_train, y_pred))
print('MSE: ', mean_squared_error(y_train, y_pred))


# In[559]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 

models = []
models.append(('RandomForest Regressor (평균 제곱근오차, MSE):', RandomForestRegressor()))
models.append(('Linear Regressor', LinearRegression()))


# In[563]:


# cross validation KFold
from sklearn.model_selection import KFold, cross_val_score, KFold

results = []
names = []

for name, model in models: 
    kfold = KFold(n_splits=5, random_state=13, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    
    print(name, cv_results.mean(), cv_results.std())


# In[ ]:


# Random forest를 보면 예측을 굉장히 잘 하는 것을 볼 수 있었다. 


# ### Random Forest Regression

# In[552]:


from sklearn.model_selection import GridSearchCV


# In[564]:


# Gridsearch CV를 통해 Best Parameter 찾고 fit 해서 모델을 튜닝해주기. 
random_forest_tuning = RandomForestRegressor(random_state = 13)
param_grid = {
    'n_estimators' : [100, 200, 500],
    'max_features' : ['auto', 'sqrt', 'log2'], 
    'max_depth' : [4,5,6,7,8],
    'criterion' : ['mse', 'mae'],
}
GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)
GSCV.fit(X_train, y_train)
GSCV.best_params_


# In[565]:


from math import sqrt
from sklearn.metrics import mean_absolute_error

random_forest = RandomForestRegressor(random_state = 13)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred)) # 평균절대오차: 실제값과 예측 값과의 차이. 모든 절대 오차의 평균
print('MSE: ', mean_squared_error(y_test, y_pred)) # 평균제곱 오차: 오차의 제곱에 대한 평균을 취한 값
print('RMSE: ', sqrt(mean_squared_error(y_test, y_pred)))


# In[566]:


# 만약 out of bag value가 높다면, 0.75가 넘으면 모델이 과적합이 아니라고 볼 수 있다. 
random_forest_out_of_bag = RandomForestRegressor(oob_score=True)
random_forest_out_of_bag.fit(X_train, y_train)
print(random_forest_out_of_bag.oob_score_)


# In[580]:


# 과적합이 아닌지 판단하기 위해 train과 test의 MSE와 R2값을 구했습니다.

random_forest = RandomForestRegressor(random_state = 13)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print('R2: ', r2_score(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))


# In[583]:


# 두 개가 비슷하면 과적합이 아니라고 결론을 낼 수 있습니다
random_forest = RandomForestRegressor(random_state = 13)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_train)

print('R2: ', r2_score(y_train, y_pred))
print('MSE: ', mean_squared_error(y_train, y_pred))


# In[576]:


# scatter plot으로 시각화. 꽤나 정확해보였다.
plt.fig = plt.subplots(figsize=(18, 12))
plt.scatter(X_test['ma_min_temp'].values, y_test, color = 'red', label="test")
plt.scatter(X_test['ma_min_temp'].values, y_pred, color = 'green', label="predicted")
plt.title('Random Forest Regression')
plt.xlabel('moving_temp')
plt.ylabel('Price')
plt.legend()
plt.show() 


# In[585]:


plt.figure(figsize=(12,6))
sns.regplot(y_test, predictions, line_kws={"color":"r","alpha":0.7,"lw":5}, label="predicted value")
plt.xlabel('test')
plt.ylabel('predicted')
plt.title('Random Forest ')
plt.plot(y_test, y_test, linestyle='--', color='r', lw=3, scalex=False, scaley=False)


# In[ ]:


# walk-forward validation
# https://www.youtube.com/watch?v=4rikgkt4IcU


# In[ ]:


# transform into supervised learning problem


# In[220]:


df = datas[['avg_price']].copy()


# In[221]:


df.head()


# In[222]:


df["target"] = df.avg_price.shift(-1)
df.head()


# In[223]:


df.dropna(inplace=True)


# In[224]:


def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


# In[225]:


train, test = train_test_split(df, 0.2)


# In[226]:


print(len(df))
print(len(train))
print(len(test))


# In[227]:


X = train[:, :-1]
y = train[:, -1]


# In[228]:


from xgboost import XGBRegressor


# In[229]:


model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X, y)


# In[230]:


test[0] #실제 데이터 


# In[231]:


val = np.array(test[0, 0]).reshape(1, -1)

pred = model.predict(val)
print(pred[0]) # 맞춘 데이터


# In[232]:


# Predict 


# In[233]:


def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.fit(X, y)
    
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]


# In[234]:


xgb_predict(train, test[0, 0])


# ### Walk forward Validation
# - Time series이기 때문에 walk forward validation으로 predict를 해보았다. 
# - price 값만 가지고 현재의 가격과 다음날의 가격을 더한 컬럼을 만들어서 다음 컬럼을 예측하는 것.

# In[ ]:


# walk-forward validation - one step forward prediction. 
# use train dataset, predict one step into the future, add that to trainset, retrain, predict second item, and add to trainset ...etc
# evaluate with RMSE metric


# In[241]:


from sklearn.metrics import mean_squared_error

def validate(data, perc):
    predictions = []
    
    train, test = train_test_split(data, perc)
    
    history = [x for x in train]
    
    for i in range(len(test)):
        test_X, test_y = test[i, :-1], test[i, -1]
        
        pred = xgb_predict(history, test_X[0])
        predictions.append(pred)
        
        history.append(test[i])
        
    error = mean_squared_error(test[:, -1], predictions, squared=False)
    
    return error, test[:, -1], predictions


# In[243]:


get_ipython().run_cell_magic('time', '', 'rmse, y, pred = validate(df, 0.2)\n\nprint(rmse)')


# In[249]:


# 타임 시리즈 데이터를 다루기 좋은 형태로 바꾸기
# https://machinelearningmastery.com/random-forest-for-time-series-forecasting/#:~:text=Random%20Forest%20can%20also%20be,a%20supervised%20learning%20problem%20first.&text=Random%20Forest%20is%20an%20ensemble,classification%20and%20regression%20predictive%20modeling.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

# train test split
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict([testX])
    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = random_forest_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# In[255]:


series = datas
values = series.values
data = series_to_supervised(values, n_in=6)
mae, y, yhat = walk_forward_validation(data, 12)
print('MAE: %.3f' % mae)


# In[251]:


from matplotlib import pyplot
# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()


# ## 소감

# <img src = './datas/다운로드.jpeg'>

# - 시계열의 한계
#     : 마지막에 보여드린 workfoward validation을 보면, 실제값과 예측값이 유연하게 맞아떨어지는 것을 알 수 있다. workfoward는 쉽게 말해
#     어제의 파 가격와 오늘의 파 가격 두 개의 컬럼으로 하여 시계열이라는 데이터의 특성을 이용해 데이터의 흐름을 예측하는 것으로, 보다시피 원활히 예측된다.
#     즉 해당 프로젝트에서 회귀분석 또한 분명히 의미 있는 분석이었지만, 나아가 시계열 예측 모델을 활용하면 회귀 분석에서의 한계를
#     상호보완 해 줄 수 있을 것으로 기대된다.
#     
#     따라서 이번 프로젝트는 여기서 그치지 않고, 시계열 모델을 별도로 학습하여 회귀와 시계열 두가지 방법을 비교하고,
#     일정 기간 동안의 평균 온도, 강수량 입력 -> 파 가격 OR 특정 날짜 입력 -> 파 가격 예측 과 같은 모델을 구현할 생각이다.  
#         
# 
# - 데이터의 부족
#   : heatmap을 그려보고, Randomforest importance도 써 봤을 떄 가장 유의미한 관계를 보여준 컬럼은 농수산물 물가, 물가, 재배면적이었는데, 
#     해당 데이터들은 월 단위, 년 단위로만 나와 있어서 비교적 유의미하게 쓰이지 못했다. 해당 데이터들이 일 단위로 집계되어 있었다면 보다 타당하고
#     신뢰도 높은 연구 결과를 도출할 수 있었을 것이다.
# 
# 
# - 많 ~~~ 은 공부가 되었다
#     : 모두가 염려하는 시계열과의 싸움에서 이기기 위해서 데이터 크롤링, 데이터 전처리에도 많은 시간을 썼고 회귀분석 또한 다각도로 깊이 있게 고민해 볼     수 있었다. 해당 자료에는 모두 담지 못하고 삭제된 부분(tensorflow, RMSLE 등..)이 너무나도 많지만 결과로 담지 못하였다고 하더라도 그 과정들은 모두 의미있었다.
#     결과의 성적 여부와는 별개로, 원하는 기획을 구현하기 위한 실력이 뒷받침 되도록 더욱 정진해야 겠다는 목표의식이 뚜렷해질 수 있었던 프로젝트였다.
