# Projeto DL Forecasting 15 Dias Volatilidade - PETR4
____

__Bussines Problem:__
> Durante a rotina de investimentos torna-se importante a estimação da volatilidade tanto para estimar a variação de portfólios ou para operações estruturadas de compra ou venda de volatilidade, sendo necessário o desenvolvimento de modelos quantitativos para o forecasting futuro de movimentos de alta ou baixa de volatilidade.

__Objetivo:__   

> Desenvolver um modelo quantitativo de Deep Learning, utilizando o framework keras-tensorflow para o forecasting da volatilidade em 15 dias do ativo PETR4 listado na bolsa de valores brasileira B3, o modelo deve estimar a regressão da volatilidade futura.

__Autor:__  
   - Bruno Priantti.
    
__Contato:__  
  - bpriantti@gmail.com

__Encontre-me:__  
   -  https://www.linkedin.com/in/bpriantti/  
   -  https://github.com/bpriantti
   -  https://www.instagram.com/brunopriantti/
   
__Frameworks Utilizados:__

- Numpy: https://numpy.org/doc/  
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/ 
- Seaborn: https://seaborn.pydata.org/  
- Plotly: https://plotly.com/  
- Scikit learn: https://scikit-learn.org/stable/index.html
- Statsmodels: https://www.statsmodels.org/stable/index.html  
- https://keras.io/

___
## Contents:
 - [Data Request e Data Wralling Base de Dados](#data-request-e-data-wralling-base-de-dados) 
 - [Treinamento do Modelo](#treinamento-do-modelo)
 - [Resultado Treinamento e Teste dos Modelos](#resultado-treinamento-e-teste-dos-modelos)
 - [Conclusões e Trabalhos Futuros](#conclusões-e-trabalhos-futuros)

 
### Data Request e Data Wralling Base de Dados:

> Para este projeto optou-se por utilizar a metodologia de forward forecasting que consiste em cross validation para séries temporais em que se treina-testa o modelo em dados in-sample e dados out-of-sample, o ativo escolhido para este projeto foi a empresa do ramo de extração de petróleo chamada petrobras com o ticker PETR4, optou-se por utilizar a base de dados histórica do período de 2001 a 2021, utilizou-se o provedor de dados yfinance, realizou-se o request dos dados com o script de códigos abaixo:

```python
#download base de dados:
ticker = "PETR4.SA"

#datas - atentar para inicio em janeiro do ano de inicio do ativo ou proximo.
start =  "2011-01-01" 
end =    "2021-12-31"

#API - yahoo finance:
data = yf.download(ticker, start, end)
```

> Em seguida realizou-se o processo de inspeção da base dados realizando uma visualização da série histórica e em seguida verificando se os mesmos foram baixados corretamente do provedor, sem outliers ou inconformidades visíveis, para este projeto utiliza-se apenas os dados de fechamento ajustado.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/image-01.png?raw=true"  width="760" height = "400">

> Calculou-se para a base de dados o log-return para a série de fechamentos e também realizou-se o processo de split dos dados, em 4 steps estes que posteriormente servirão para a divisão entre treinamento e teste para o treinamento e teste do modelo.
   
```python
# calc returns:
data['log_returns'] = np.log(data.Close/data.Close.shift(1))

#---:
data['vol_15'] = data['log_returns'].rolling(16).std()

#---:
data = data[['log_returns','vol_15']].copy()
data.dropna(inplace = True)

#data split:
step_1 = data.loc['2011':'2017']
step_2 = data.loc['2012':'2018']
step_3 = data.loc['2013':'2019']
step_4 = data.loc['2014':'2020']   
```

### Treinamento do Modelo:
   
> Para este projeto optou-se por realizar o treinamento do modelo em steps, como demonstrado na imagem abaixo, dividindo o dataset step em treinamento e step para cada step como demonstrado na imagem abaixo.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/image-02.PNG?raw=true">
   
__Processo realizado para cada step:__
```   
   - Dividindo o bloco de dados (step) em treinamento e teste.
   - Preparando os dados para o treinamento de modelo.
   - Config rnn - lstm.
   - Treinamento do modelo.
   - Teste do modelo.
   - Avaliando métricas.
   - Salvando informações do step atual.
```
   
> Realizou-se este processo até o fim dos dados no caso 4 steps, em seguida armazenou-se os dados de cada step em um dicionário e em seguida um arquivo txt para posterior análise de todo o conjunto de predições.
   

__Script Pré-Processamento dos Dados/Treinamento do Modelo:__ 
```python
#---:
#normalizacao dos dados para compatilibilidade com o framework tensorflow
trainingd = data_train['vol_15'].values.reshape(-1,1)
testingd = data_test['vol_15'].values.reshape(-1,1)

#---:
#padronizacao dos dados, para compatibilidade com o framework tensorflow
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(trainingd)

#---:
#processo de vetorizacao dos dados:
x_train = []
y_train = []

#---:
#set windown size em 45 janelas:
timestamp = 45
length = len(trainingd)

for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])

#---:
x_train = np.array(x_train)
y_train = np.array(y_train)

#---:
#compatibilidade para input do modelo:
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#---:
#input sistema de redes neurais recorrentes - LSTM
model = Sequential() 

model.add(LSTM(units = 120, return_sequences = True, input_shape = (x_train.shape[1], 1))) 
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = False)) 
model.add(Dropout(0.2))
model.add(Dense(units = 1)) 
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#treinando o modelo:
model.fit(x_train, y_train, epochs = 30, batch_size = 32)

#---:
#aplicando o modelo treinado para base de dados desconhecidos, teste.
test_set_scaled = sc.transform(testingd)

#---:
x_test = []
y_test = []

#---:
timestamp = 45
length = len(testingd)

#---:
for i in range(timestamp, length):
    x_test.append(test_set_scaled[i-timestamp:i, 0])
    y_test.append(test_set_scaled[i, 0])

#---:
x_test = np.array(x_test)
y_test = np.array(y_test)

#---:
#realizando predict do modelo:
y_pred = model.predict(x_test)

#---:
#armazenando dados em dataframe:
y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test.reshape(-1,1))

y_pred.columns = ['sinal']
y_test.columns = ['sinal']

```
__Script Avaliação do Modelo(Métricas Regressão):__ 
```python
#---:
#avaliando o modelo e metricas:
import statsmodels.api as sm
from sklearn import metrics

MAE_test_sm = sm.tools.eval_measures.meanabs(y_test['sinal'], y_pred['sinal']) 
RMSE_test_sm = sm.tools.eval_measures.rmse(y_test['sinal'], y_pred['sinal'])

MAE = round(MAE_test_sm, 3)
RMSE = round(RMSE_test_sm, 3)
VOL_BASE = round(y_test['sinal'].mean(),3)
RATIO_MAE = round(metrics.mean_absolute_error(y_test['sinal'], y_pred['sinal'])/y_test['sinal'].mean()*100, 2)

eval = [MAE,RMSE,VOL_BASE,RATIO_MAE]

print("")
print("----- Avaliação do teste -----")
print('MAE:  ', MAE)
print('RMSE: ', RMSE )
print("")
print("A volatilidade média da base é: ")
print(VOL_BASE)
print()
print("O percentual do MAE em relaçao à média da base: ")
print(RATIO_MAE)      
   
```
   
### Resultado Treinamento e Teste dos Modelos:
   

__step 01:__  
   
- Performance Modelo Para Base de Teste:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step1.png?raw=true" width="760" height = "400">
         
- Resultado Grafico de QQ-plot:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step1b.png?raw=true" width="760" height = "400">
   
- Metricas de Avaliacao Regressao:

   Métrica           | Valor
   ---------         | ------
   MAE               | 0.029
   RMSE              | 0.052
   STD_AMOSTRA       | 0.264
   % MAE/STD_AMOSTRA | 11.11 
   
__step 02:__  

- Performance Modelo Para Base de Teste:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step2.png?raw=true" width="760" height = "400">
         
- Resultado Grafico de QQ-plot:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step2b.png?raw=true" width="760" height = "400">
   
- Métricas de Avaliação Regressão:

   Métrica           | Valor
   ---------         | ------
   MAE               | 0.033
   RMSE              | 0.054
   STD_AMOSTRA       | 0.372
   % MAE/STD_AMOSTRA | 8.94 
   
__step 03:__  

- Performance Modelo Para Base de Teste:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step3.png?raw=true" width="760" height = "400">
         
- Resultado Grafico de QQ-plot:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step3b.png?raw=true" width="760" height = "400">
   
- Métricas de Avaliação Regressão:

   Métrica           | Valor
   ---------         | ------
   MAE               | 0.014
   RMSE              | 0.022
   STD_AMOSTRA       | 0.136
   % MAE/STD_AMOSTRA | 10.08 
  
__step 04:__  

- Performance Modelo Para Base de Teste:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step4.png?raw=true" width="760" height = "400">
         
- Resultado Grafico de QQ-plot:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/step4b.png?raw=true" width="760" height = "400">
   
- Métricas de Avaliação Regressão:

   Metrica           | Valor
   ---------         | ------
   MAE               | 0.039
   RMSE              | 0.074
   STD_AMOSTRA       | 0.398
   % MAE/STD_AMOSTRA | 9.86  
___
   
### Conclusões e Trabalhos Futuros:

Podemos concluir que os modelos desenvolvidos com base nas métricas de avaliação apresentaram performances satisfatórias nos horizontes de teste mostrando a eficiência dos modelos de lstm para forecasting de volatilidade em mercados financeiros, Neste projeto não realizou-se o processo de tratamento e modificações dos layers das rnn, em projetos futuros implementar este processo pode ser interessante para a melhoria dos resultados, outro trabalho futuro seria realizar o deploy do modelo e criar uma api juntamente com uma interface para realizar operações estruturadas de compra e venda de volatilidade.

