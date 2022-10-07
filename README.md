# Projeto DL Forecasting 15 Dias Volatilidade - PETR4
____

__Bussines Problem:__
> Durante a rotina de investimentos torna-se importante a estimação da volatilidade tanto para estimar a variação de portfólios ou para operações estruturadas de compra ou venda de volatilidade, sendo necessário o desenvolvimento de modelos quantitativos para o forecasting futuro de movimentos de alta ou baixa de volatilidade.

__Objetivo:__   

> Desenvolver um modelo quantitativo de Deep Learning, utilizando o framework sklearn, statsmodels para o forecasting da volatilidade em 15 dias do ativo PETR4 listado na bolsa de valores brasileira B3, utilizando o framework keras-tensor-flow.

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
 - [Metodologia Foward Forecasting](#metodologia-foward-forecasting) 
 - [Data Request](#data-request) 
 -
 
### Metodologia Foward Forecasting:

> Para este projeto optou-se por utilizar a metodologia de foward forecasting que consiste em cross validation para series temporais em que se treina-testa o modelo em dados in-sample e dados out-of-sample, o ativo escolhido para este projeto foi a empresa do ramo de extracao de petroleo chamada petrobras com o ticker PETR4, optou-se por utilizar a base de dados historica do periodo de 2001 a 2021, utilizou-se o provedor de dados yfinance, realizou-se o request dos dados com o script de codigos abaixo:

```
#download base de dados:
ticker = "PETR4.SA"

#datas - atentar para inicio em janeiro do ano de inicio do ativo ou proximo.
start =  "2011-01-01" 
end =    "2021-12-31"

#API - yahoo finance:
data = yf.download(ticker, start, end)
```

> Em seguida realizou-se o processo de inpecao da base dados verificando se os mesmos foram baixados corretamente do provedor, sem outliers ou incoformidades visiveis, para este projeto utiliza-se apenas os dados de fechamento ajustado.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_dl_regressor_time_series_volatilidade_em_15_dias_PETR4/blob/main/images/image-01.png?raw=true"  width="800" height = "400">


#processo por step
#processo completo
