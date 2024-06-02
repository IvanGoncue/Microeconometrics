# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:56:04 2024

@author: GonCue
"""


import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

import numpy as np

from arch import arch_model

from statsmodels.tsa.api import VAR

import seaborn as sns


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


############################ SCRAPEAR DATOS DE YAHOO FINANCE


sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)

ticker = data_table[0]['Symbol'].tolist()

snp_prices = yf.download(ticker, start='2000-01-01', end='2024-01-01')['Adj Close']

#########################################################################3

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start='2000-01-01', end='2024-01-01')

# Seleccionar solo la columna 'Adj Close' (precios de cierre ajustados)
vix_prices = vix_data['Adj Close']

# Convertir los datos de fecha a una matriz de NumPy
dates = np.array(vix_data.index)

# Convertir los precios de cierre ajustados del VIX a una matriz de NumPy
vix_prices_array = np.array(vix_data['Adj Close'])

# Graficar los precios del VIX
plt.figure(figsize=(10, 6))
plt.plot(dates, vix_prices_array, color='blue')
plt.title('Precios de cierre ajustados del VIX (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.grid(True)
plt.show()

##############################################################################

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP) desde Yahoo Finance
gcsp_prices = yf.download('^GSPC', start='2000-01-01', end='2024-01-01')['Adj Close']

#############################################################################

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start='2000-01-01', end='2024-01-01')

print(tnx_data)

#^TNX: Rendimiento del Tesoro de EE.UU. a 10 años
#^IRX: Rendimiento del Tesoro de EE.UU. a 13 semanas
#^TYX: Rendimiento del Tesoro de EE.UU. a 30 años
#^FVX: Rendimiento del Tesoro de EE.UU. a 5 años

#################################################################################

import pandas_datareader.data as web
import datetime

# Definir las fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos de la tasa de fondos federales
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start, end)

print(fed_funds_rate)


#FRED (Federal Reserve Economic Data). El código especifica el uso del identificador 'FEDFUNDS' para la tasa de fondos federales.


##########################################################################################

# Descargar los datos de la tasa de descuento
discount_rate = web.DataReader('INTDSRUSM193N', 'fred', start, end)

print(discount_rate)

############################################################################################3

import pandas_datareader.data as web
import datetime

# Definir las fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

print(unemployment_rate)


import seaborn as sns

# Establecer el estilo de los gráficos
sns.set(style="darkgrid")

# Graficar los precios del VIX
plt.figure(figsize=(10, 6))
sns.lineplot(data=vix_data['Adj Close'], color='blue')
plt.title('Precios de cierre ajustados del VIX (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.show()

# Graficar los precios ajustados de cierre del índice S&P 500 (GCSP)
plt.figure(figsize=(10, 6))
sns.lineplot(data=gcsp_prices, color='green')
plt.title('Precios de cierre ajustados del S&P 500 (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.show()

# Graficar los datos del rendimiento del Tesoro de EE.UU. a 10 años
plt.figure(figsize=(10, 6))
sns.lineplot(data=tnx_data['Adj Close'], color='red')
plt.title('Rendimiento del Tesoro de EE.UU. a 10 años (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento')
plt.show()

# Graficar los datos de la tasa de fondos federales
plt.figure(figsize=(10, 6))
sns.lineplot(data=fed_funds_rate, color='orange')
plt.title('Tasa de fondos federales (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Tasa de interés')
plt.show()

# Graficar los datos de la tasa de descuento
plt.figure(figsize=(10, 6))
sns.lineplot(data=discount_rate, color='purple')
plt.title('Tasa de descuento (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Tasa de interés')
plt.show()

# Graficar los datos de la tasa de desempleo
plt.figure(figsize=(10, 6))
sns.lineplot(data=unemployment_rate, color='brown')
plt.title('Tasa de desempleo (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Tasa de desempleo')
plt.show()
















































#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

###################################### Buy and hold cruce de medias moviles

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir el símbolo del activo (por ejemplo, el S&P 500)
symbol = 'SPY'

# Descargar los datos históricos del activo
data = yf.download(symbol, start='2000-01-01', end='2024-01-01')

# Calcular las rentabilidades diarias
data['Daily Return'] = data['Adj Close'].pct_change()

# Estrategia 1: Buy and Hold
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

# Estrategia 2: Cruce de Medias Móviles
data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
data['SMA200'] = data['Adj Close'].rolling(window=200).mean()
data['Signal'] = 0
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)
data['Position'] = data['Signal'].diff()
data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Configuración de seaborn y matplotlib
sns.set(style='whitegrid')

# Gráfico 1: Rentabilidad de la Estrategia Buy and Hold
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Return'], color='blue')
plt.title('Rentabilidad de la Estrategia Buy and Hold')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

# Gráfico 2: Evolución del Activo y Estrategia de Cruce de Medias Móviles
plt.figure(figsize=(12, 6))
sns.lineplot(data=data[['Adj Close', 'SMA50', 'SMA200']], palette=['blue', 'green', 'purple'])
sns.scatterplot(data=data[data['Signal'] == 1], x=data[data['Signal'] == 1].index, y='SMA50', color='red', marker='^', label='Señal de Compra')
sns.scatterplot(data=data[data['Signal'] == -1], x=data[data['Signal'] == -1].index, y='SMA50', color='black', marker='v', label='Señal de Venta')
plt.title('Evolución del Activo y Estrategia de Cruce de Medias Móviles')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()

########################################### CRISIS

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir el símbolo del activo (por ejemplo, el S&P 500)
symbol = 'SPY'

# Descargar los datos históricos del activo
data = yf.download(symbol, start='2000-01-01', end='2024-01-01')

# Calcular las rentabilidades diarias
data['Daily Return'] = data['Adj Close'].pct_change()

# Estrategia 1: Buy and Hold
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

# Estrategia 2: Cruce de Medias Móviles
data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
data['SMA200'] = data['Adj Close'].rolling(window=200).mean()
data['Signal'] = 0
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)
data['Position'] = data['Signal'].diff()
data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Configuración de seaborn y matplotlib
sns.set(style='whitegrid')

# Gráfico 1: Rentabilidad de la Estrategia Buy and Hold
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Return'], color='blue')
plt.title('Rentabilidad de la Estrategia Buy and Hold')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

# Gráfico 2: Rentabilidad de la Estrategia de Cruce de Medias Móviles
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Strategy Return'], color='red')
plt.title('Rentabilidad de la Estrategia de Cruce de Medias Móviles')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

################################################ RENTABILIDADES

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir el símbolo del activo (por ejemplo, el S&P 500)
symbol = 'SPY'

# Descargar los datos históricos del activo
data = yf.download(symbol, start='2000-01-01', end='2024-01-01')

# Filtrar datos para el intervalo 2006-2009
data = data.loc['2006-01-01':'2009-12-31']

# Calcular las rentabilidades diarias
data['Daily Return'] = data['Adj Close'].pct_change()

# Estrategia 1: Buy and Hold
data['Cumulative Return Buy and Hold'] = (1 + data['Daily Return']).cumprod()

# Estrategia 2: Cruce de Medias Móviles
data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
data['SMA200'] = data['Adj Close'].rolling(window=200).mean()
data['Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, 0)
data['Position'] = data['Signal'].diff()
data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Configuración de seaborn y matplotlib
sns.set(style='whitegrid')

# Gráfico 1: Rentabilidad de la Estrategia Buy and Hold
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Return Buy and Hold'], color='blue')
plt.title('Rentabilidad de la Estrategia Buy and Hold (2006-2009)')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

# Gráfico 2: Rentabilidad de la Estrategia de Cruce de Medias Móviles
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Strategy Return'], color='red')
plt.title('Rentabilidad de la Estrategia de Cruce de Medias Móviles (2006-2009)')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()



################################################# CRUCE DE MEDIAS

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir el símbolo del activo (por ejemplo, el S&P 500)
symbol = 'SPY'

# Descargar los datos históricos del activo
data = yf.download(symbol, start='2000-01-01', end='2024-01-01')

# Filtrar datos para el intervalo 2006-2009
data = data.loc['2000-01-01':'2024-01-01']############################ CAMBIAR

# Calcular las rentabilidades diarias
data['Daily Return'] = data['Adj Close'].pct_change()

# Estrategia 1: Buy and Hold
data['Cumulative Return Buy and Hold'] = (1 + data['Daily Return']).cumprod()

# Estrategia 2: Cruce de Medias Móviles
data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
data['SMA200'] = data['Adj Close'].rolling(window=200).mean()
data['Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, 0)
data['Position'] = data['Signal'].diff()
data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Configuración de seaborn y matplotlib
sns.set(style='whitegrid')

# Gráfico 1: Rentabilidad de la Estrategia Buy and Hold
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Return Buy and Hold'], color='blue')
plt.title('Rentabilidad de la Estrategia Buy and Hold (2006-2009)')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

# Gráfico 2: Evolución del Activo y Estrategia de Cruce de Medias Móviles
plt.figure(figsize=(12, 6))
sns.lineplot(data=data[['Adj Close', 'SMA50', 'SMA200']], palette=['blue', 'green', 'purple'])
sns.scatterplot(data=data[data['Signal'] == 1], x=data[data['Signal'] == 1].index, y='SMA50', color='red', marker='^', label='Señal de Compra', s=200)
sns.scatterplot(data=data[data['Signal'] == -1], x=data[data['Signal'] == -1].index, y='SMA50', color='black', marker='v', label='Señal de Venta')
plt.title('Evolución del Activo y Estrategia de Cruce de Medias Móviles (2006-2009)')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()


# Gráfico: Rentabilidad de la Estrategia de Cruce de Medias Móviles
plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Cumulative Strategy Return'], color='red')
plt.title('Rentabilidad de la Estrategia de Cruce de Medias Móviles')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Acumulada')
plt.show()

######################################### AJUSTE DE MODELOS



########################################### VIX logit binario


import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt

# Definir fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP)
gcsp_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Combinar los datos en un DataFrame
data = pd.concat([vix_data, gcsp_data, tnx_data, unemployment_rate], axis=1)
data.columns = ['VIX', 'SP500_Return', 'TNX', 'Unemployment_Rate']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Definir la variable dependiente (elección)
# Por ejemplo, podríamos asumir que la elección es entre "alto VIX" y "bajo VIX"
data['High_VIX'] = (data['VIX'] > data['VIX'].median()).astype(int)

# Definir variables independientes
X = data[['SP500_Return', 'TNX', 'Unemployment_Rate']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_VIX'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    ax.scatter(X.iloc[:, i + 1], result.predict(), alpha=0.5)
    ax.set_title(f'Probability vs {X.columns[i + 1]}')
    ax.set_xlabel(X.columns[i + 1])
    ax.set_ylabel('Predicted Probability of High VIX')
    ax.axhline(y=data['High_VIX'].mean(), color='r', linestyle='--')
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

######################################################################3


import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2010, 1, 1)

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP)
gcsp_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Combinar los datos en un DataFrame
data = pd.concat([vix_data, gcsp_data, tnx_data, unemployment_rate], axis=1)
data.columns = ['VIX', 'SP500_Return', 'TNX', 'Unemployment_Rate']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Definir la variable dependiente (elección)
# Por ejemplo, podríamos asumir que la elección es entre "alto VIX" y "bajo VIX"
data['High_VIX'] = (data['VIX'] > data['VIX'].median()).astype(int)

# Definir variables independientes
X = data[['SP500_Return', 'TNX', 'Unemployment_Rate']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_VIX'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=X.iloc[:, i + 1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {X.columns[i + 1]}')
    ax.set_xlabel(X.columns[i + 1])
    ax.set_ylabel('Predicted Probability of High VIX')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(X.iloc[:, i + 1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=X.iloc[:, i + 1], y=lr.predict(X.iloc[:, i + 1].values.reshape(-1, 1)), ax=ax, color='blue')
    
    ax.axhline(y=data['High_VIX'].mean(), color='r', linestyle='--')
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()


# Visualizar los datos originales con Seaborn
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
sns.lineplot(data=data['VIX'], label='VIX')
plt.title('VIX')

plt.subplot(4, 1, 2)
sns.lineplot(data=data['SP500_Return'], label='S&P 500 Returns')
plt.title('S&P 500 Returns')

plt.subplot(4, 1, 3)
sns.lineplot(data=data['TNX'], label='10-Year Treasury Yield')
plt.title('10-Year Treasury Yield')

plt.subplot(4, 1, 4)
sns.lineplot(data=data['Unemployment_Rate'], label='Unemployment Rate', color='orange')
plt.title('Unemployment Rate')

plt.tight_layout()
plt.show()




















import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2014, 1, 1)

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP)
gcsp_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start, end)

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Combinar los datos en un DataFrame
data = pd.concat([vix_data, gcsp_data, fed_funds_rate, unemployment_rate], axis=1)
data.columns = ['VIX', 'SP500_Return', 'FEDFUNDS', 'Unemployment_Rate']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Definir la variable dependiente (elección)
# Por ejemplo, podríamos asumir que la elección es entre "alto VIX" y "bajo VIX"
data['High_VIX'] = (data['VIX'] > data['VIX'].median()).astype(int)

# Definir variables independientes
X = data[['SP500_Return', 'FEDFUNDS', 'Unemployment_Rate']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_VIX'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=X.iloc[:, i + 1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {X.columns[i + 1]}')
    ax.set_xlabel(X.columns[i + 1])
    ax.set_ylabel('Predicted Probability of High VIX')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(X.iloc[:, i + 1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=X.iloc[:, i + 1], y=lr.predict(X.iloc[:, i + 1].values.reshape(-1, 1)), ax=ax, color='blue')
    
    ax.axhline(y=data['High_VIX'].mean(), color='r', linestyle='--')
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()


# Visualizar los datos originales con Seaborn
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
sns.lineplot(data=data['VIX'], label='VIX')
plt.title('VIX')

plt.subplot(4, 1, 2)
sns.lineplot(data=data['SP500_Return'], label='S&P 500 Returns')
plt.title('S&P 500 Returns')

plt.subplot(4, 1, 3)
sns.lineplot(data=data['FEDFUNDS'], label='FEDFUNDS')
plt.title('FEDFUNDS')

plt.subplot(4, 1, 4)
sns.lineplot(data=data['Unemployment_Rate'], label='Unemployment Rate', color='orange')
plt.title('Unemployment Rate')

plt.tight_layout()
plt.show()























########################################################################33

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt

# Definir fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP)
gcsp_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de descuento
discount_rate = web.DataReader('INTDSRUSM193N', 'fred', start, end)

# Combinar los datos en un DataFrame
data = pd.concat([vix_data, gcsp_data, tnx_data, discount_rate], axis=1)
data.columns = ['VIX', 'SP500_Return', 'TNX', 'Discount_Rate']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Definir la variable dependiente (elección)
# Por ejemplo, podríamos asumir que la elección es entre "alto VIX" y "bajo VIX"
data['High_VIX'] = (data['VIX'] > data['VIX'].median()).astype(int)

# Definir variables independientes
X = data[['SP500_Return', 'TNX', 'Discount_Rate']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_VIX'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    ax.scatter(X.iloc[:, i + 1], result.predict(), alpha=0.5)
    ax.set_title(f'Probability vs {X.columns[i + 1]}')
    ax.set_xlabel(X.columns[i + 1])
    ax.set_ylabel('Predicted Probability of High VIX')
    ax.axhline(y=data['High_VIX'].mean(), color='r', linestyle='--')
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

##########################################################################

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP)
gcsp_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de descuento
discount_rate = web.DataReader('INTDSRUSM193N', 'fred', start, end)

# Combinar los datos en un DataFrame
data = pd.concat([vix_data, gcsp_data, tnx_data, discount_rate], axis=1)
data.columns = ['VIX', 'SP500_Return', 'TNX', 'Discount_Rate']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Definir la variable dependiente (elección)
# Por ejemplo, podríamos asumir que la elección es entre "alto VIX" y "bajo VIX"
data['High_VIX'] = (data['VIX'] > data['VIX'].median()).astype(int)

# Definir variables independientes
X = data[['SP500_Return', 'TNX', 'Discount_Rate']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_VIX'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha con Seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=X.iloc[:, i + 1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {X.columns[i + 1]}')
    ax.set_xlabel(X.columns[i + 1])
    ax.set_ylabel('Predicted Probability of High VIX')
    ax.axhline(y=data['High_VIX'].mean(), color='r', linestyle='--')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(X.iloc[:, i + 1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=X.iloc[:, i + 1], y=lr.predict(X.iloc[:, i + 1].values.reshape(-1, 1)), ax=ax, color='blue')

plt.tight_layout()
plt.show()


























































########################################### sp500 logit binario


import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos del índice S&P 500
sp500_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos del rendimiento del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de descuento
discount_rate = web.DataReader('INTDSRUSM193N', 'fred', start, end)

# Descargar los datos del VIX para calcular la variable dependiente
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Combinar los datos en un DataFrame
data = pd.concat([sp500_data, tnx_data, discount_rate, vix_data], axis=1)
data.columns = ['SP500_Return', 'TNX', 'Discount_Rate', 'VIX']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Calcular la variable dependiente (elección) en base al S&P 500
data['High_SP500'] = (data['SP500_Return'] > data['SP500_Return'].median()).astype(int)

# Definir variables independientes
X = data[['TNX', 'Discount_Rate', 'VIX']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_SP500'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha con Seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=data.iloc[:, i+1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {data.columns[i+1]}')
    ax.set_xlabel(data.columns[i+1])
    ax.set_ylabel('Predicted Probability of High S&P 500')
    ax.axhline(y=data['High_SP500'].mean(), color='r', linestyle='--')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(data.iloc[:, i+1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=data.iloc[:, i+1], y=lr.predict(data.iloc[:, i+1].values.reshape(-1, 1)), ax=ax, color='blue')

plt.tight_layout()
plt.show()

################################################################

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2014, 1, 1)

# Descargar los datos del índice S&P 500
sp500_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de fondos federales
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start, end)

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Descargar los datos del VIX para calcular la variable dependiente
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Combinar los datos en un DataFrame
data = pd.concat([sp500_data, fed_funds_rate, unemployment_rate, vix_data], axis=1)
data.columns = ['SP500_Return', 'Fed_Funds_Rate', 'Unemployment_Rate', 'VIX']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Calcular la variable dependiente (elección) en base al S&P 500
data['High_SP500'] = (data['SP500_Return'] > data['SP500_Return'].median()).astype(int)

# Definir variables independientes
X = data[['Fed_Funds_Rate', 'Unemployment_Rate', 'VIX']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Logit
model = Logit(data['High_SP500'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha con Seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=data.iloc[:, i+1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {data.columns[i+1]}')
    ax.set_xlabel(data.columns[i+1])
    ax.set_ylabel('Predicted Probability of High S&P 500')
    ax.axhline(y=data['High_SP500'].mean(), color='r', linestyle='--')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(data.iloc[:, i+1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=data.iloc[:, i+1], y=lr.predict(data.iloc[:, i+1].values.reshape(-1, 1)), ax=ax, color='blue')

plt.tight_layout()
plt.show()













################################################################# PROBIT
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definir fechas de inicio y fin
start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2014, 1, 1)

# Descargar los datos del índice S&P 500
sp500_data = yf.download('^GSPC', start=start, end=end)['Adj Close']

# Descargar los datos de la tasa de fondos federales
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start, end)

# Descargar los datos de la tasa de desempleo
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Descargar los datos del VIX para calcular la variable dependiente
vix_data = yf.download('^VIX', start=start, end=end)['Adj Close']

# Combinar los datos en un DataFrame
data = pd.concat([sp500_data, fed_funds_rate, unemployment_rate, vix_data], axis=1)
data.columns = ['SP500_Return', 'Fed_Funds_Rate', 'Unemployment_Rate', 'VIX']

# Eliminar filas con valores NaN (en caso de que haya)
data = data.dropna()

# Calcular la variable dependiente (elección) en base al S&P 500
data['High_SP500'] = (data['SP500_Return'] > data['SP500_Return'].median()).astype(int)

# Definir variables independientes
X = data[['Fed_Funds_Rate', 'Unemployment_Rate', 'VIX']]

# Agregar una constante para el término independiente
X = add_constant(X)

# Ajustar el modelo Probit
model = Probit(data['High_SP500'], X)
result = model.fit()

# Imprimir los resultados del modelo
print(result.summary())

# Gráficos de dispersión y probabilidad predicha con Seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop para trazar cada variable independiente contra la probabilidad predicha
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=data.iloc[:, i+1], y=result.predict(), ax=ax, alpha=0.5)
    ax.set_title(f'Probability vs {data.columns[i+1]}')
    ax.set_xlabel(data.columns[i+1])
    ax.set_ylabel('Predicted Probability of High S&P 500')
    ax.axhline(y=data['High_SP500'].mean(), color='r', linestyle='--')
    
    # Ajuste de una línea de regresión
    lr = LinearRegression()
    lr.fit(data.iloc[:, i+1].values.reshape(-1, 1), result.predict())
    sns.lineplot(x=data.iloc[:, i+1], y=lr.predict(data.iloc[:, i+1].values.reshape(-1, 1)), ax=ax, color='blue')

plt.tight_layout()
plt.show()


















































##################################### eleccion bonos, acciones


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf
import datetime

# Descargar los datos del S&P 500 desde yfinance
snp_data = yf.download('^GSPC', start='2000-01-01', end='2024-01-01')
snp_returns = snp_data['Adj Close'].pct_change().fillna(0)

# Descargar los datos del bono del Tesoro de EE.UU. a 10 años
tnx_data = yf.download('^TNX', start='2000-01-01', end='2024-01-01')
tnx_returns = tnx_data['Adj Close'].pct_change().fillna(0)

# Preprocesamiento de datos
data = pd.DataFrame({
    'sp500_return': snp_returns,
    'tnx_return': tnx_returns,
})

# Eliminar filas con valores NaN
data.dropna(inplace=True)

# Definir la variable objetivo basada en la comparación de rentabilidades
data['choice'] = np.where(data['sp500_return'] > data['tnx_return'], 'acciones', 'bonos')

# Definir las características y la variable objetivo
X = data[['sp500_return', 'tnx_return']]
y = data['choice']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir y entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



import matplotlib.pyplot as plt

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Crear el gráfico de dispersión
plt.figure(figsize=(8, 6))

# Puntos clasificados correctamente
plt.scatter(X_test[y_pred == 'acciones']['sp500_return'], X_test[y_pred == 'acciones']['tnx_return'], c='blue', label='Acciones (Predicción correcta)')
plt.scatter(X_test[y_pred == 'bonos']['sp500_return'], X_test[y_pred == 'bonos']['tnx_return'], c='red', label='Bonos (Predicción correcta)')

# Puntos clasificados incorrectamente
plt.scatter(X_test[y_pred != y_test]['sp500_return'], X_test[y_pred != y_test]['tnx_return'], c='yellow', label='Predicción incorrecta')

# Límites y etiquetas
plt.xlabel('Retorno del S&P 500')
plt.ylabel('Retorno del bono del Tesoro a 10 años')
plt.title('Predicciones del modelo')
plt.legend()
plt.grid(True)

plt.show()



from sklearn.preprocessing import LabelEncoder  # Importación de LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score


# Obtener las probabilidades predichas por el modelo para los datos de prueba
y_probs = model.predict_proba(X_test)

# Codificar las etiquetas de clase
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs[:, 1])

# Calcular el área bajo la curva ROC (AUC)
auc = roc_auc_score(y_test_encoded, y_probs[:, 1])

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()





































################################################# variable instrumental

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm

# Definir las fechas de inicio y fin
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Descargar los datos necesarios
snp_prices = yf.download('^GSPC', start=start, end=end)['Adj Close']
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start, end)
unemployment_rate = web.DataReader('UNRATE', 'fred', start, end)

# Calcular los rendimientos diarios
snp_returns = snp_prices.pct_change().dropna()

# Alinear las series temporales
data = pd.DataFrame({
    'snp_return': snp_returns,
    'unemployment_rate': unemployment_rate['UNRATE'],
    'fed_funds_rate': fed_funds_rate['FEDFUNDS']
}).dropna()

# Definir las variables dependiente, endógena y el instrumento
y = data['snp_return']
X = data[['unemployment_rate']]
Z = data[['fed_funds_rate']]

# Agregar constante al modelo
X = sm.add_constant(X)
Z = sm.add_constant(Z)

# Ajustar el modelo de variable instrumental
iv_model = IV2SLS(y, X, Z).fit()

# Resumen de los resultados
print(iv_model.summary())





# Visualizar los datos originales con Seaborn
plt.figure(figsize=(14, 7))
plt.subplot(3, 1, 1)
sns.lineplot(data=data['snp_return'], label='S&P 500 Returns')
plt.title('S&P 500 Returns')

plt.subplot(3, 1, 2)
sns.lineplot(data=data['unemployment_rate'], label='Unemployment Rate', color='orange')
plt.title('Unemployment Rate')

plt.subplot(3, 1, 3)
sns.lineplot(data=data['fed_funds_rate'], label='Fed Funds Rate', color='green')
plt.title('Fed Funds Rate')

plt.tight_layout()
plt.show()


# Residuos del modelo con Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=iv_model.resid, markers=True)
plt.hlines(0, data.index.min(), data.index.max(), colors='r', linestyles='--')
plt.title('Residuals of the IV Model')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Visualización de la relación entre las variables manualmente
plt.figure(figsize=(12, 8))

# Gráfico de dispersión para la relación entre 'snp_return' y 'unemployment_rate' con línea de regresión
plt.subplot(2, 2, 1)
sns.regplot(x=data['unemployment_rate'], y=data['snp_return'])
plt.title('S&P 500 Returns vs Unemployment Rate')

# Gráfico de dispersión para la relación entre 'snp_return' y 'fed_funds_rate' con línea de regresión
plt.subplot(2, 2, 2)
sns.regplot(x=data['fed_funds_rate'], y=data['snp_return'])
plt.title('S&P 500 Returns vs Fed Funds Rate')

# Gráfico de dispersión para la relación entre 'unemployment_rate' y 'fed_funds_rate' con línea de regresión
plt.subplot(2, 2, 3)
sns.regplot(x=data['unemployment_rate'], y=data['fed_funds_rate'])
plt.title('Unemployment Rate vs Fed Funds Rate')

plt.tight_layout()
plt.show()








































