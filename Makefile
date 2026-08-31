PY=python
SRC_PATH=src

# Roda a inferencia, o interesse do trabalho
all:
	$(PY) $(SRC_PATH)/inferencia.py

# Comando para executar o scrap de dados dos sites selecionados
scraper:
	$(PY) $(SRC_PATH)/webscraper.py

# Comando para executar o arquivo de treino.
# Este arquivo salva as colunas do modelo, o encoder, o modelo e o scaler na pasta joblib
train: dados.csv
	$(PY) $(SRC_PATH)/treino.py

# Comando para executar o arquivo de inferencia
# Para executar, necessita apenas que exista os seguintes arquivos na pasta joblib: colunas_modelo.joblib, encoder.joblib, modelo.joblib, scaler.joblib
inf: entrada.csv
	$(PY) $(SRC_PATH)/inferencia.py

# Comando para limpar TODOS os arquivos do joblib
clean: 
	rm -f joblib/*joblib