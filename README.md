# 2026-1-projeto-grupo-5
## Membros: Aline Manhães, Marcela Carpenter e Raony Togneri

Este repositório contém o material completo do projeto de Machine Learning focado em prever os valores de aluguéis de imóveis na região da Grande Vitória. O projeto engloba desde a coleta e limpeza dos dados até o treinamento e inferência utilizando o algoritmo XGBoost.

## Estrutura do Repositório

Para garantir a reprodutibilidade e organização, os arquivos exigidos estão distribuídos da seguinte forma:

* *`dados.csv`*: Contém o **dataset coletado** original, gerado, processado e limpo pelo `webscraper.py`, utilizado pelo treinamento em `treino.py`.
* `webscraper.py`: **Scripts de scraping**, responsáveis por ler os dados da Olx e do NetImoveis disponíveis na internet, limpar e tratar esses dados e disponibilizá-los para o treinamento.
* `treino.py`: **Scripts de treinamento**, responsáveis por carregar os dados, aplicar as transformações (como Target Encoding e One Hot Encoding), escolher os melhores parâmetros do modelo com RandomizedSearchCV, treinar o modelo e exportá-lo.
* `inferencia.py`: **Scripts de inferência**, responsáveis por carregar o modelo treinado e realizar novas predições em dados inéditos disponibilizados pelo professor em `entrada.csv`. Gera o arquivo `resultado.csv` com valores preditos pelo modelo.
* `imovel.py`: Classe de um Imovel e funções utilitárias de limpeza e formatação de texto (normalização de bairros, etc).
* `const.py`: Arquivo com variáveis úteis para todos os scripts, como path de arquivos e links para o scraping.
* *`/joblib/`*: Diretório onde o **modelo treinado final, encoder, scaler, colunas do treino e os imputador de área, quartos e banheiros** (ex: `modelo.joblib`) são salvos após a execução do script de treino `treino.py`. É desse diretório que `inferencia.py` recupera informações geradas no treinamento.
* *`requirements.txt`*: Lista com todas as dependências e bibliotecas necessárias para executar o projeto.
* *`/graficos/`*: Diretório onde o gráficos gerados na execução dos scripts são salvos.


---

## Instruções de Execução

Siga os passos abaixo para configurar o ambiente e executar os scripts de treinamento e inferência na sua máquina.

### 1. Pré-requisitos
Certifique-se de ter o Python 3.10 (ou superior) instalado. É altamente recomendável utilizar um ambiente virtual (venv) para isolar as bibliotecas deste projeto.

Abra o terminal e execute os seguintes comandos:

Clone o repositório privado (caso ainda não tenha feito)

```bash
git clone https://github.com/ia-ufes/2026-1-projeto-grupo-5.git
cd 2026-1-projeto-grupo-5
```
Crie e ative um ambiente virtual (Linux/Mac)
```bash
python -m venv venv
source venv/bin/activate
```
Instale todas as dependências necessárias listadas no arquivo requirements.txt
```bash
pip install -r requirements.txt
```
### 2. Scraping
Os dados já foram lidos da web via scraping e estão em `dados.csv`, mas se você desejar ler novos dados disponíveis, basta executar o comando abaixo.
```bash
make scraper
```
### 3. Treinamento
O modelo de predição já foi treinado e está disponivel em `/joblib/`, mas se você desejar replicar os resultados obtidos, basta executar o comando abaixo.
```bash
make train
```
### 4. Inferência
Para realizar a inferência dos dados presentes em `entrada.csv`, basta executar o comando abaixo.
```bash
make inf
```
Ou o seguinte comando
```bash
python src/inferencia.py
```
Isso gerará o arquivo `resultado.csv` com os valores preditos pelo modelo.

## Dicionário de Dados (Atributos de Entrada)

Para que os scripts de inferência funcionem corretamente, o dataset (`entrada.csv`) deve conter os seguintes atributos claramente definidos:

* **`id`**: identificação única da amostra (imóvel);
* **`cidade`**: cidade onde o imóvel está localizado (ex: Vitória, Vila Velha);
* **`bairro`**: bairro onde o imóvel está localizado (ex: Praia do Canto, Jardim da Penha);
* **`tipo`**: tipo do imóvel, sendo 0 para "casa" e 1 para "apartamento";
* **`area`**: área total em m² do imóvel;
* **`quartos`**: número de quartos do imóvel;
* **`banheiros`**: número de banheiros do imóvel;
* **`vagas`**: número de vagas de garagem disponíveis;
* **`elevador`**: indica a presença de elevador no prédio (1 para sim, 0 para não);
* **`porteiro`**: indica a presença de portaria/porteiro (1 para sim, 0 para não);
* **`area_lazer`**: indica se o imóvel/condomínio possui área de lazer, equivalente a churrasqueira, salão de festas, piscina ou quadra (1 para sim, 0 para não);
* **`academia`**: indica se o imóvel/condomínio possui academia (1 para sim, 0 para não);
* **`varanda`**: indica se o imóvel/condomínio possui varanda (1 para sim, 0 para não);
* **`mobiliado`**: indica se o imóvel é alugado já mobiliado (1 para sim, 0 para não).

## Participação individual no trabalho
1. **`Aline Manhães`**: desenvolvimento da inferência e do treino, treinando os modelos e descobrindo os melhores parâmetros encontrados para tais.
2. **`Marcela Carpenter`**: desenvolvimento do treino com validação cruzada, estratificação das amostras e tratamento dos outliers.
3. **`Raony Togneri`**: desenvolvimento da coleta e tratamento dos dados lidos ou ignorados, auxílio com testes e de funcionamento dos modelos e ajustes para melhorar o resultado do treino e da inferência.


Apesar das definições acima, todas as pessoas envolvidas no trabalho participaram ativamente na tomada de decisões em todos níveis (coleta, treino e inferência) a fim de otimizar o código e melhorar o resultado obtido. Além disso, todos os membros participaram da construção do relatório e do vídeo.
