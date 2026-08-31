import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsRegressor
from const import *
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

"""
código dos modelos para a função testa_parametros
"""
RANDOM_FOREST = 1
XGBOOST = 2
KNN = 3
HISTGBOOST = 4

def remover_outliers(df, coluna, multiplicador=1.7):
    """
    remove os outliers com base nos limites calculos com iqr sobre a variável em escala logarítmica
    """
    coluna_log = np.log1p(df[coluna])
    
    q1, q3 = coluna_log.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inferior = q1 - multiplicador * iqr
    limite_superior = q3 + multiplicador * iqr
    tamanho_antigo = len(df)
    metrica = ''

    if coluna == 'area':
        limite_minimo_area_log = np.log1p(10.0) 
        limite_inferior = max(limite_inferior, limite_minimo_area_log)
        metrica = 'm2'
        
    if coluna == 'preco':
        limite_minimo_preco_log = np.log1p(700)
        limite_inferior = max(limite_inferior, limite_minimo_preco_log)
        metrica = 'R$'
    
    df_filtrado = df[(coluna_log >= limite_inferior) & (coluna_log <= limite_superior)]

    # só para o print, convertemos os limites de volta para as métricas reais
    limite_inf_real = np.expm1(limite_inferior)
    limite_sup_real = np.expm1(limite_superior)
    
    plot(df, coluna, limite_inf_real, limite_sup_real)
    
    print(f"\t{tamanho_antigo - len(df_filtrado)} imóveis removidos como outliers de '{coluna}' (via Log-IQR)")
    print(f"\t(Faixa aceita equivalente em {metrica}: {limite_inf_real:.2f} a {limite_sup_real:.2f})")
    
    return df_filtrado

def agrupar_bairros(treino, teste, contagem_min_bairros=QTD_MIN_BAIRROS):
    """
    agrupa os bairros que ocorreram menos que min_bairros vezes em 'Outro'
    """
    treino = treino.copy()
    teste = teste.copy()
    
    contagem_bairros = treino['bairro'].value_counts()
    bairros_raros = contagem_bairros[contagem_bairros < contagem_min_bairros].index
    
    treino['bairro'] = treino['bairro'].apply(lambda b: 'Outro' if b in bairros_raros else b)
    teste['bairro'] = teste['bairro'].apply(lambda b: 'Outro' if b in bairros_raros else b)
    
    print(f"\t{len(bairros_raros)} bairros (de {len(contagem_bairros)}) agrupados em 'Outro' "f"(menos de {contagem_min_bairros} anúncios)")
    return treino, teste

def one_hot_encoding(df):
    """
    aplica one hot encoding para o atributo cidade
    """
    encoded_df = pd.get_dummies(
        df,
        columns=['cidade'],
        drop_first=False
    )

    for col in encoded_df.columns:
        if encoded_df[col].dtype == bool:
            encoded_df[col] = encoded_df[col].astype(int)

    y = encoded_df['preco']
    X = encoded_df.drop(columns=["preco"], inplace=False)

    return X, y

def testa_parametros(X_train, y_train, modelo):
    """
    testa parametros para RandomForestRegressor com RandomizedSearchCV
    - 1: RandomForest
    - 2: XGBoost
    - 3: KNN
    - 4: HistGradientBoosting
    """
    if modelo == RANDOM_FOREST:        
        param_dist = {
            'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)], # Número de árvores
            'max_features': ['sqrt', 'log2', None],                                    # Recursos por divisão
            'max_depth': [int(x) for x in np.linspace(5, 50, num=10)] + [None],        # Profundidade máxima
            'min_samples_split': [2, 5, 10],                                           # Mínimo para dividir nó
            'min_samples_leaf': [1, 2, 4],                                             # Mínimo por folha
            'bootstrap': [True, False]                                                 # Amostragem bootstrap
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_random = RandomizedSearchCV(
            estimator=rf, 
            param_distributions=param_dist, 
            n_iter=50,                         # Testa 50 combinações aleatórias distintas
            cv=5,                              # Validação cruzada de 5 subconjuntos
            scoring='neg_mean_squared_error',  # Métrica de otimização (MSE negativo)
            verbose=1,                         # Exibe o progresso do treinamento
            random_state=42,                   # Garante a reprodutibilidade dos testes
            n_jobs=-1                          # Executa utilizando todos os núcleos do processador
        )
        
        rf_random.fit(X_train, y_train)
        
        print("Melhores parâmetros encontrados:")
        print(rf_random.best_params_)

        return rf_random.best_estimator_
    
    elif modelo == XGBOOST:
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],          # Quantidade de árvores
            'learning_rate': [0.01, 0.05, 0.075, 0.1],       # Taxa de aprendizado (passos menores = mais precisão)
            'max_depth': [4, 5, 6, 7],                     # Profundidade de cada árvore
            'subsample': [0.7, 0.8, 0.9, 1.0],             # % de dados usados para construir cada árvore
            'colsample_bytree': [0.5, 0.75, 1.0],      # % de colunas usadas para construir cada árvore
            'reg_alpha': [0.1, 1, 5, 10],           # Regularização L1 (penaliza pesos, pode zerar features)
            'reg_lambda': [10, 15, 20, 25, 30],            # Regularização L2 (suaviza os pesos, reduz variância)
            'min_child_weight': [5, 7, 10, 15],            # Mínimo de "peso" numa folha p/ permitir a divisão (evita nós muito específicos)
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_random = RandomizedSearchCV(
            estimator=xgb_model, 
            param_distributions=param_dist, 
            n_iter=80,                         # Testa 80 combinações aleatórias distintas
            cv=5,                              # Validação cruzada de 5 subconjuntos
            scoring='neg_mean_absolute_percentage_error',  # Métrica de otimização (MAPE negativo)
            random_state=42,                   # Garante a reprodutibilidade dos testes
            n_jobs=-1                          # Executa utilizando todos os núcleos do processador
        )
        
        xgb_random.fit(X_train, y_train)
        
        print("Melhores parâmetros encontrados:")
        print(xgb_random.best_params_)

        return xgb_random.best_estimator_
    
    elif modelo == KNN:
        param_dist = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],               # Quantos imóveis "vizinhos" ele deve consultar?
            'weights': ['uniform'],                                # Os vizinhos têm pesos iguais
            'p': [1, 2],                                           # Como medir a distância? 1 = Manhattan, 2 = Euclidiana
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] # Como ele procura os vizinhos
        }
        
        knn = KNeighborsRegressor()
        knn_random = RandomizedSearchCV(
            estimator=knn,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_absolute_error',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        
        knn_random.fit(X_train, y_train)
        
        print("Melhores parâmetros encontrados:")
        print(knn_random.best_params_)

        return knn_random.best_estimator_
    
    elif modelo == HISTGBOOST:
        hgb = HistGradientBoostingRegressor(random_state=42)
    
        param_dist = {
            'max_iter': [100, 300, 500], # Equivalente ao número de árvores (n_estimators)
            'learning_rate': [0.01, 0.05, 0.1], # O impacto de cada árvore
            'max_leaf_nodes': [15, 31, 63], # Quantidade máxima de "folhas" (divisões finais)
            'max_depth': [5, 7, 9], # Profundidade máxima de cada árvore
            'min_samples_leaf': [5, 10, 15, 20], # Mínimo de imóveis necessários numa folha para ela existir
            # 'l2_regularization': [0, 0.1, 1.0] # Penaliza modelos muito complexos (ajuda a evitar overfitting)
        }
        
        hgb_random = RandomizedSearchCV(
            estimator=hgb,
            param_distributions=param_dist,
            n_iter=30, 
            scoring='neg_mean_absolute_error',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        
        hgb_random.fit(X_train, y_train)
        
        print("Melhores parâmetros encontrados:")
        print(hgb_random.best_params_)

        return hgb_random.best_estimator_


def imprimir_comparacao(y_real, y_previsto, qtd=10, ordenar_por=None):
    """
    imprime a comparação entre valores reais e previstos, calculando a diferença percentual
    ordenar_por: None mantém a ordem original, 'maiores' ou 'menores' ordena pelo valor real
    """
    comparacao = pd.DataFrame({'Real': np.asarray(y_real), 'Previsto': np.asarray(y_previsto)})

    if ordenar_por == 'maiores':
        comparacao = comparacao.sort_values(by='Real', ascending=False)
    elif ordenar_por == 'menores':
        comparacao = comparacao.sort_values(by='Real', ascending=True)

    for _, row in comparacao.head(qtd).iterrows():
        real, previsto = row['Real'], row['Previsto']
        erro_percentual = 100 - (min(real, previsto) / max(real, previsto) * 100)
        print(f"\tReal: R$ {real:.2f} | Previsto: R$ {previsto:.2f} | Diferença: {erro_percentual:.2f}%")

def imprimir_resultados_cv(cv_r2_treino, cv_r2_teste, cv_rmse, cv_mae, cv_mape):
    """
    imprime a média das métricas obtidas na validação cruzada
    """
    print(f"R² Médio (Treino): {np.mean(cv_r2_treino):.4f}")
    print(f"R² Médio (Teste):  {np.mean(cv_r2_teste):.4f}")
    print(f"RMSE Médio:        R$ {np.mean(cv_rmse):.2f}")
    print(f"MAE Médio:         R$ {np.mean(cv_mae):.2f}")
    print(f"MAPE Médio:        {np.mean(cv_mape):.2f}%")
    
def criar_jobs(modelo, scaler, encoder, X_train, imputador_area, imputador_quartos, imputador_banheiros):
    """
    salva o modelo criado com joblib para ser reutilizado em inferencia.py
    """
    os.makedirs(PATH_PASTA_JOBLIB, exist_ok=True)
    joblib.dump(modelo, f'{PATH_PASTA_JOBLIB}/modelo.joblib')
    joblib.dump(scaler, f'{PATH_PASTA_JOBLIB}/scaler.joblib')
    joblib.dump(encoder, f'{PATH_PASTA_JOBLIB}/encoder.joblib')
    joblib.dump(list(X_train.columns), f'{PATH_PASTA_JOBLIB}/colunas_modelo.joblib')
    joblib.dump(imputador_area, f'{PATH_PASTA_JOBLIB}/imputador_area.pkl')
    joblib.dump(imputador_quartos, f'{PATH_PASTA_JOBLIB}/imputador_quartos.pkl')
    joblib.dump(imputador_banheiros, f'{PATH_PASTA_JOBLIB}/imputador_banheiros.pkl')

def gerar_grafico(importancias, colunas):
    """
    gera um gráfico que exibe a importancia dada a cada atributo pelo modelo
    """
    df_importancia = pd.DataFrame({
        'Atributo': colunas,
        'Importancia': importancias
    })

    # Ordenamos do mais importante para o menos importante
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

    # Desenha o gráfico
    plt.figure(figsize=(10, 6))
    
    # Desenha o gráfico com as novas regras do Seaborn
    sns.barplot(x='Importancia', y='Atributo', data=df_importancia, hue='Atributo', palette='viridis', legend=False)
    plt.title('O que mais impacta o valor do aluguel?', fontsize=14, fontweight='bold')
    plt.xlabel('Grau de Importância (XGBoost)', fontsize=12)
    plt.ylabel('Características do Imóvel', fontsize=12)

    # Adiciona os valores numéricos no final de cada barra para facilitar a leitura
    for index, value in enumerate(df_importancia['Importancia']):
        plt.text(value, index, f' {value:.3f}', va='center')

    plt.tight_layout()
    os.makedirs('graficos', exist_ok=True)
    plt.savefig('graficos/grafico_importancia.png', dpi=300, bbox_inches='tight')
    
def plot(df: pd.DataFrame, coluna, low=0, high=30_000, min_freq=10):
    fig, ax = plt.subplots(figsize=(12, 6))

    # para exibir valores numericos
    if pd.api.types.is_numeric_dtype(df[coluna]):
        max_col = high
        if coluna == 'preco': max_col = 100_000
        elif coluna == 'area': max_col = 1_000
        
        s = df.loc[df[coluna] < max_col, coluna].value_counts().sort_index()
        moda = df.loc[df[coluna] < max_col, coluna].mode()[0] # moda dos dados ate a limitacao
        ax.plot(s.index, s.values, marker='o', linestyle='None')

        # linhas verticais em x
        ax.axvline(x=low,  color='red',    linestyle='--', linewidth=2, label=f'Low = {low:.2f}')
        ax.axvline(x=high, color='green',  linestyle='--', linewidth=2, label=f'High = {high:.2f}')
        ax.axvline(x=moda, color='orange', linestyle='--', linewidth=2, label=f'Moda = {moda:.2f}')

        ax.legend()

    # para exibir bairros, cidades
    else:
        s = df[coluna].value_counts()
        s = s.head(min_freq)
        ax.bar(s.index, s.values)
        plt.xticks(rotation=45, ha="right")

    ax.set(xlabel=coluna.capitalize(), ylabel="Frequência", title=f"{coluna.capitalize()} x Frequência")
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()
  
def main():
    """
    aplicando tecnicas de one hot encoding, estratificação, validação cruzada, 
    target encoding e randomized search, treina o modelo de previsão
    de valores para aluguel de imoveis na grande vitoria
    """
    df = pd.read_csv(PATH_DADOS)
    df.drop(columns=["id", "url"], inplace=True)
    
    print(f"@ Total de imóveis: {len(df)}")
    df = remover_outliers(df, 'preco', 1.6)
    df = remover_outliers(df, 'area', 2.5)
    print(f"\tTotal de imóveis após limpeza: {len(df)}")
    
    plot(df, 'bairro', min_freq=45)
        
    X, y = one_hot_encoding(df)

    # estratificação para garantir a mesma distribuição de preços em todos os folds
    n_bins = int(np.floor(1 + np.log2(len(df)))) # numero de bins
    y_cat = pd.cut(y, bins=n_bins, labels=False) # transformação do y em categorias de acordo com o num de bins
    skf_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # divisao dos dados em splits

    # listas para guardar a performance de cada fold e tirar a media no final
    cv_r2_treino = []
    cv_r2_teste = []
    cv_rmse = []
    cv_mae = []
    cv_mape = []
    
    # loop principal treinamento, com a divisão dos conjuntos de treino/teste de acordo com os splits
    for _, (train_idx, test_idx) in enumerate(skf_split.split(X, y_cat)): 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # transforma a coluna y em logaritimo para facilitar o modelo a lidar com outliers
        y_train_log = np.log1p(y_train) 

        # transformaçao da categoria 'bairro' de categórica para numérica
        X_train, X_test = agrupar_bairros(X_train, X_test)
        encoder = TargetEncoder(cols=['bairro'], smoothing=15)
        X_train['bairro_codificado'] = encoder.fit_transform(X_train['bairro'], y_train / X_train['area']) # faz fit apenas no treino para calcular média sem olhar para o teste
        X_test['bairro_codificado'] = encoder.transform(X_test['bairro']) # se um bairro que não existia no teste aparecer no treino, preenche com a média global de preco

        # calcula as medias de area por bairro para imputar na inferencia
        medias_area_bairro = X_train.groupby('bairro')['area'].mean().to_dict()
        media_area_global = X_train['area'].mean()
        imputador_area = {
            'por_bairro': medias_area_bairro,
            'global': media_area_global
        }
        
        # calcula as medias de quartos por bairro para imputar na inferencia
        medias_quartos_bairro = X_train.groupby('bairro')['quartos'].mean().to_dict()
        media_quartos_global = X_train['quartos'].mean()
        imputador_quartos = {
            'por_bairro': medias_quartos_bairro,
            'global': media_quartos_global
        } 
        
        # calcula as medias de banheiros por bairro para imputar na inferencia
        medias_banheiros_bairro = X_train.groupby('bairro')['banheiros'].mean().to_dict()
        media_banheiros_global = X_train['banheiros'].mean()
        imputador_banheiros = {
            'por_bairro': medias_banheiros_bairro,
            'global': media_banheiros_global
        } 
        
        # remove a coluna bairro original para manter apenas a codificada
        X_train = X_train.drop('bairro', axis=1)
        X_test = X_test.drop('bairro', axis=1)

        # faz o escalonamento dos dados de treino e aplica no de teste
        scaler = StandardScaler()
        X_treino_escalonado = scaler.fit_transform(X_train)
        X_teste_escalonado = scaler.transform(X_test)

        # aplicando transformação logaritmica
        y_train_log = np.log1p(y_train)
        
        # testando parâmetros do modelo escolhido com RandomizedSearchCV
        modelo = testa_parametros(X_treino_escalonado, y_train_log, XGBOOST)        
        modelo.fit(X_treino_escalonado, y_train_log)
            
        # realiza a predição e desfaz a transformação logaritmica para calcular métricas em reais(R$)
        previsoes_treino = np.expm1(modelo.predict(X_treino_escalonado))
        previsoes_teste = np.expm1(modelo.predict(X_teste_escalonado))
    
        # calcula métricas
        mse = mean_squared_error(y_test, previsoes_teste)
        r2_treino = r2_score(y_train, previsoes_treino)
        r2_teste = r2_score(y_test, previsoes_teste)
        mae = mean_absolute_error(y_test, previsoes_teste)
        rmse = root_mean_squared_error(y_test, previsoes_teste)
        mape = mean_absolute_percentage_error(y_test, previsoes_teste) * 100

        # Armazena na lista para o resumo final
        cv_r2_treino.append(r2_treino)
        cv_r2_teste.append(r2_teste)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_mape.append(mape)

    # impressão dos resultados obtidos
    print("\n@ Resultados finais Validação Cruzada")
    imprimir_resultados_cv(cv_r2_treino, cv_r2_teste, cv_rmse, cv_mae, cv_mape)
    
    print("\n@ Valores com maior importancia:")
    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importancias.head(10))

    print("\n@ Comparacao (real x previsto x diferenca)")
    imprimir_comparacao(y_test, previsoes_teste)
    
    print("\n@ Comparacao das amostras MAIS CARAS")
    imprimir_comparacao(y_test, previsoes_teste, qtd=20, ordenar_por='maiores')
    
    print("\n@ Comparacao das amostras MAIS BARATAS")
    imprimir_comparacao(y_test, previsoes_teste, qtd=20, ordenar_por='menores') 
    
    criar_jobs(modelo, scaler, encoder, X_train, imputador_area, imputador_quartos, imputador_banheiros)
    gerar_grafico(modelo.feature_importances_, X_train.columns)

if __name__ == '__main__':
    main()
