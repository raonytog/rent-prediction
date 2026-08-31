import numpy as np
import pandas as pd
import joblib
from src.const import *
from src.imovel import Imovel

def carregar_modelo(pasta=PATH_PASTA_JOBLIB):
    """
    Carrega o modelo, o scaler, o encoder,a lista de colunas e os 
    imputadores de área, quartos e banheiros salvos por treino.py via joblib
    """
    modelo = joblib.load(f'{pasta}/modelo.joblib')
    scaler = joblib.load(f'{pasta}/scaler.joblib')
    encoder = joblib.load(f'{pasta}/encoder.joblib')
    colunas_modelo = joblib.load(f'{pasta}/colunas_modelo.joblib')
    imputador_area = joblib.load(f'{pasta}/imputador_area.pkl')
    imputador_quartos = joblib.load(f'{pasta}/imputador_quartos.pkl')
    imputador_banheiros = joblib.load(f'{pasta}/imputador_banheiros.pkl')
    return modelo, scaler, encoder, colunas_modelo, imputador_area, imputador_quartos, imputador_banheiros

def prever_preco(df, modelo, scaler, encoder, colunas_modelo):
    """
    Ajusta o df ao modelo de previsão e aplica o modelo
    """
    # one-hot da cidade
    df = pd.get_dummies(df, columns=['cidade'], drop_first=False)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    # target encoding do bairro usando o encoder já treinado 
    df['bairro_codificado'] = encoder.transform(df['bairro'])
    df = df.drop('bairro', axis=1)

    # garante que todas as colunas usadas no treino existam, na mesma ordem
    df = df.reindex(columns=colunas_modelo, fill_value=0)

    #aplica o scaler que padroniza os valores dos atributos
    df_escalonado = scaler.transform(df)
    
    #aplica o modelo e a funcao exponencial para tirar da escala logaritmica
    preco_previsto = np.expm1(modelo.predict(df_escalonado))
    
    return preco_previsto

def limpa_tipo(valor):
    """
    Limpa o atributo tipo do df, verificando o seu tipo
    O objetivo é converter o valor para inteiro
    """
    if np.isnan(valor): return np.nan
    return int(valor)

def aplicar_imputacao_area(linha, imputador_area):
    """
    Imputa o valor médio encontrado no treino para a área
    """
    # Verifica se a área está realmente vazia (NaN)
    if pd.isna(linha['area']):
        bairro_atual = linha['bairro']
        # Tenta pegar a média do bairro. Se o bairro não existir no dicionário, usa a global
        return imputador_area['por_bairro'].get(bairro_atual, imputador_area['global'])
    
    # Se já tiver área, devolve o valor original
    return linha['area']

def aplicar_imputacao_quartos(linha, imputador_quartos):
    """
    Imputa o valor médio encontrado no treino para quartos
    """
    # Verifica se quartos está realmente vazio (NaN)
    if pd.isna(linha['quartos']):
        bairro_atual = linha['bairro']
        # Tenta pegar a média do bairro. Se o bairro não existir no dicionário, usa a global
        return imputador_quartos['por_bairro'].get(bairro_atual, imputador_quartos['global'])
    
    # Se já tiver quartos, devolve o valor original
    return linha['quartos']

def aplicar_imputacao_banheiros(linha, imputador_banheiros):
    """
    Imputa o valor médio encontrado no treino para banheiros
    """
    # Verifica se banheiros está realmente vazio (NaN)
    if pd.isna(linha['banheiros']):
        bairro_atual = linha['bairro']
        # Tenta pegar a média do bairro. Se o bairro não existir no dicionário, usa a global
        return imputador_banheiros['por_bairro'].get(bairro_atual, imputador_banheiros['global'])
    
    # Se já tiver banheiros, devolve o valor original
    return linha['banheiros']

def tratar_dados(df, imputador_area, imputador_quartos, imputador_banheiros):
    """
    Trata os dados, removendo texto dos valores numericos, imputando valores faltantes quando possível
    """    
    df = df.drop(columns=["id"], inplace=False)
    
    # formata e trata parte dos dados
    df['cidade']        = df['cidade'].apply(Imovel.normalizar_palavras)

    colunas_ausentes = ['vagas', 'elevador', 'porteiro', 'area_lazer', 'academia', 'varanda', 'mobiliado']
    for col in df.columns:
        if df[col].isnull().any():
            #cria a coluna que sinaliza que o atributo veio vazio
            if col in colunas_ausentes:
                df[f'{col}_ausente'] = df[col].isnull().astype(int)
            
                # imputa valores 0 para essas colunas
                df[col] = df[col].fillna(0)
            
        # verifica se é uma das cidades esperadas. Se não for, coloca como vazio
        if col == 'cidade':
            df[col] = df[col].where(df[col].isin(CIDADES_NORM), None)
        
        # verifica se é um dos valores esperados. Se não for, coloca como vazio
        elif col == 'tipo':
            df[col] = df[col].where(df[col].isin({0,1}), None)
        
        # verifica se é um dos valores esperados. Se não for, coloca como 0
        elif col in ['elevador', 'porteiro', 'area_lazer', 'academia', 'varanda', 'mobiliado']:
            df[col] = df[col].where(df[col].isin({0,1}), 0)

    # formata e trata resto dos dados
    df['bairro']        = df['bairro'].apply(Imovel.normalizar_palavras)
    df['vagas']         = df['vagas'].apply(Imovel.limpar_qtde)
    df['area']          = df['area'].apply(Imovel.limpar_area)
    df['quartos']       = df['quartos'].apply(Imovel.limpar_qtde)
    df['banheiros']     = df['banheiros'].apply(Imovel.limpar_qtde)
    df['tipo']          = df['tipo'].apply(limpa_tipo)
    
    #preenche a área vazia com o valor da media por bairro calculado no treinamento
    df['area'] = df.apply(aplicar_imputacao_area, axis=1, args=(imputador_area,))
    #preenche a coluna quartos vazia com o valor da media por bairro calculado no treinamento
    df['quartos'] = df.apply(aplicar_imputacao_quartos, axis=1, args=(imputador_quartos,))
    #preenche a coluna banheiros vazia com o valor da media por bairro calculado no treinamento
    df['banheiros'] = df.apply(aplicar_imputacao_banheiros, axis=1, args=(imputador_banheiros,))
    
    #converte o tipo das colunas para inteiro
    colunas_int = ['elevador', 'porteiro', 'area_lazer', 'academia', 'varanda', 'mobiliado']      
    for col in colunas_int:
        if col in df.columns: df[col] = df[col].astype(int)
    
    #transforma todos os None em NaN para serem melhor aplicados no modelo
    df = df.fillna(value=np.nan)
    
    # gera dados no csv de saída para conferir se está tudo correto
    # df.to_csv('saida.csv', index=False, encoding='utf-8-sig', sep=',')
        
    return df

def gera_saida(id, previsao):
    """
    Salva os dados preditos no arquivo resultado.csv
    """
    df = pd.DataFrame({
        "id": id,
        "valor_previsto": previsao.round(2)
    })
    df.to_csv(PATH_RESULTADO, index=False, encoding='utf-8-sig', sep=',')

def main():
    """
    Colunas a serem lidas para a previsão:
    id,cidade,bairro,tipo,area,quartos,banheiros,vagas,elevador,porteiro,area_lazer,academia,varanda,mobiliado
    """  
    modelo, scaler, encoder, colunas_modelo, imputador_area, imputador_quartos, imputador_banheiros = carregar_modelo()
    
    df = pd.read_csv(PATH_ENTRADA)
    df_limpo = tratar_dados(df, imputador_area, imputador_quartos, imputador_banheiros)
    preco = prever_preco(df_limpo, modelo, scaler, encoder, colunas_modelo)
    gera_saida(df['id'], preco)

if __name__ == "__main__":
    main()