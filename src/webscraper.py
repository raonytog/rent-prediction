import os, re, json, time
import pandas as pd
from curl_cffi import requests
from numpy import random
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from imovel import Imovel
from const import *

# lista de palavras chaves que podem ser puladasd imediatamente quando encontradas no titulo de um anuncio
TITULOS_PULAVEIS = ['diaria', 'temporada', 'temporadas', 'dividir', 'divisao', 'compartilhar', 'compartilhamento', 'evento', 'festa', 'casamento', 'hostel']

# jeitos difernetes de escrever kitnet
KITNETS_ESCRITAS = ['kitnet', 'kitinete', 'kitchenette', 'kit net', 'quitinete']

def encontrar_ads(obj):
    """
    Função recursiva para encontrar a tag ADS no json
    """
    if isinstance(obj, dict):
        if 'ads' in obj and isinstance(obj['ads'], list): return obj['ads']
        for valor in obj.values():
            resultado = encontrar_ads(valor)
            if resultado is not None: return resultado
            
    elif isinstance(obj, list):
        for item in obj:
            resultado = encontrar_ads(item)
            if resultado is not None: return resultado
            
    return None

def tratar_kitnet(titulo, quartos, banheiros, area):
    """
    Analisa e trata os valores de um imovel com base no titulo e no que foi obtido para saber se é
    um imovel que deve ser ignorado, ou é uma kitnet
    
    Se nao for kitnet, nao tiver ou for 0 para area, banheiro ou quarto, entao é um imovel irregular e deve ser pulado da analise
    Se for kitnet e nao tiver ou for 0 as informacoes de quarto, banheiro ou area, infere-se por ser uma kitnet:
    quarto = 1, banheiro = 1 e area = 30 (media fixa das areas de uma kitnet)
    """
    skip = False
    eh_kitnet = any(k in titulo for k in KITNETS_ESCRITAS)  # sem area ou sem banheiro ou quarto e nao é kitnet
    if (not eh_kitnet) and (area in (None, 0) or banheiros in (None, 0) or quartos in (None, 0)): skip = True
    
    if eh_kitnet:
        if quartos in (None, 0):   quartos = 1
        if banheiros in (None, 0): banheiros = 1
        if area in (None, 0):      area = MIN_AREA_KITNET
        
    return skip, quartos, banheiros, area

def eh_galpao(titulo, quartos):
    """
    Verifica se um determinado anuncio é de um galpao com babse no titulo dele
    """
    return (('galpao' in titulo) or ('galpoes' in titulo)) and quartos == 0

def skip_imovel(cidade):
    """
    Verifica se a cidade está na Grande Vitória
    """
    if (cidade not in CIDADES) and (cidade not in CIDADES_NORM): return True
    return False

def extrair_numero_html(texto):
    """
    Pega so os numeros da string do html pra evitar casos tipo
    5 quartos -> 5
    """
    if not texto: return None
    if '-' in texto and not any(c.isdigit() for c in texto): return None
    
    res = re.sub(r'[^0-9,]', '', texto).replace(',', '.')
    try:                return float(res) if '.' in res else int(res)
    except ValueError:  return None
    
def coletar_dados_netimoveis(url_base):
    """
    Coleta os dados necessarios do netimoveis com base no html
    """
    dados_imoveis = []
    ids_vistos = set()

    url_base_limpa = re.sub(r'[?&]pagina=\d+', '', url_base)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        page.goto(url_base_limpa, wait_until="networkidle", timeout=60000)

        for pagina in range(1, QTD_PAGINAS + 1):
            try:
                page.wait_for_selector('article.card-imovel', timeout=15000)
                page.wait_for_timeout(2000)
                html_completo = page.content()

            except Exception as e:
                print(f"Erro ou fim dos anúncios na página {pagina}: {e}")
                break

            soup = BeautifulSoup(html_completo, 'html.parser')
            artigos = soup.find_all("article", class_=re.compile("card-imovel"))

            novos_anuncios = 0
            for artigo in artigos:
                id_anuncio = artigo.get('id')
                if not id_anuncio or id_anuncio in ids_vistos: continue
                
                link_tag = artigo.select_one('a.link-imovel')
                link_anuncio = f"https://www.netimoveis.com{link_tag['href']}" if link_tag else ""
                
                endereco_div = artigo.select_one('.endereco')
                bairro = endereco_div.text.split(',')[0].strip() if endereco_div else None
                
                preco_div = artigo.select_one('.valor')
                preco = preco_div.text if preco_div else None
                
                # tipo
                tipo_div = artigo.select_one('.tipo')
                casa_apartamento = tipo_div.text.strip().lower() if tipo_div else ""
                                
                # cidade
                cidade = ''
                for c in CIDADES_NORM:
                    if c in link_anuncio.lower():
                        cidade = c
                        break
                if skip_imovel(cidade): continue
                
                # caracteristicas principais
                area =      extrair_numero_html(artigo.select_one('.caracteristica.area').text)      if artigo.select_one('.caracteristica.area') else None
                quartos =   extrair_numero_html(artigo.select_one('.caracteristica.quartos').text)   if artigo.select_one('.caracteristica.quartos') else None
                banheiros = extrair_numero_html(artigo.select_one('.caracteristica.banheiros').text) if artigo.select_one('.caracteristica.banheiros') else None
                vagas =     extrair_numero_html(artigo.select_one('.caracteristica.vagas').text)     if artigo.select_one('.caracteristica.vagas') else 0
                
                # tratamento de alguns dados skipaveis e/ou trataveis
                # esta sendo feito aqui porque o titulo nao é um atribto armazenado
                titulo_div = artigo.select_one('.imovel-title')
                # titulo = titulo_div.text.strip().lower()
                titulo = Imovel.normalizar_palavras(titulo_div.text)
                skip = any(k in titulo for k in TITULOS_PULAVEIS)
                if skip or eh_galpao(titulo, quartos): continue

                skip, quartos, banheiros, area = tratar_kitnet(titulo, quartos, banheiros, area)
                if skip: continue
                    
                # opcionais
                lista_opcionais = [t.text.lower() for t in artigo.select('.imovel-caracteristicas-opcionais')]
                detalhes_lower = ", ".join(lista_opcionais)
                
                imovel = Imovel(id_anuncio, link_anuncio, cidade, bairro, casa_apartamento, area, quartos, banheiros, vagas, detalhes_lower, preco)

                ids_vistos.add(id_anuncio)
                dados_imoveis.append(imovel)
                novos_anuncios += 1
            
            print(f"Página {pagina}/{QTD_PAGINAS} - ({(pagina / QTD_PAGINAS)*100:.2f}%): {novos_anuncios} ads encontrados na NETIMOVEIS.", flush=True)
            
            # nao precisa clicar em avancar aqui (nem tem)
            if pagina == QTD_PAGINAS:
                break
            
            botao_proxima = page.locator('a.next.page-link') 
            
            if botao_proxima.is_visible():
                botao_proxima.click()
                page.wait_for_load_state("networkidle")
                
            else:
                print("Botão 'Próxima' não encontrado. Fim da paginação.")
                break
            
        browser.close()
    return dados_imoveis

def coletar_dados_olx(url_base):
    """
    Coleta os dados necessarios da OLX com base no html e json
    """
    dados_imoveis = []
    ids_vistos = set()

    # headers para evitar bloqueio por scrap nas paginas 
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "sec-ch-ua": '"Google Chrome";v="120", "Not:A-Brand";v="8", "Chromium";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for pagina in range(1, QTD_PAGINAS + 1):
        if "?" in url_base: url_atual = f"{url_base}&o={pagina}"
        else:               url_atual = f"{url_base}?o={pagina}"
        
        try:
            response = requests.get(url_atual, headers=headers, impersonate="chrome120", timeout=20)
        except Exception as e:
            print(f"Erro no request da pagina {pagina}: {e}")
            break
        
        if response.status_code != 200:
            print(f"Erro entrando na pagina {pagina}. Cod: {response.status_code}")
            break

        ads = None
        blocos_next = re.finditer(r'self\.__next_f\.push\((\[.*?\])\)', response.text) # self.next_f.push é onde o json com as informadcoes estao armazenadas
        for bloco in blocos_next: # percorre blocos para encontrar jsons com lista de anuncio para ser iterada e extraida no proximo for
            texto_array = bloco.group(1)
            try:
                dados_array = json.loads(texto_array)
                if isinstance(dados_array, list) and len(dados_array) > 1 and isinstance(dados_array[1], str):
                    inner_text = dados_array[1]
                    json_str = re.sub(r'^\d+:', '', inner_text)

                    try:
                        tree_data = json.loads(json_str)
                        encontrados = encontrar_ads(tree_data)
                        if encontrados:
                            ads = encontrados
                            break
                    except json.JSONDecodeError: continue
            except json.JSONDecodeError: continue

        if not ads:
            print("Anúncios não encontrados nesta página. Podemos ter chegado ao fim.")
            break

        novos_anuncios_na_pagina = 0
        # percorre cada div de anuncio e pega os atributos necessarios do json da div
        for ad in ads:
            if 'subject' not in ad: continue
            
            id_anuncio = ad.get('listId')
            if not id_anuncio or id_anuncio in ids_vistos: continue
                
            ids_vistos.add(id_anuncio)
            novos_anuncios_na_pagina += 1
            
            link_anuncio = ad.get('url')
            loc = ad.get('locationDetails', {})
            cidade = loc.get('municipality')
            if skip_imovel(cidade): continue
            
            bairro = loc.get('neighbourhood')
            preco = ad.get('priceValue')
            casa_apartamento = ad.get('categoryName')
            
            props = {p.get('name'): p.get('value') for p in ad.get('properties', [])}
            area = props.get('size')
            detalhes = props.get('re_features', '') + ", " + props.get('re_complex_features', '')
            detalhes_lower = detalhes.lower()
            
            quartos = props.get('rooms')
            banheiros = props.get('bathrooms')
            vagas = props.get('garage_spaces')
            
            # tratamento de alguns dados skipaveis e/ou trataveis
            # esta sendo feito aqui porque o titulo nao é um atribto armazenado
            titulo = Imovel.normalizar_palavras(ad.get('subject', ''))
            skip = any(k in titulo for k in TITULOS_PULAVEIS)
            if skip or eh_galpao(titulo, quartos): continue
            
                
            skip, quartos, banheiros, area = tratar_kitnet(titulo, quartos, banheiros, area)
            if skip: continue
                
            imovel = Imovel(id_anuncio, link_anuncio, cidade, bairro, casa_apartamento, area, quartos, banheiros, vagas, detalhes_lower, preco)

            dados_imoveis.append(imovel)

        print(f"Página {pagina}/{QTD_PAGINAS} - ({(pagina / QTD_PAGINAS)*100:.2f}%): {novos_anuncios_na_pagina} ads encontrados na OLX.\n", flush=True)
        time.sleep(random.uniform(3, 7))

    return dados_imoveis

def tratar_dados(df):
    """
    Trata os dados, removendo texto dos valores numericos, imputando valores faltantes inferiveis, removendo linhas nao inferiveis
    """    
    tamanho_original = len(df)
    
    # caracteristicas obrgiatorias dos imvoeis
    df['cidade']        = df['cidade'].apply(Imovel.normalizar_palavras)
    df['bairro']        = df['bairro'].apply(Imovel.normalizar_palavras)
    df['tipo']          = df['tipo'].apply(lambda x: CASAS if x == 'Casas' else APARTAMENTOS)
    df['area']          = df['area'].apply(Imovel.limpar_area)
    df['quartos']       = df['quartos'].apply(Imovel.limpar_qtde)
    df['banheiros']     = df['banheiros'].apply(Imovel.limpar_qtde)
    df['vagas']         = df['vagas'].apply(Imovel.limpar_qtde)

    # caracteristicas opcionais e inferiveis
    # elevador porteiro area_lazer academia varanda mobiliado
    df['elevador']      = df['elevador'].str.contains('elevador', na=False).astype(int)
    df['porteiro']      = df['porteiro'].str.contains('portaria|porteiro', case=False, na=False).astype(int)
    df['area_lazer']    = df['area_lazer'].str.contains('piscina|salão de festas|lazer|area de churrasco|quadra', case=False, na=False).astype(int)
    df['academia']      = df['academia'].str.contains('academia', case=False, na=False).astype(int)
    df['varanda']       = df['varanda'].str.contains('varanda', case=False, na=False).astype(int)
    df['mobiliado']     = df['mobiliado'].str.contains('mobiliado', case=False, na=False).astype(int)

    # label
    df['preco']         = df['preco'].apply(Imovel.limpar_preco)
    
    # remove as linhas com preco, cidade, bairro ou tipo nulos
    df = df.dropna(subset=['preco', 'cidade', 'bairro', 'tipo'])
    print(f"\t{tamanho_original - len(df)} imóveis removidos por preço, cidade, bairro ou tipo vazios")
    
    # imputa valores 0 o mediana para as seguintes colunas
    colunas_para_imputar = ['vagas', 'elevador', 'porteiro', 'area_lazer', 'academia', 'varanda', 'mobiliado']
    for col in colunas_para_imputar:
        if col in df.columns and df[col].isnull().any():
            df[f'{col}_ausente'] = df[col].isnull().astype(int)
            # se n tiver valor nessas colunas, preenche com zero
            df[col] = df[col].fillna(0)
                
    # transforma as colunas de colunas_int para o tipo inteiro
    colunas_int = ['quartos', 'banheiros', 'vagas']
    for col in colunas_int:
        if col in df.columns: df[col] = df[col].astype(int)
        
    return df

def juntar_dados(df_novo):
    """
    Faz merge nos dados antigos ja existentes no repositorio com os dados lidos agora.
    Alem disso, se houver duplicatas, remove-as com base no id dos imóveis
    """
    os.makedirs('planilhas', exist_ok=True)
    
    df_completo = None
    if os.path.exists(PATH_DADOS):
        df_antigo = pd.read_csv(PATH_DADOS)
        df_completo = pd.concat([df_antigo, df_novo], ignore_index=True)
        print(f"Antigo [{len(df_antigo)}] x novo [{len(df_novo)}]")
        
    else:
        print("Nao ha dados antigos; criado a primeira planilha")
        df_completo = df_novo
        
    return df_completo.drop_duplicates(subset=['id'], keep='last')

def salvar_dados(df):
    """
    Salva os dados coletados na pasta planilhas/dados.csv
    """
    df.to_csv(PATH_DADOS, index=False, encoding='utf-8-sig', sep=',')
    print(f"Total de imóveis salvos: {len(df)}")

def plotar_distribuicoes(df, max_value):
    """
    Plota a distribuição do preço original filtrado até 1 milhão,
    com a frequência no eixo Y, os preços no eixo X e uma curva de tendência.
    """
    # Filtra os dados apenas para o plot, para que os outliers não estraguem 
    # o cálculo dos "bins" (barras) do histograma e nem a curva.
    df_plot = df[df['preco'] <= max_value]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Histograma de Frequência ---
    # Usamos o df_plot que já está filtrado
    df_plot['preco'].plot.hist(bins=50, alpha=0.6, color='dodgerblue', ax=ax1)
    
    ax1.set_title("Distribuição de Preços")
    ax1.set_xlabel("Preço")
    ax1.set_ylabel("Frequência", color='dodgerblue')
    
    # Trava o limite do eixo X visualmente de 0 a 1 milhão
    ax1.set_xlim(0, max_value)
    
    # Formata o eixo X para mostrar os números inteiros
    ax1.ticklabel_format(style='plain', axis='x')

    # --- Curva de Densidade (KDE) ---
    ax2 = ax1.twinx()
    df_plot['preco'].plot.kde(ax=ax2, color='red', linewidth=2)
    
    # Garante que a curva acompanhe os mesmos limites do eixo X principal
    ax2.set_xlim(0, max_value)
    ax2.set_ylim(bottom=0) # Evita que a curva ultrapasse o "chão" do gráfico
    ax2.set_yticks([]) 
    ax2.set_ylabel("Curva de Tendência", color='red')

    # Ajusta o layout e salva a imagem
    plt.tight_layout()
    nome_arquivo = "distribuicao_precos.png"
    plt.savefig(nome_arquivo, dpi=300)
    print(f"Gráfico de distribuição salvo como '{nome_arquivo}'")
    
    # Limpa a figura da memória
    plt.close()
    
    return df

def main():
    print("Procurando dados da OLX:")
    dados_olx = coletar_dados_olx(URL_OLX)
    print()
    
    # Há essa separação pois o site não deixava adicionar mais filtros do que 5.
    # Por serem 7 cidades, foi necessário coletar em duas etapas/urls diferentes
    print("\nProcurando dados da NETIMOVEIS:")
    dados_netimoveis_5 = coletar_dados_netimoveis(URL_NETIMOVEIS_5)
    dados_netimoveis_2 = coletar_dados_netimoveis(URL_NETIMOVEIS_2)
    print()
    
    print(len(dados_olx))
    print(len(dados_netimoveis_5))
    print(len(dados_netimoveis_2))
    
    # junta os dados
    dados_totais = ((dados_olx or []) + (dados_netimoveis_5 or []) + (dados_netimoveis_2 or []))
    
    if not dados_totais:
        print("Nenhum dado encontrado em nenhum dos sites.")
        return
    
    df = pd.DataFrame([i.__dict__ for i in dados_totais]) # transforma cada item da classe Imovel em um dicionário e converte pra um dataFrame
    df = tratar_dados(df)
    df = juntar_dados(df)
    
    salvar_dados(df)

if __name__ == '__main__':
    main()