import re
import unicodedata

import numpy as np
import pandas as pd

class Imovel():
    def __init__(self, id, url, cidade, bairro, tipo, area, quartos, banheiros, vagas, propriedades, preco):
        self.id = id
        self.url = url
        self.cidade = cidade
        self.bairro = bairro
        self.tipo = tipo
        self.area = area
        self.quartos = quartos
        self.banheiros = banheiros
        self.vagas = vagas
        self.elevador = propriedades    # 1 if 'elevador' in propriedades else 0
        self.porteiro = propriedades    # 1 if 'portaria' in propriedades or 'porteiro' in propriedades else 0
        self.area_lazer = propriedades  # 1 if 'piscina' in propriedades or 'salão de festas' in propriedades or 'lazer' in propriedades else 0
        self.academia = propriedades    # 1 if 'academia' in propriedades else 0
        self.varanda = propriedades     # 1 if 'varanda' in propriedades else 0
        self.mobiliado = propriedades   # 1 if 'mobiliado' in propriedades else 0
        self.preco = preco
        
    def normalizar_palavras(cidade_bairro):
        """
        remove acentos e letras maiusculas das palavras
        """
        if not isinstance(cidade_bairro, str): return
        if not cidade_bairro: return None
        return ''.join(c for c in unicodedata.normalize('NFD', cidade_bairro) if unicodedata.category(c) != 'Mn').lower().replace(" ", "-")
        
    def limpar_area(area_str):
        """
        remove tudo que não for número do campo de área
        assume-se que as áreas sempre serão em metros quadrados (m²)
        """
        if not area_str: return None
        
        res = re.sub(r'[^0-9.]', '', str(area_str).replace(',', '.'))
        try: return float(res)
        except: return None

    def limpar_preco(preco_str):
        """
        remove tudo que não for número do campo do preço
        assume-se que os preços estão em reais (ex: R$ 1.000,00 )
        converte o número em inteiro
        """
        if not preco_str: return None
    
        res = re.sub(r'[^0-9]', '', str(preco_str))
        try: return int(res)
        except: return None
        
    def limpar_qtde(valor):
        """
        se for float, retorna a conversão do valor para int
        remove tudo que não for número do campo
        
        especialmente para a olx, onde o numero é truncado para 5, tratamos essa string para inteiro e ela é fixa em 5
        """
        # if valor is None or np.isnan(valor): return None
        if pd.isna(valor): return None
        
        if isinstance(valor, float):
            # NaN
            if valor != valor: return None
            return int(valor)
        
        str_val = str(valor).lower()
        if "5 ou mais" in str_val: return 5
        res = re.sub(r'[^0-9]', '', str_val)
        
        try:    return int(res)
        except: return None
        
    def getCidade(self):
        return self.cidade
    
    def getBairro(self):
        return self.bairro