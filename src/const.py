# cidades da grande vitoria sem e com normalizacao
CIDADES      = ["Vitória", "Vila Velha", "Serra", "Cariacica", "Guarapari", "Viana", "Fundão"]
CIDADES_NORM = ["vitoria", "vila-velha", "serra", "cariacica", "guarapari", "viana", "fundao"]

# link olx es
URL_OLX = 'https://www.olx.com.br/imoveis/aluguel/estado-es'

# link netimoveis com vitoria, vila velha, serra, cariacica e guarapari
URL_NETIMOVEIS_5 = 'https://www.netimoveis.com/aluguel/espirito-santo/vitoria?transacao=aluguel&localizacao=BR-ES-vitoria---%2CBR-ES-vila-velha---%2CBR-ES-serra---%2CBR-ES-cariacica---%2CBR-ES-guarapari---'
# link netimoveis com fundao e viana
URL_NETIMOVEIS_2 = 'https://www.netimoveis.com/locacao/espirito-santo/fundao?transacao=locacao&localizacao=BR-ES-fundao---%2CBR-ES-viana---'


# diretorios dos arquivos a serem lidos e gerados
PATH_DADOS = 'dados.csv'
PATH_ENTRADA = 'entrada.csv'
PATH_RESULTADO = 'resultado.csv'
PATH_PASTA_JOBLIB = 'joblib'

# codigo do atributo 'tipo'
CASAS = 0
APARTAMENTOS = 1

# constantes de auxilio:
# qtd de paginas a serem lidas no scraper
QTD_PAGINAS = 100
# qtd min que um bairro tem para ser agrupado em 'Outros'
QTD_MIN_BAIRROS = 3

# area media deuma kitnet
MIN_AREA_KITNET = 30