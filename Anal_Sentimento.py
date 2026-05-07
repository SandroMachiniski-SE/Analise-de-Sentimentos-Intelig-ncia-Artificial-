# Sandro Machiniski, Diego Cunha e Lucas Eufrasio

# 1. Importar bibliotecas necessárias
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import FreqDist, classify, NaiveBayesClassifier

# Baixar recursos do NLTK (se ainda não estiverem instalados)
nltk.download('stopwords')
nltk.download('rslp')

# 2. Base de treinamento (frases com emoção associada)
base_treinamento = [
    ("eu amo este lanche", "alegria"),
    ("este atendimento foi irritante", "raiva"),
    ("estou surpreso com a entrega", "surpresa"),
    ("estou desapontado com a qualidade", "tristeza"),
    ("fiquei com medo do resultado", "medo"),
    ("o sabor está repugnante", "desgosto"),
]

# 3. Remoção de stopwords e stemização
stopwords_pt = stopwords.words('portuguese')
stemmer = RSLPStemmer()

def preprocess(frase):
    tokens = [stemmer.stem(p.lower()) for p in frase.split() if p not in stopwords_pt]
    return tokens

# 4. Criar base com stemming
frases_stemizadas = [(preprocess(frase), emocao) for (frase, emocao) in base_treinamento]

# 5. Extrair palavras únicas
def extrair_palavras(frases):
    todas = []
    for (palavras, emocao) in frases:
        todas.extend(palavras)
    return todas

palavras_todas = extrair_palavras(frases_stemizadas)
frequencia = FreqDist(palavras_todas)
palavras_unicas = list(frequencia.keys())

# 6. Função para extrair características
def extrair_caracteristicas(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavras_unicas:
        caracteristicas[f'contém({palavra})'] = (palavra in doc)
    return caracteristicas

# 7. Preparar base para treinamento
base_completa = [(extrair_caracteristicas(frase), emocao) for (frase, emocao) in frases_stemizadas]

# 8. Treinar classificador Naïve Bayes
classificador = NaiveBayesClassifier.train(base_completa)

# 9. Testar com novas frases
teste1 = preprocess("o lanche está bonito e saboroso")
print("Frase:", teste1, "-> Emoção:", classificador.classify(extrair_caracteristicas(teste1)))

teste2 = preprocess("estou amedrontado com a situação")
print("Frase:", teste2, "-> Emoção:", classificador.classify(extrair_caracteristicas(teste2)))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Base de teste
base_teste = [
    ("o atendimento foi bom", "alegria"),
    ("estou muito irritado", "raiva"),
    ("a entrega foi surpreendente", "surpresa"),
    ("estou triste com o resultado", "tristeza"),
    ("tenho medo de falhar", "medo"),
    ("o sabor está nojento", "desgosto"),
]

# Classificar frases da base de teste
esperado = []
previsto = []
for (frase, emocao) in base_teste:
    tokens = preprocess(frase)
    esperado.append(emocao)
    previsto.append(classificador.classify(extrair_caracteristicas(tokens)))

# Criar matriz de confusão
cm = confusion_matrix(esperado, previsto, labels=classificador.labels())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classificador.labels())
disp.plot(cmap=plt.cm.Blues)
plt.show()
