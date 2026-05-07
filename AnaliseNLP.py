# Sandro Machiniski, Diego Cunha e Lucas Eufrasio

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Baixar recursos necessários
nltk.download('stopwords')
nltk.download('rslp')

# 1. Base de treinamento (frases positivas e negativas do IMDb)
base_treinamento = [
    ("este filme é maravilhoso, adorei cada cena", "positivo"),
    ("um dos melhores filmes que já vi, excelente atuação", "positivo"),
    ("história envolvente e muito bem dirigida", "positivo"),
    ("ótima fotografia e trilha sonora incrível", "positivo"),
    ("personagens bem construídos e roteiro inteligente", "positivo"),
    ("péssimo filme, perdi meu tempo", "negativo"),
    ("atuação fraca e roteiro horrível", "negativo"),
    ("não gostei, muito chato e cansativo", "negativo"),
    ("história sem sentido e direção ruim", "negativo"),
    ("efeitos especiais pobres e enredo previsível", "negativo"),
]

# 2. Pré-processamento
stopwords_pt = stopwords.words('portuguese')
stemmer = RSLPStemmer()

def preprocess(frase):
    tokens = [stemmer.stem(p.lower()) for p in frase.split() if p not in stopwords_pt]
    return tokens

frases_stemizadas = [(preprocess(frase), sentimento) for (frase, sentimento) in base_treinamento]

# 3. Extrair palavras únicas
def extrair_palavras(frases):
    todas = []
    for (palavras, sentimento) in frases:
        todas.extend(palavras)
    return todas

palavras_unicas = list(set(extrair_palavras(frases_stemizadas)))

# 4. Função para extrair características
def extrair_caracteristicas(documento):
    doc = set(documento)
    return {f'contém({palavra})': (palavra in doc) for palavra in palavras_unicas}

# 5. Preparar base para treinamento
base_completa = [(extrair_caracteristicas(frase), sentimento) for (frase, sentimento) in frases_stemizadas]

# 6. Treinar classificador Naïve Bayes
classificador = nltk.NaiveBayesClassifier.train(base_completa)

# 7. Base de teste (novas frases IMDb)
base_teste = [
    ("o filme foi incrível e emocionante", "positivo"),
    ("péssima direção e atores ruins", "negativo"),
    ("excelente história e ótimas atuações", "positivo"),
    ("não recomendo, muito fraco", "negativo"),
    ("um espetáculo visual e muito divertido", "positivo"),
    ("um desastre total, não vale a pena", "negativo"),
]

# 8. Classificar frases da base de teste
esperado = []
previsto = []
for (frase, sentimento) in base_teste:
    tokens = preprocess(frase)
    esperado.append(sentimento)
    previsto.append(classificador.classify(extrair_caracteristicas(tokens)))

# 9. Gerar matriz de confusão
cm = confusion_matrix(esperado, previsto, labels=["positivo", "negativo"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positivo", "negativo"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 10. Mostrar acurácia
acertos = sum([1 for e, p in zip(esperado, previsto) if e == p])
print("Acurácia:", acertos / len(esperado))
