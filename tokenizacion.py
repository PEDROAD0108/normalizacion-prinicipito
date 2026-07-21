import spacy

# Cargar el modelo de spaCy en español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo...")
    from spacy.cli import download

    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Leer el archivo de texto
with open("cap1_principito.txt", "r", encoding="utf-8") as f:
    texto_principito = f.read()

print(
    f"Texto cargado con éxito. "
    f"Longitud: {len(texto_principito)} caracteres."
)

# Tokenización
# spaCy procesa el texto y crea un objeto doc
doc = nlp(texto_principito)

# Mostrar la cantidad y los primeros 20 tokens
print(f"\n--- 1. Tokenización (Total tokens: {len(doc)}) ---")
print([token.text for token in doc[:20]])

# 2. FILTRADO DE STOP WORDS
# Separamos las palabras relevantes de las palabras gramaticales

tokens_relevantes = []
tokens_ruido = []

for token in doc:
    # Comprobamos que no sea stop word, puntuación ni espacio
    if not token.is_stop and not token.is_punct and token.text.strip():
        tokens_relevantes.append(token.text)

    # Guardamos las stop words como ruido
    elif token.is_stop:
        tokens_ruido.append(token.text)

print("\n--- 2. Filtrado de Stop Words ---")
print(f"Palabras eliminadas (Ruido): {tokens_ruido[:10]}...")
print(f"Palabras conservadas (Contenido): {tokens_relevantes[:10]}...")
print(
    f"Reducción de tamaño: de {len(doc)} "
    f"a {len(tokens_relevantes)} tokens."
)


# 3. LEMATIZACIÓN Y NORMALIZACIÓN FINAL
# Reducimos las palabras a su raíz (Lema) y estandarizamos a minúsculas
# Objetivo: Que "hablo", "hablaré" y "habla" cuenten como el mismo concepto: "hablar"

tokens_normalizados = []
cambios_interesantes = []

for token in doc:
    # Aplicamos los mismos filtros de calidad que en el paso 2
    if not token.is_stop and not token.is_punct and token.text.strip():
        
        # AQUÍ OCURRE LA MAGIA: 
        # 1. Extraemos el lema (token.lemma_)
        # 2. Convertimos a minúsculas (.lower())
        lema = token.lemma_.lower()
        tokens_normalizados.append(lema)
        
        # Para fines educativos: Guardamos casos donde la palabra cambió drásticamente
        # Ej: "fui" -> "ir"
        if token.text.lower() != lema:
            cambios_interesantes.append(f"{token.text} ➡ {lema}")

print(f"\n--- 3. Lematización y Normalización ---")
print(f"Total de tokens procesados: {len(tokens_normalizados)}")
print(f"Ejemplos de transformaciones (Palabra original ➡ Lema):")
# Mostramos solo los primeros 5 cambios para no saturar la pantalla
print(cambios_interesantes[:10]) 

print(f"\nResultado final (Primeros 10 tokens):")
print(tokens_normalizados[:10])


# 3. STEMMING VS LEMATIZACIÓN

import pandas as pd
from nltk.stem import SnowballStemmer

# Configurar el stemmer para español
stemmer = SnowballStemmer("spanish")

# Lista para guardar la comparación
data_comparativa = []

for token in doc:
    # Analizar únicamente palabras
    if not token.is_punct and not token.is_space:

        # A. Stemming: corta los sufijos de las palabras
        raiz_stem = stemmer.stem(token.text)

        # B. Lematización: obtiene la forma base usando spaCy
        lema = token.lemma_

        data_comparativa.append(
            {
                "Original": token.text,
                "Stemming (Corte)": raiz_stem,
                "Lematización (Diccionario)": lema,
                "¿Coinciden?": raiz_stem == lema,
            }
        )

# Crear una tabla con pandas
df = pd.DataFrame(data_comparativa)

print("\n--- 3. Stemming vs Lematización ---")

# Buscar palabras específicas para mostrar la comparación
palabras_interesantes = [
    "hombres",
    "olvidado",
    "eres",
    "domesticado",
    "invisible",
    "ojos",
]

filtro = df[df["Original"].str.lower().isin(palabras_interesantes)]

# 4. REPRESENTACIÓN VECTORIAL

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Crear un corpus: cada elemento será una oración limpia y lematizada
corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if not token.is_punct
        and not token.is_space
        and not token.is_stop
    ]

    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(
    f"\nTotal de oraciones procesadas: "
    f"{len(corpus_lematizado)}"
)

# A. BAG OF WORDS
bow_vectorizer = CountVectorizer()

# Entrenar el vectorizador y transformar el corpus
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

print("\n--- 4. Representación Bag of Words ---")
print(f"Dimensiones de la matriz: {X_bow.shape}")
print(f"Cantidad de palabras únicas: {len(bow_vectorizer.vocabulary_)}")

# Mostrar algunas palabras del vocabulario
vocabulario_bow = bow_vectorizer.get_feature_names_out()
print(f"Primeras palabras del vocabulario: {vocabulario_bow[:20]}")

# Mostrar el vector de la primera oración
print("\nPrimera oración lematizada:")
print(corpus_lematizado[0])

print("\nVector numérico de la primera oración:")
print(X_bow[0].toarray())


corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if not token.is_punct
        and not token.is_space
        and not token.is_stop
    ]

    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"Total de oraciones procesadas: {len(corpus_lematizado)}")


# 5. GRÁFICAS 3D DE BAG OF WORDS Y TF-IDF

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.decomposition import PCA


def graficar_palabras_3d(
    ax,
    matriz,
    vocabulario,
    titulo,
    color_puntos,
):
    # Transponer para que las filas representen palabras
    matriz_palabras = matriz.T

    # Reducir los datos a tres dimensiones mediante PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())

    # Extraer las coordenadas
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Crear los puntos
    ax.scatter(
        x,
        y,
        z,
        c=color_puntos,
        s=80,
        edgecolors="k",
        alpha=0.8,
        depthshade=True,
    )

    # Colocar el nombre de cada palabra
    for i, palabra in enumerate(vocabulario):
        ax.text(
            x[i],
            y[i],
            z[i] + 0.1,
            palabra,
            fontsize=9,
        )

    ax.set_title(titulo)
    ax.set_xlabel("Comp. Principal 1")
    ax.set_ylabel("Comp. Principal 2")
    ax.set_zlabel("Comp. Principal 3")

    # Líneas de referencia
    ax.plot(
        [0, 0],
        [0, 0],
        [z.min(), z.max()],
        c="grey",
        ls="--",
        lw=0.5,
        alpha=0.3,
    )

    ax.plot(
        [x.min(), x.max()],
        [0, 0],
        [0, 0],
        c="grey",
        ls="--",
        lw=0.5,
        alpha=0.3,
    )

    ax.plot(
        [0, 0],
        [y.min(), y.max()],
        [0, 0],
        c="grey",
        ls="--",
        lw=0.5,
        alpha=0.3,
    )


# Crear la figura con las dos gráficas
fig = plt.figure(figsize=(18, 8))

# A. Bag of Words
ax1 = fig.add_subplot(121, projection="3d")

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
vocab_bow = bow_vectorizer.get_feature_names_out()

graficar_palabras_3d(
    ax1,
    X_bow,
    vocab_bow,
    "Espacio BoW 3D (Conteos)",
    "orange",
)

# B. TF-IDF
ax2 = fig.add_subplot(122, projection="3d")

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(
    corpus_lematizado
)
vocab_tfidf = tfidf_vectorizer.get_feature_names_out()

graficar_palabras_3d(
    ax2,
    X_tfidf,
    vocab_tfidf,
    "Espacio TF-IDF 3D (Importancia)",
    "teal",
)

plt.tight_layout()

# Guardar la gráfica como evidencia
plt.savefig(
    "comparacion_bow_tfidf_3d.png",
    dpi=300,
    bbox_inches="tight",
)

# Mostrar la gráfica
plt.show()
