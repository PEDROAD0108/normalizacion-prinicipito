import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# 1. PREPARACIÓN DE DATOS
# ---------------------------------------------------------

nlp = spacy.load("es_core_news_sm")

with open(
    "cap1_principito.txt",
    "r",
    encoding="utf-8",
) as archivo:
    texto_principito = archivo.read()

print("Procesando texto con spaCy...")

doc = nlp(texto_principito)

sentences = []

for sent in doc.sents:
    # Lematizar y eliminar stop words, puntuación y espacios
    tokens = [
        token.lemma_.lower()
        for token in sent
        if not token.is_stop
        and not token.is_punct
        and token.text.strip()
    ]

    # Conservar oraciones con más de un token
    if len(tokens) > 1:
        sentences.append(tokens)

print(f"Total de oraciones procesadas: {len(sentences)}")

if sentences:
    print(f"Ejemplo de tokens: {sentences[0]}")
else:
    raise ValueError("No se obtuvieron oraciones para entrenar el modelo.")


# ---------------------------------------------------------
# 2. ENTRENAMIENTO DE WORD2VEC
# ---------------------------------------------------------

print("\nEntrenando red neuronal Word2Vec...")

model = Word2Vec(
    sentences=sentences,
    vector_size=10,
    window=5,
    min_count=1,
    workers=multiprocessing.cpu_count(),
    seed=42,
)

print("Modelo Word2Vec entrenado correctamente.")


# ---------------------------------------------------------
# 3. EXPLORACIÓN SEMÁNTICA
# ---------------------------------------------------------

def mostrar_similares(palabra):
    try:
        similares = model.wv.most_similar(
            palabra,
            topn=3,
        )

        print(
            f"\nPalabras más cercanas semánticamente "
            f"a '{palabra}':"
        )

        for palabra_similar, puntuacion in similares:
            print(
                f"  - {palabra_similar} "
                f"(Similitud: {puntuacion:.4f})"
            )

    except KeyError:
        print(
            f"\nLa palabra '{palabra}' "
            f"no está en el vocabulario."
        )


mostrar_similares("cordero")
mostrar_similares("esencial")


# ---------------------------------------------------------
# 4. VISUALIZACIÓN 3D
# ---------------------------------------------------------

vocabulario = list(model.wv.index_to_key)
vectores = model.wv[vocabulario]

# Reducir los vectores de 10 a 3 dimensiones
pca = PCA(n_components=3)
vectores_3d = pca.fit_transform(vectores)

# Crear un DataFrame
df_3d = pd.DataFrame(
    vectores_3d,
    columns=["x", "y", "z"],
)

df_3d["palabra"] = vocabulario

# Crear gráfica
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df_3d["x"],
    df_3d["y"],
    df_3d["z"],
    c="crimson",
    s=80,
    edgecolors="white",
    alpha=0.8,
)

# Agregar las palabras como etiquetas
for _, row in df_3d.iterrows():
    ax.text(
        row["x"],
        row["y"],
        row["z"],
        f" {row['palabra']}",
        size=10,
    )

ax.set_title(
    "Espacio Semántico (Word Embeddings) - El Principito",
    fontsize=14,
)

ax.set_xlabel("Dimensión Latente 1")
ax.set_ylabel("Dimensión Latente 2")
ax.set_zlabel("Dimensión Latente 3")

plt.tight_layout()

# Guardar la gráfica
plt.savefig(
    "word2vec_principito_3d.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ---------------------------------------------------------
# 5. MOSTRAR EL VECTOR DE UNA PALABRA
# ---------------------------------------------------------

print(
    "\nAsí ve la máquina la palabra 'zorro' "
    "(vector de 10 dimensiones):"
)

try:
    print(model.wv["zorro"])
except KeyError:
    print(
        "La palabra 'zorro' no aparece en este capítulo. "
        "Intentando con 'cordero':"
    )

    try:
        print(model.wv["cordero"])
    except KeyError:
        print(
            "La palabra 'cordero' tampoco está "
            "en el vocabulario."
        )
