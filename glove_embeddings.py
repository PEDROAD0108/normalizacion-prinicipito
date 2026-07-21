import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# CARGAR MODELO GLOVE
# ---------------------------------------------------------

print("Cargando modelo GloVe...")

model = api.load("glove-wiki-gigaword-50")

print("Modelo GloVe cargado correctamente.")


# ---------------------------------------------------------
# 1. HEATMAP DE EMBEDDINGS
# ---------------------------------------------------------

words = [
    "man",
    "woman",
    "boy",
    "girl",
    "king",
    "queen",
]

# Obtener los vectores de 50 dimensiones
vecs = np.array([model[word] for word in words])

plt.figure(figsize=(14, 4))

plt.imshow(
    vecs,
    cmap="RdBu_r",
    aspect="auto",
)

plt.yticks(
    range(len(words)),
    words,
    fontsize=13,
)

plt.xlabel("Dimensión")
plt.title("Embeddings GloVe (50d)")
plt.colorbar(label="Valor")
plt.tight_layout()

# Guardar la primera gráfica
plt.savefig(
    "heatmap.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ---------------------------------------------------------
# 2. ANALOGÍA: KING - MAN + WOMAN
# ---------------------------------------------------------

king = model["king"]
man = model["man"]
woman = model["woman"]
queen = model["queen"]

# El resultado debería ser aproximadamente queen
result = king - man + woman

labels = [
    "king",
    "- man",
    "+ woman",
    "= resultado",
    "queen (real)",
]

vecs_analogia = np.array(
    [
        king,
        -man,
        woman,
        result,
        queen,
    ]
)

fig, ax = plt.subplots(figsize=(14, 4))

imagen = ax.imshow(
    vecs_analogia,
    cmap="RdBu_r",
    aspect="auto",
)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(
    labels,
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel("Dimensión")
ax.set_title(
    "Analogía: king - man + woman ≈ queen",
    fontsize=14,
    fontweight="bold",
)

plt.colorbar(
    imagen,
    ax=ax,
    label="Valor",
)

# Calcular similitud coseno
similitud = cosine_similarity(
    [result],
    [queen],
)[0][0]

ax.annotate(
    f"Similitud coseno(resultado, queen) = {similitud:.4f}",
    xy=(0.5, -0.22),
    xycoords="axes fraction",
    ha="center",
    fontsize=12,
    fontstyle="italic",
)

plt.tight_layout()

# Guardar la segunda gráfica
plt.savefig(
    "analogy_heatmap.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
plt.close()

print("\nGráficas generadas correctamente:")
print("- heatmap.png")
print("- analogy_heatmap.png")
print(f"\nSimilitud entre el resultado y queen: {similitud:.4f}")
