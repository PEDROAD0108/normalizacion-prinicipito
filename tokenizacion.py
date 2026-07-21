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

