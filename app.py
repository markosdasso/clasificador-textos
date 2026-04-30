# Para correr la app en cmd anaconda "streamlit run Classificator_app.py"# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import io
import os
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Clasificador Inteligente", layout="wide")
DATASET_PATH = "dataset_entrenado.csv"

# =========================
# FUNCIONES
# =========================

def limpiar_texto(df):
    df = df.copy()
    df["texto"] = (
        df["texto"]
        .astype(str)
        .str.replace("\n", " ")
        .str.replace("\r", " ")
        .str.replace("\t", " ")
        .str.strip()
    )
    return df


def validar_dataset(df):
        
    df = limpiar_textos_entrada(df)

    posibles_etiquetas = ["etiquetas", "labels", "categoria", "tag", "tags"]

    col_etiquetas = next((c for c in df.columns if c in posibles_etiquetas), None)

    if col_etiquetas is None:
        raise ValueError(
            f"No se encontró columna de etiquetas. Columnas: {df.columns.tolist()}"
        )

    df = df.rename(columns={col_etiquetas: "etiquetas"})

    df["etiquetas"] = df["etiquetas"].astype(str).str.strip()
    df = df[df["etiquetas"] != ""]

    if "origen" not in df.columns:
            df["origen"] = "modelo"
            
    return df.drop_duplicates("texto").reset_index(drop=True)


def guardar_dataset(df):
    df.to_csv(DATASET_PATH, index=False, sep=";", encoding="utf-8")


@st.cache_data
def cargar_dataset():
    if os.path.exists(DATASET_PATH):
        return validar_dataset(pd.read_csv(DATASET_PATH, sep=";"))
    return pd.DataFrame(columns=["texto", "etiquetas"])


def get_hash(df):
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

def limpiar_textos_entrada(df, posibles_cols=None):

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if posibles_cols is None:
        posibles_cols = ["texto", "text", "descripcion", "detalle"]

    col_texto = next((c for c in df.columns if c in posibles_cols), None)

    if col_texto is None:
        raise ValueError(f"No se encontró columna de texto. Columnas: {df.columns.tolist()}")

    df = df.rename(columns={col_texto: "texto"})

    # 🔥 limpieza fuerte
    df["texto"] = df["texto"].astype(str)
    df["texto"] = df["texto"].str.replace("\n", " ")
    df["texto"] = df["texto"].str.replace("\r", " ")
    df["texto"] = df["texto"].str.replace("\t", " ")
    df["texto"] = df["texto"].str.strip()

    df = df[df["texto"] != ""]
    df = df.reset_index(drop=True)

    return df
# =========================
# MODELO
# =========================

def crear_modelo():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])


def entrenar(df):
    if len(df) < 2:
        return None, None

    y = df["etiquetas"].apply(lambda x: x.split("|"))
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    modelo = crear_modelo()
    modelo.fit(df["texto"], y_bin)

    return modelo, mlb


def obtener_modelo(df):

    if len(df) < 2:
        return None, None

    h = get_hash(df)

    if "modelo" not in st.session_state or st.session_state.get("hash") != h:
        modelo, mlb = entrenar(df)

        if modelo is None:
            return None, None

        st.session_state.modelo = modelo
        st.session_state.mlb = mlb
        st.session_state.hash = h

    return st.session_state.modelo, st.session_state.mlb

# =========================
# PREDICCIÓN
# =========================

def predecir_topk(modelo, mlb, texto, threshold=0.3, top_k=3):

    probs = modelo.predict_proba([texto])[0]

    etiquetas = [
        mlb.classes_[i]
        for i, p in enumerate(probs)
        if p > threshold
    ]

    top_idx = probs.argsort()[::-1][:top_k]

    sugerencias = [
        (mlb.classes_[i], float(probs[i]))
        for i in top_idx
    ]

    probs_dict = {
        mlb.classes_[i]: float(probs[i])
        for i in range(len(probs))
    }   

    return etiquetas, sugerencias, float(probs.max()), probs_dict


def predecir_similitud(df, texto):
    vec = TfidfVectorizer()
    X = vec.fit_transform(df["texto"])
    q = vec.transform([texto])
    sims = cosine_similarity(q, X)[0]
    i = sims.argmax()
    return df.iloc[i]["etiquetas"].split("|"), float(sims[i])

# =========================
# EXPORT
# =========================

def to_excel(df):
    out = io.BytesIO()
    df.to_excel(out, index=False)
    return out.getvalue()

# =========================
# STATE
# =========================

if "dataset" not in st.session_state:
    st.session_state.dataset = cargar_dataset()

if "lote" not in st.session_state:
    st.session_state.lote = pd.DataFrame()

# =========================
# UI
# =========================

st.title("🧠 Clasificador Inteligente")

# -------------------------
# CARGA DATASET
# -------------------------

archivo = st.file_uploader("📥 Subir dataset entrenamiento", type=["csv","xlsx"])

if archivo:
    df = pd.read_csv(archivo, sep=";") if archivo.name.endswith(".csv") else pd.read_excel(archivo)
    df = validar_dataset(df)

    st.session_state.dataset = pd.concat([st.session_state.dataset, df])
    guardar_dataset(st.session_state.dataset)
    cargar_dataset.clear()

    st.success(f"{len(df)} registros cargados")

# -------------------------
# CONFIG
# -------------------------

tipo = st.selectbox("Modelo", ["Multilabel", "Similitud"])
threshold = st.slider("Threshold", 0.1, 0.9, 0.3)

# -------------------------
# INPUT
# -------------------------

texto = st.text_area("Texto a clasificar")

if st.button("Clasificar"):

    modelo, mlb = obtener_modelo(st.session_state.dataset)

    if tipo == "Similitud":
        pred, conf = predecir_similitud(st.session_state.dataset, texto)
        sugerencias = []
    else:
        pred, sugerencias, conf, probs = predecir_topk(modelo, mlb, texto, threshold)

    st.write("### Resultado")
    st.write(pred, conf)

    if sugerencias:
        st.write("### Sugerencias")
        for e,s in sugerencias:
            st.write(f"{e} ({round(s,3)})")

# -------------------------
# LOTE
# -------------------------

archivo_lote = st.file_uploader("📊 Subir lote", type=["csv","xlsx"], key="lote_file")

if archivo_lote:
    try:
        df_lote = pd.read_csv(archivo_lote, sep=";") if archivo_lote.name.endswith(".csv") else pd.read_excel(archivo_lote)

        # limpieza
        df_lote = limpiar_textos_entrada(df_lote)

        st.write("Columnas detectadas:", df_lote.columns.tolist())
        st.write("Tipos en texto:")
        st.write(df_lote["texto"].apply(type).value_counts())

        if st.button("Procesar lote"):

            modelo, mlb = obtener_modelo(st.session_state.dataset)

            if modelo is None:
                st.error("Dataset insuficiente para entrenar modelo")
                st.stop()

            resultados = []
            errores = []

            total = len(df_lote)
            # métricas
            conf_acum = 0
            low_conf_count = 0
            label_counter = {}

            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_text = st.empty()

            for i, t in enumerate(df_lote["texto"]):
            
                try:
                    t = str(t)
            
                    if tipo == "Similitud":
                        pred, conf = predecir_similitud(st.session_state.dataset, t)
                        probs = {}
                    else:
                        pred, sugerencias, conf, probs = predecir_topk(modelo, mlb, t, threshold)
            
                    resultados.append({
                        "texto": t,
                        "etiquetas": "|".join(pred),
                        "confianza": conf,
                        "origen": "modelo",
                        "probs": "|".join([f"{k}:{round(v,3)}" for k,v in probs.items()]) if probs else ""
                    })
                    # métricas
                    conf_acum += conf
                    
                    if conf < threshold:
                        low_conf_count += 1
                    
                    for label in pred:
                        label_counter[label] = label_counter.get(label, 0) + 1
            
                except Exception as e:
                    errores.append({
                        "fila": i,
                        "texto": str(t),
                        "error": str(e)
                    })
            
                # 🔥 actualizar progreso
                if i % 10 == 0 or i == total - 1:

                    progress = int((i + 1) / total * 100)
                    progress_bar.progress(progress)
                
                    status_text.text(f"Procesando: {i+1}/{total} ({progress}%)")
                
                    # métricas
                    avg_conf = conf_acum / (i + 1)
                    low_conf_pct = (low_conf_count / (i + 1)) * 100
                
                    top_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_labels_str = ", ".join([f"{k} ({v})" for k,v in top_labels])
                
                    metric_text.text(
                        f"Conf promedio: {round(avg_conf,3)} | "
                        f"Baja confianza: {round(low_conf_pct,1)}% | "
                        f"Top etiquetas: {top_labels_str}"
                    )
            st.session_state.lote = pd.DataFrame(resultados)

            st.success(f"Lote procesado: {len(resultados)} ok / {len(errores)} errores")

            if errores:
                st.warning("Errores detectados:")
                st.dataframe(pd.DataFrame(errores))

    except Exception as e:
        st.error(f"Error al cargar lote: {e}")
# -------------------------
# CORRECCIÓN INTELIGENTE
# -------------------------

if not st.session_state.lote.empty:

    df = st.session_state.lote.sort_values("confianza")
    st.dataframe(df)

    df_baja = df[df["confianza"] < threshold]

    if not df_baja.empty:

        fila = st.selectbox("Seleccionar caso", df_baja.index)
        texto_corr = df_baja.loc[fila, "texto"]

        modelo, mlb = obtener_modelo(st.session_state.dataset)
        _, sugerencias, _, _ = predecir_topk(modelo, mlb, texto_corr, threshold)

        st.write("### 🤖 Sugerencias")
        for e, s in sugerencias:
            st.write(f"{e} ({round(s,3)})")

        # 🔹 input nueva etiqueta (PRIMERO)
        nueva = st.text_input("Nueva etiqueta", key="input_nueva_etiqueta")

        # 🔹 etiquetas del dataset
        etiquetas_dataset = set(
            e for sub in st.session_state.dataset["etiquetas"]
            for e in sub.split("|")
        )

        # 🔹 inicializar nuevas etiquetas en sesión
        if "nuevas_etiquetas" not in st.session_state:
            st.session_state.nuevas_etiquetas = set()

        # 🔹 capturar nueva etiqueta
        if nueva and nueva.strip() != "":
            st.session_state.nuevas_etiquetas.add(nueva.strip())

        # 🔹 unión total
        etiquetas_existentes = sorted(
            etiquetas_dataset.union(st.session_state.nuevas_etiquetas)
        )

        # 🔹 botón usar sugerencias
        if st.button("⚡ Usar sugerencias"):
            st.session_state.sel = [e[0] for e in sugerencias]

        # 🔹 multiselect
        seleccion = st.multiselect(
            "Etiquetas",
            etiquetas_existentes,
            default=list(set(st.session_state.get("sel", []))),
            key="multi_etiquetas"
        )

        # 🔹 guardar corrección
        if st.button("Guardar"):

            nuevo = pd.DataFrame([{
                "texto": texto_corr,
                "etiquetas": "|".join(seleccion),
                "origen": "manual"
            }])

        # guardar en dataset
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, nuevo],
                ignore_index=True
    )

            guardar_dataset(st.session_state.dataset)
            cargar_dataset.clear()

    # 🔥 actualizar lote en pantalla
            st.session_state.lote.at[fila, "etiquetas"] = "|".join(seleccion)
            st.session_state.lote.at[fila, "confianza"] = 1.0
            st.session_state.lote.at[fila, "origen"] = "manual"

    # limpiar UI
            st.session_state.pop("input_nueva_etiqueta", None)
            st.session_state.pop("multi_etiquetas", None)
            st.session_state.pop("sel", None)

            st.success("Aprendido ✅")

    # 🔥 refrescar pantalla
            st.rerun()
# -------------------------
# EXPORT
# -------------------------

if not st.session_state.lote.empty:
    st.download_button("📥 Excel", to_excel(st.session_state.lote), "lote.xlsx")

# -------------------------
# RESET
# -------------------------

if st.button("Reset"):
    st.session_state.clear()
   
# -------------------------
# BOTON DESCARGA DATASET ENTRENADO
# -------------------------
st.download_button(
    "📥 Descargar dataset entrenado",
    to_excel(st.session_state.dataset),
    "dataset_entrenado.xlsx"
)
