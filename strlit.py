import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss

# Заголовок страницы
st.markdown('<div style="text-align: center; font-size: 36px;">Рекомендация статей на базе SentenceTransformer</div>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 16px;"></div>', unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("/df_full_n.csv")


@st.cache_data
def load_embeddings(model_option):
    if model_option == "cointegrated/rubert-tiny2":
        embd_path = "/Users/marinakochetova/Downloads/embedding_brt.pth"
    elif model_option == "msmarco-distilbert-base-v4":
        embd_path = "/Users/marinakochetova/Downloads/weighted_embeddings.pth"

    embeddings_tensor = torch.load(embd_path)
    embeddings = embeddings_tensor.numpy()
    faiss.normalize_L2(embeddings)
    return embeddings


@st.cache_data
def initialize_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def find_similar_articles(index, embeddings, df, art_ind, num_similar):
    query_embedding = np.array([embeddings[art_ind]], dtype='float32')
    faiss.normalize_L2(query_embedding)
    k = num_similar * 2  # Увеличиваем количество запрашиваемых ближайших соседей
    D, I = index.search(query_embedding, k)
    query_id = df.loc[art_ind, 'id']

    # Исключение исходной статьи из результатов
    seen_ids = set()
    filtered_indices = []
    filtered_distances = []

    for idx, dist in zip(I[0], D[0]):
        article_id = df.loc[idx, 'id']
        if article_id != query_id and article_id not in seen_ids:
            filtered_indices.append(idx)
            filtered_distances.append(dist)
            seen_ids.add(article_id)

    # Фильтрация результатов до num_similar статей
    filtered_indices = filtered_indices[:num_similar]
    filtered_distances = filtered_distances[:num_similar]

    return filtered_indices, filtered_distances


df = load_data()
model_option = st.selectbox("Выберите модель", ["cointegrated/rubert-tiny2", "msmarco-distilbert-base-v4"])
embeddings = load_embeddings(model_option)
index = initialize_faiss_index(embeddings)

art_ind = st.slider("Выберите статью для поиска", min_value=0, max_value=len(df) - 1, value=0)
num_similar = st.slider("Задайте количество статей для подбора похожих", min_value=5, max_value=10, value=5)
generate = st.button("Перегенерировать")

if st.session_state.get('art_ind') != art_ind or st.session_state.get('num_similar') != num_similar or generate:
    st.session_state['art_ind'] = art_ind
    st.session_state['num_similar'] = num_similar
    filtered_indices, filtered_distances = find_similar_articles(index, embeddings, df, art_ind, num_similar)

    st.write(f"Запрос: Статья {df.loc[art_ind, 'id']}, Текст: {df.loc[art_ind, 'text']}")
    st.write("\nБлижайшие статьи к запросу:")
    for i, (idx, dist) in enumerate(zip(filtered_indices, filtered_distances)):
        article_id = df.loc[idx, 'id']
        text = df.loc[idx, 'text']
        similarity = dist
        st.write(f"Статья {article_id}")
        st.write(f"Текст: {text}")
        st.write(f"Косинусное сходство: {similarity}")

