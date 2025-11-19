import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_job_clusters(jobs_data, model, n_clusters=3):
    """
    Semantic Clustering:
    1. Uses SBERT Embeddings to group jobs (Semantic similarity).
    2. Uses TF-IDF to generate human-readable labels for those groups.
    """
    if len(jobs_data) < n_clusters:
        return jobs_data

    # 1. Prepare Text
    texts = [j['text'] for j in jobs_data]
    
    # 2. Embed (Use the SBERT model passed from app.py)
    # This is much better than raw TF-IDF because it captures context.
    embeddings = model.encode(texts)

    # 3. Cluster (K-Means on Dense Vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 4. Generate Labels (What do we call "Cluster 1"?)
    # We treat all text in a cluster as one big document and run TF-IDF to find unique words.
    df = pd.DataFrame({'text': texts, 'cluster': cluster_labels})
    cluster_names = {}

    for c_id in range(n_clusters):
        # Combine all JDs in this cluster into one string
        cluster_text = " ".join(df[df['cluster'] == c_id]['text'].tolist())
        
        # Run mini TF-IDF to extract the specific keywords for this group
        # We assume the other clusters are the "background" corpus
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        try:
            # We fit on just this cluster vs the whole set to find what makes IT unique
            tfidf = vectorizer.fit_transform([cluster_text])
            top_words = vectorizer.get_feature_names_out()
            label = ", ".join([w.title() for w in top_words])
        except ValueError:
            label = f"Group {c_id+1}"
            
        cluster_names[c_id] = label

    # 5. Assign Labels back to Data
    for idx, job in enumerate(jobs_data):
        c_id = cluster_labels[idx]
        job['cluster_id'] = int(c_id)
        job['cluster_name'] = cluster_names[c_id]

    return jobs_data
