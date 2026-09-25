"""
Sistem Rekomendasi Berita Berbasis Judul (Content-Based Filtering)
====================================================================
Skenario: Sebuah website news membuat rekomendasi berita HANYA
berdasarkan judul artikel. Seorang pengunjung membaca artikel
judul pertama, lalu sistem merekomendasikan artikel lain yang
judulnya paling mirip.

Pendekatan: TF-IDF (Term Frequency-Inverse Document Frequency)
untuk mengubah tiap judul menjadi vektor angka, lalu Cosine
Similarity untuk mengukur kemiripan antar judul.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Baca data judul artikel
df = pd.read_csv("news_rec.csv")
df.columns = ["judul"]
df["judul"] = df["judul"].str.strip()
df = df.dropna(subset=["judul"])
df = df[df["judul"] != ""].reset_index(drop=True)

print("=== Daftar Artikel ===")
for i, judul in enumerate(df["judul"]):
    print(f"[{i}] {judul}")

# 2. Ubah semua judul menjadi vektor TF-IDF
#    stop_words tidak dipakai bawaan sklearn karena teks berbahasa
#    Indonesia, jadi kita pakai daftar stopword sederhana manual.
stopwords_id = [
    "yang", "di", "ke", "dari", "ini", "itu", "dan", "akan",
    "usai", "sore", "akhir", "pekan", "ini,", "bagaimana",
]

vectorizer = TfidfVectorizer(stop_words=stopwords_id)
tfidf_matrix = vectorizer.fit_transform(df["judul"])

# 3. Artikel yang sedang dibaca pengunjung = artikel pertama (index 0)
artikel_dibaca_idx = 0
print(f"\n=== Pengunjung sedang membaca ===\n[{artikel_dibaca_idx}] {df['judul'][artikel_dibaca_idx]}")

# 4. Hitung cosine similarity antara artikel yang dibaca dengan semua artikel lain
sim_scores = cosine_similarity(
    tfidf_matrix[artikel_dibaca_idx], tfidf_matrix
).flatten()

# 5. Susun hasil, urutkan dari yang paling mirip, buang artikel itu sendiri
hasil = (
    pd.DataFrame({"judul": df["judul"], "skor_similarity": sim_scores})
    .drop(index=artikel_dibaca_idx)
    .sort_values(by="skor_similarity", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Rekomendasi Artikel (paling mirip di atas) ===")
for i, row in hasil.iterrows():
    print(f"{i+1}. ({row['skor_similarity']:.3f})  {row['judul']}")

# 6. Simpan hasil ke CSV
hasil.to_csv("hasil_rekomendasi.csv", index=False)
print("\nHasil rekomendasi disimpan ke hasil_rekomendasi.csv")
