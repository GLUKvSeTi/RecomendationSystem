# ============================================================
# ОБОГАЩЕНИЕ ДАННЫХ ЧЕРЕЗ OPEN LIBRARY API
# 100% самодостаточная ячейка
# ============================================================
import requests, json, os, urllib3
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# 0. ПОДГОТОВКА: создаём всё что нужно
# ============================================================
books = books.reset_index(drop=True)

# Заполняем пропуски
books['Title']     = books['Title'].fillna('').astype(str)
books['Author']    = books['Author'].fillna('').astype(str)
books['Publisher'] = books['Publisher'].fillna('').astype(str)

# Маппинги
book_isbns = books['ISBN'].values
isbn2tfidf = {isbn: i for i, isbn in enumerate(book_isbns)}

# Базовый контент
books['content'] = books['Title'] + ' ' + books['Author'] + ' ' + books['Publisher']

# Базовый TF-IDF
tfidf_base = TfidfVectorizer(stop_words='english', max_features=10000,
                              ngram_range=(1,2), min_df=2)
tfidf_matrix_base = tfidf_base.fit_transform(books['content'])
print(f"Базовый TF-IDF: {tfidf_matrix_base.shape}")

# ============================================================
# 1. КЕШ
# ============================================================
CACHE_FILE = 'openlibrary_cache.json'
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    print(f"Загружено из кеша: {len(cache)} записей")
else:
    cache = {}
    print("Кеш пуст, начинаем с нуля")

# ============================================================
# 2. ФУНКЦИЯ ЗАПРОСА К OPEN LIBRARY
# ============================================================
def fetch_book(isbn):
    try:
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        r = requests.get(url, timeout=5, verify=False)
        info = r.json().get(f"ISBN:{isbn}", {})
        subjects = ' '.join([s['name'] for s in info.get('subjects', [])])
        return isbn, subjects
    except Exception:
        return isbn, ''

# ============================================================
# 3. СКАЧИВАЕМ
# ============================================================
to_fetch = [isbn for isbn in books['ISBN'].values if isbn not in cache]
print(f"Нужно скачать: {len(to_fetch)} книг")

if to_fetch:
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_book, isbn): isbn for isbn in to_fetch}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Open Library"):
            isbn, subjects = f.result()
            cache[isbn] = subjects
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"Кеш сохранён: {len(cache)} записей")

# ============================================================
# 4. ДОБАВЛЯЕМ SUBJECTS В BOOKS
# ============================================================
books['subjects'] = books['ISBN'].map(lambda x: cache.get(x, ''))
enriched = (books['subjects'].str.len() > 0).sum()
print(f"Обогащено: {enriched} из {len(books)} ({100*enriched/len(books):.1f}%)")

# Пример
sample_enriched = books[books['subjects'].str.len() > 0][['Title','subjects']].head(3)
print("\nПримеры обогащённых данных:")
for _, row in sample_enriched.iterrows():
    print(f"  «{row['Title'][:50]}» → {row['subjects'][:100]}")

# ============================================================
# 5. ОБОГАЩЁННЫЙ TF-IDF
# ============================================================
books['content_enriched'] = (
    books['Title'] + ' ' +
    books['Author'] + ' ' +
    books['Publisher'] + ' ' +
    books['subjects'] + ' ' + books['subjects']   # двойной вес тем
)

tfidf_enriched = TfidfVectorizer(stop_words='english', max_features=15000,
                                  ngram_range=(1,2), min_df=2)
tfidf_matrix_enriched = tfidf_enriched.fit_transform(books['content_enriched'])
print(f"\nОбогащённый TF-IDF: {tfidf_matrix_enriched.shape}")

# ============================================================
# 6. ФАБРИКА РЕКОМЕНДЕРА
# ============================================================
def make_content_rec(matrix):
    def rec(uid, top_n=10):
        ur = train[train['User-ID']==uid]
        vecs, w = [], []
        for _, r in ur.iterrows():
            if r['ISBN'] in isbn2tfidf:
                vecs.append(matrix[isbn2tfidf[r['ISBN']]])
                w.append(r['Rating'])
        if not vecs:
            return []
        prof = vstack(vecs).multiply(np.array(w)[:, None]).sum(axis=0) / sum(w)
        sims = cosine_similarity(np.asarray(prof), matrix).flatten()
        seen = set(ur['ISBN'])
        cands = [(book_isbns[i], sims[i]) for i in range(len(sims))
                 if book_isbns[i] not in seen]
        cands.sort(key=lambda x: -x[1])
        return [c[0] for c in cands[:top_n]]
    return rec

# ============================================================
# 7. СРАВНЕНИЕ ДО / ПОСЛЕ
# ============================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ CONTENT-MODEL ДО И ПОСЛЕ ОБОГАЩЕНИЯ")
print("="*60)

content_results = []
content_results.append(
    evaluate(make_content_rec(tfidf_matrix_base), test,
             name="Content (Title+Author+Publisher)")
)
content_results.append(
    evaluate(make_content_rec(tfidf_matrix_enriched), test,
             name="Content + OpenLibrary subjects")
)
results_df = pd.DataFrame(content_results)
print(results_df.to_string(index=False))

# ============================================================
# 8. ОБНОВЛЯЕМ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ОСТАЛЬНОГО ПАЙПЛАЙНА
# ============================================================
tfidf_matrix      = tfidf_matrix_enriched
recommend_content = make_content_rec(tfidf_matrix)

# Демонстрация
if 'sample_user' in dir():
    print(f"\nОбогащённый Content рекомендует для юзера {sample_user}:")
    print(show_books(recommend_content(sample_user)))

print("\n✓ Готово. Переменная tfidf_matrix обновлена на обогащённую версию.")
print("✓ Гибрид и остальные модели автоматически подхватят новое представление.")