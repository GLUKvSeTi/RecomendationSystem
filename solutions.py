# ============================================================
# ОБОГАЩЕНИЕ: Open Library + Google Books (комбо)
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

# === 0. Подготовка ===
books = books.reset_index(drop=True)
books['Title']     = books['Title'].fillna('').astype(str)
books['Author']    = books['Author'].fillna('').astype(str)
books['Publisher'] = books['Publisher'].fillna('').astype(str)

book_isbns = books['ISBN'].values
isbn2tfidf = {isbn: i for i, isbn in enumerate(book_isbns)}

books['content'] = books['Title']+' '+books['Author']+' '+books['Publisher']

tfidf_base = TfidfVectorizer(stop_words='english', max_features=10000,
                              ngram_range=(1,2), min_df=2)
tfidf_matrix_base = tfidf_base.fit_transform(books['content'])
print(f"Базовый TF-IDF: {tfidf_matrix_base.shape}")

# === 1. Кеш ===
CACHE_FILE = 'book_meta_cache.json'
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Загружено из кеша: {len(cache)}")
else:
    cache = {}

# === 2. Open Library: пробуем два endpoint-а ===
def fetch_openlibrary(isbn):
    """Пробует сначала /isbn/.json, потом /api/books"""
    try:
        # endpoint 1: прямой
        r = requests.get(f"https://openlibrary.org/isbn/{isbn}.json",
                         timeout=5, verify=False, allow_redirects=True)
        if r.status_code == 200:
            data = r.json()
            subjects = data.get('subjects', [])
            if subjects:
                return ' '.join(subjects)
            # Если subjects пустой — пробуем подтянуть work
            works = data.get('works', [])
            if works:
                work_key = works[0].get('key')
                if work_key:
                    rw = requests.get(f"https://openlibrary.org{work_key}.json",
                                      timeout=5, verify=False)
                    if rw.status_code == 200:
                        wdata = rw.json()
                        subjects = wdata.get('subjects', [])
                        if subjects:
                            return ' '.join(subjects[:20])  # ограничим
    except Exception:
        pass
    return ''

# === 3. Google Books: запасной источник ===
def fetch_google(isbn):
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        r = requests.get(url, timeout=5, verify=False)
        data = r.json()
        if data.get('totalItems', 0) > 0:
            info = data['items'][0]['volumeInfo']
            cats = info.get('categories', [])
            desc = info.get('description', '')
            return ' '.join(cats) + ' ' + desc[:500]  # часть описания
    except Exception:
        pass
    return ''

# === 4. Совмещаем источники ===
def fetch_book(isbn):
    # Сначала Open Library
    text = fetch_openlibrary(isbn)
    # Если пусто — пробуем Google
    if not text.strip():
        text = fetch_google(isbn)
    return isbn, text.strip()

# === 5. Скачиваем ===
to_fetch = [isbn for isbn in books['ISBN'].values if isbn not in cache]
print(f"Нужно скачать: {len(to_fetch)}")

if to_fetch:
    # Уменьшаем число потоков чтобы не словить rate-limit Google
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_book, isbn): isbn for isbn in to_fetch}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures))):
            isbn, text = f.result()
            cache[isbn] = text
            # Сохраняем кеш каждые 500 запросов на случай падения
            if (i+1) % 500 == 0:
                with open(CACHE_FILE, 'w') as fp:
                    json.dump(cache, fp)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"Сохранено в кеш: {len(cache)}")

# === 6. Добавляем в books ===
books['subjects'] = books['ISBN'].map(lambda x: cache.get(x, ''))
enriched = (books['subjects'].str.len() > 0).sum()
print(f"Обогащено: {enriched} из {len(books)} ({100*enriched/len(books):.1f}%)")

# Примеры
print("\nПримеры обогащённых данных:")
ex = books[books['subjects'].str.len() > 10][['Title','subjects']].head(3)
for _, row in ex.iterrows():
    print(f"  «{row['Title'][:40]}» → {row['subjects'][:120]}")

# === 7. Обогащённый TF-IDF ===
books['content_enriched'] = (
    books['Title'] + ' ' +
    books['Author'] + ' ' +
    books['Publisher'] + ' ' +
    books['subjects'] + ' ' + books['subjects']
)

tfidf_enriched = TfidfVectorizer(stop_words='english', max_features=20000,
                                  ngram_range=(1,2), min_df=2)
tfidf_matrix_enriched = tfidf_enriched.fit_transform(books['content_enriched'])
print(f"\nОбогащённый TF-IDF: {tfidf_matrix_enriched.shape}")

# === 8. Фабрика рекомендера ===
def make_content_rec(matrix):
    def rec(uid, top_n=10):
        ur = train[train['User-ID']==uid]
        vecs, w = [], []
        for _, r in ur.iterrows():
            if r['ISBN'] in isbn2tfidf:
                vecs.append(matrix[isbn2tfidf[r['ISBN']]])
                w.append(r['Rating'])
        if not vecs: return []
        prof = vstack(vecs).multiply(np.array(w)[:,None]).sum(axis=0)/sum(w)
        sims = cosine_similarity(np.asarray(prof), matrix).flatten()
        seen = set(ur['ISBN'])
        cands = [(book_isbns[i], sims[i]) for i in range(len(sims))
                 if book_isbns[i] not in seen]
        cands.sort(key=lambda x: -x[1])
        return [c[0] for c in cands[:top_n]]
    return rec

# === 9. Сравнение ===
print("\n" + "="*60)
print("СРАВНЕНИЕ")
print("="*60)
content_results = [
    evaluate(make_content_rec(tfidf_matrix_base),     test, name="Content base"),
    evaluate(make_content_rec(tfidf_matrix_enriched), test, name="Content + OL/Google"),
]
print(pd.DataFrame(content_results).to_string(index=False))

# === 10. Глобальные переменные ===
tfidf_matrix = tfidf_matrix_enriched
recommend_content = make_content_rec(tfidf_matrix)

if 'sample_user' in dir():
    print(f"\nОбогащённый Content рекомендует для юзера {sample_user}:")
    print(show_books(recommend_content(sample_user)))