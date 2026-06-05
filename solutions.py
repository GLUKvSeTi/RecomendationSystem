# ============================================================
# ОБОГАЩЕНИЕ ДАННЫХ ЧЕРЕЗ OPEN LIBRARY API
# ============================================================
import requests
import json
import os
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CACHE_FILE = 'openlibrary_cache.json'

# 1. Загружаем кеш
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Загружено из кеша: {len(cache)}")
else:
    cache = {}

# 2. Функция запроса
def fetch_book(isbn):
    try:
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        r = requests.get(url, timeout=5, verify=False)
        info = r.json().get(f"ISBN:{isbn}", {})
        subjects = ' '.join([s['name'] for s in info.get('subjects', [])])
        return isbn, subjects
    except:
        return isbn, ''

# 3. Скачиваем то, чего нет в кеше
to_fetch = [isbn for isbn in books['ISBN'].values if isbn not in cache]
print(f"Нужно скачать: {len(to_fetch)}")

if to_fetch:
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_book, isbn): isbn for isbn in to_fetch}
        for f in tqdm(as_completed(futures), total=len(futures)):
            isbn, subjects = f.result()
            cache[isbn] = subjects
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"Сохранено в кеш: {len(cache)}")

# 4. Добавляем subjects в books
books['subjects'] = books['ISBN'].map(lambda x: cache.get(x, ''))
enriched = (books['subjects'].str.len() > 0).sum()
print(f"Обогащено: {enriched} из {len(books)} ({100*enriched/len(books):.1f}%)")

# 5. Новый content + новый TF-IDF
books['content_enriched'] = (
    books['Title'] + ' ' +
    books['Author'] + ' ' +
    books['Publisher'] + ' ' +
    books['subjects'] + ' ' + books['subjects']  # двойной вес тем
)

tfidf_v2 = TfidfVectorizer(stop_words='english', max_features=15000,
                            ngram_range=(1,2), min_df=2)
tfidf_matrix_v2 = tfidf_v2.fit_transform(books['content_enriched'])
print("Обогащённый TF-IDF shape:", tfidf_matrix_v2.shape)

# 6. Фабрика рекомендера
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

# 7. Сравнение
print("\n=== Сравнение Content до/после обогащения ===")
content_results = []
content_results.append(evaluate(make_content_rec(tfidf_matrix),    test, name="Content base"))
content_results.append(evaluate(make_content_rec(tfidf_matrix_v2), test, name="Content + OpenLibrary"))
print(pd.DataFrame(content_results))

# 8. Переопределяем основной рекомендер на обогащённой версии
tfidf_matrix = tfidf_matrix_v2
recommend_content = make_content_rec(tfidf_matrix)

print("\nОбогащённый Content рекомендует:")
print(show_books(recommend_content(sample_user)))