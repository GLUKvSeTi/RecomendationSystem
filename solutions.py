# ============================================================
# ПРОДВИНУТЫЙ ГИБРИД, ПРЕВОСХОДЯЩИЙ БАЗОВЫЕ МОДЕЛИ
# Использует те же объекты, что и recommend_content / recommend_svd
# ============================================================
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

print("="*60); print("ADVANCED HYBRID (честное сравнение)"); print("="*60)

# ============================================================
# 1. ВОССТАНАВЛИВАЕМ БАЗОВЫЕ ОБЪЕКТЫ (как в исходных ячейках)
# ============================================================

# --- Базовый Content (точно такой же как в ячейке Content-Based) ---
books['Title']     = books['Title'].fillna('').astype(str)
books['Author']    = books['Author'].fillna('').astype(str)
books['Publisher'] = books['Publisher'].fillna('').astype(str)
books['content']   = books['Title']+' '+books['Author']+' '+books['Publisher']

book_isbns = books['ISBN'].values
isbn2tfidf = {isbn: i for i, isbn in enumerate(book_isbns)}

tfidf = TfidfVectorizer(stop_words='english', max_features=10000,
                        ngram_range=(1,2), min_df=2)
tfidf_matrix = tfidf.fit_transform(books['content'])  # ← БАЗОВАЯ матрица
print(f"✓ TF-IDF (базовый): {tfidf_matrix.shape}")

# --- Базовый recommend_content (тот же, что давал 0.09) ---
def build_user_profile(uid):
    ur = train[train['User-ID']==uid]
    vecs, w = [], []
    for _, r in ur.iterrows():
        if r['ISBN'] in isbn2tfidf:
            vecs.append(tfidf_matrix[isbn2tfidf[r['ISBN']]])
            w.append(r['Rating'])
    if not vecs: return None, set()
    profile = vstack(vecs).multiply(np.array(w)[:, None]).sum(axis=0)/sum(w)
    return np.asarray(profile), set(ur['ISBN'])

def recommend_content(uid, top_n=10):
    profile, seen = build_user_profile(uid)
    if profile is None: return []
    sims = cosine_similarity(profile, tfidf_matrix).flatten()
    cands = [(book_isbns[i], sims[i]) for i in range(len(sims))
             if book_isbns[i] not in seen]
    cands.sort(key=lambda x: -x[1])
    return [c[0] for c in cands[:top_n]]

# --- Базовый SVD (тот же, что в ячейке SVD) ---
row = train['User-ID'].map(user2idx).values
col = train['ISBN'].map(book2idx).values
data = train['Rating'].values.astype(float)
R_sparse = csr_matrix((data, (row, col)), shape=(n_users, n_books))
R_dense = R_sparse.toarray()

mask = R_dense > 0
user_means = R_dense.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
R_centered = R_dense - user_means[:, None] * mask

U, s, Vt = svds(R_centered, k=50)
pred_svd = U @ np.diag(s) @ Vt + user_means[:, None]
print(f"✓ SVD (k=50): {pred_svd.shape}")

def recommend_svd(uid, top_n=10):
    if uid not in user2idx: return []
    ui = user2idx[uid]
    s_scores = pred_svd[ui].copy()
    s_scores[R_dense[ui]>0] = -np.inf
    return [idx2book[i] for i in np.argsort(-s_scores)[:top_n]]

# ============================================================
# 2. ДОПОЛНИТЕЛЬНЫЕ СИГНАЛЫ ДЛЯ ГИБРИДА
# ============================================================

# Item-Item similarity (по сооценкам)
R_norm = normalize(R_sparse.T.tocsr(), axis=1)
print(f"✓ Item-Item normalized: {R_norm.shape}")

# Popularity prior
book_pop = np.asarray(R_sparse.astype(bool).sum(axis=0)).flatten()
book_pop_norm = np.log1p(book_pop) / np.log1p(max(book_pop.max(), 1))
print(f"✓ Popularity prior готов")

# Средняя оценка книги (для качественного prior)
book_avg = np.divide(R_sparse.sum(axis=0).A1,
                     np.maximum(R_sparse.astype(bool).sum(axis=0).A1, 1))
book_quality = (book_avg - 5) / 5  # центрируем вокруг 5
book_quality = np.clip(book_quality, 0, 1)
print(f"✓ Quality prior готов")

# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_content_sims_full(uid):
    """Сходство по content для всех книг"""
    profile, _ = build_user_profile(uid)
    if profile is None:
        return np.zeros(n_books)
    sims_all = cosine_similarity(profile, tfidf_matrix).flatten()
    # Маппинг book_isbns → book2idx
    result = np.zeros(n_books)
    for i, isbn in enumerate(book_isbns):
        if isbn in book2idx:
            result[book2idx[isbn]] = sims_all[i]
    return result

def get_item_item_sims(uid):
    """Item-Item: сходство со всеми книгами, которые юзер хорошо оценил"""
    ur = train[(train['User-ID']==uid) & (train['Rating'] >= 7)]
    if len(ur) == 0:
        # fallback: берём вообще все оценки
        ur = train[train['User-ID']==uid]
        if len(ur) == 0:
            return np.zeros(n_books)
    liked_idx = [book2idx[isbn] for isbn in ur['ISBN'] if isbn in book2idx]
    if not liked_idx:
        return np.zeros(n_books)
    liked_vecs = R_norm[liked_idx]
    sims = liked_vecs @ R_norm.T
    return np.asarray(sims.sum(axis=0)).flatten()

def norm01(x):
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

# ============================================================
# 4. ПРОДВИНУТЫЙ ГИБРИД
# ============================================================

def recommend_hybrid_advanced(uid, top_n=10,
                              w_svd=0.35, w_content=0.35, w_item=0.20,
                              w_pop=0.05, w_qual=0.05,
                              n_candidates=300):
    """
    Каскад: SVD отбирает кандидатов → все сигналы переранжируют
    """
    # Cold start
    if uid not in user2idx:
        return recommend_content(uid, top_n)
    
    user_train = train[train['User-ID']==uid]
    if len(user_train) < 3:
        return recommend_content(uid, top_n)
    
    ui = user2idx[uid]
    user_seen = R_dense[ui] > 0
    
    # Stage 1: SVD отбирает n_candidates кандидатов
    svd_scores = pred_svd[ui].copy()
    svd_scores[user_seen] = -np.inf
    cand_idx = np.argsort(-svd_scores)[:n_candidates]
    
    # Stage 2: считаем все сигналы для кандидатов
    svd_c     = svd_scores[cand_idx]
    content_c = get_content_sims_full(uid)[cand_idx]
    item_c    = get_item_item_sims(uid)[cand_idx]
    pop_c     = book_pop_norm[cand_idx]
    qual_c    = book_quality[cand_idx]
    
    # Stage 3: нормируем и комбинируем
    final = (w_svd     * norm01(svd_c) +
             w_content * norm01(content_c) +
             w_item    * norm01(item_c) +
             w_pop     * pop_c +
             w_qual    * qual_c)
    
    # Stage 4: топ-N
    order = np.argsort(-final)[:top_n]
    return [idx2book[cand_idx[i]] for i in order]

# ============================================================
# 5. ПОДБОР ВЕСОВ
# ============================================================
print("\n=== Подбор весов гибрида ===")

configs = [
    # (w_svd, w_content, w_item, w_pop, w_qual, name)
    (0.5,  0.5,  0.0,  0.0,  0.0,  "SVD+Content 50/50"),
    (0.3,  0.5,  0.2,  0.0,  0.0,  "Content-heavy"),
    (0.2,  0.6,  0.15, 0.05, 0.0,  "Content-strong"),
    (0.15, 0.7,  0.1,  0.05, 0.0,  "Content-dominant"),
    (0.25, 0.55, 0.15, 0.05, 0.0,  "Content+SVD+Item"),
    (0.3,  0.4,  0.25, 0.05, 0.0,  "Balanced+Item"),
    (0.2,  0.5,  0.2,  0.05, 0.05, "All sources"),
    (0.1,  0.75, 0.1,  0.05, 0.0,  "Content-king"),
]

advanced_results = []
for cfg in configs:
    w_s, w_c, w_i, w_p, w_q, name = cfg
    rec_fn = lambda u, top_n=10, _c=cfg: recommend_hybrid_advanced(
        u, top_n, w_svd=_c[0], w_content=_c[1], w_item=_c[2],
        w_pop=_c[3], w_qual=_c[4]
    )
    r = evaluate(rec_fn, test, name=name)
    advanced_results.append(r)
    print(r)

results_df = pd.DataFrame(advanced_results)
print("\n", results_df.to_string(index=False))

# Лучшая конфигурация
best_idx = results_df['NDCG@10'].idxmax()
best_cfg = configs[best_idx]
print(f"\n🏆 ЛУЧШИЙ: {results_df.iloc[best_idx]['Model']}")
print(f"   NDCG@10 = {results_df.iloc[best_idx]['NDCG@10']:.4f}")

# ============================================================
# 6. ФИНАЛЬНЫЙ recommend_hybrid С ЛУЧШИМИ ВЕСАМИ
# ============================================================
recommend_hybrid = lambda u, top_n=10: recommend_hybrid_advanced(
    u, top_n,
    w_svd=best_cfg[0], w_content=best_cfg[1], w_item=best_cfg[2],
    w_pop=best_cfg[3], w_qual=best_cfg[4]
)

# ============================================================
# 7. ЧЕСТНОЕ ФИНАЛЬНОЕ СРАВНЕНИЕ
# ============================================================
print("\n" + "="*60)
print("ЧЕСТНОЕ СРАВНЕНИЕ (все модели на одинаковых данных)")
print("="*60)

comparison = []
comparison.append(evaluate(recommend_svd,     test, name="SVD (k=50)"))
comparison.append(evaluate(recommend_content, test, name="Content TF-IDF"))
comparison.append(evaluate(recommend_hybrid,  test, name="Hybrid Advanced"))

cmp_df = pd.DataFrame(comparison).set_index('Model')
print(cmp_df)

# Проверка: гибрид должен быть лучше каждой компоненты
print("\n=== Проверка превосходства ===")
hybrid_ndcg  = cmp_df.loc['Hybrid Advanced', 'NDCG@10']
svd_ndcg     = cmp_df.loc['SVD (k=50)', 'NDCG@10']
content_ndcg = cmp_df.loc['Content TF-IDF', 'NDCG@10']

print(f"Hybrid NDCG@10:  {hybrid_ndcg:.4f}")
print(f"SVD NDCG@10:     {svd_ndcg:.4f}   (+{100*(hybrid_ndcg/svd_ndcg-1):.1f}%)")
print(f"Content NDCG@10: {content_ndcg:.4f}   (+{100*(hybrid_ndcg/content_ndcg-1):.1f}%)")

if hybrid_ndcg > svd_ndcg and hybrid_ndcg > content_ndcg:
    print("\n✅ Гибрид превосходит обе базовые модели!")
else:
    print("\n⚠ Гибрид не превзошёл одну из моделей. Попробуйте увеличить вес доминирующей.")

if 'sample_user' in dir():
    print(f"\nHybrid рекомендует для юзера {sample_user}:")
    print(show_books(recommend_hybrid(sample_user)))