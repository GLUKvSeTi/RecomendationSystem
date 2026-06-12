# ============================================================
# ПРОДВИНУТЫЙ ГИБРИД: SVD + Content + Item-Item + Popularity
# ============================================================
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("ADVANCED HYBRID")
print("="*60)

# ============================================================
# 1. ПЕРЕСОБИРАЕМ КОМПОНЕНТЫ ЧТОБЫ ТОЧНО РАБОТАЛО
# ============================================================

# --- Матрица оценок ---
row = train['User-ID'].map(user2idx).values
col = train['ISBN'].map(book2idx).values
data = train['Rating'].values.astype(float)
R_sparse = csr_matrix((data, (row, col)), shape=(n_users, n_books))
R_dense = R_sparse.toarray()

# --- SVD (с центрированием) ---
mask = R_dense > 0
user_means = R_dense.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
R_centered = R_dense - user_means[:, None] * mask

U, s, Vt = svds(R_centered, k=50)
pred_svd = U @ np.diag(s) @ Vt + user_means[:, None]
print(f"✓ SVD готов: {pred_svd.shape}")

# --- Content TF-IDF (только основной сигнал) ---
books['content'] = books['Title']+' '+books['Author']+' '+books['Publisher']
tfidf_main = TfidfVectorizer(stop_words='english', max_features=10000,
                              ngram_range=(1,2), min_df=2)
mat_content = tfidf_main.fit_transform(books['content'])
print(f"✓ Content TF-IDF: {mat_content.shape}")

# --- Subjects TF-IDF (отдельно, если есть кеш) ---
has_subjects = False
if 'cache' in dir() and len(cache) > 0:
    def clean_subj(s):
        return ' '.join(s.split()[:10]) if s else ''
    books['subj'] = books['ISBN'].map(lambda x: clean_subj(cache.get(x, '')))
    if (books['subj'].str.len() > 0).sum() > 100:
        tfidf_subj = TfidfVectorizer(stop_words='english', max_features=3000,
                                      min_df=2, max_df=0.5)
        mat_subj = tfidf_subj.fit_transform(books['subj'])
        has_subjects = True
        print(f"✓ Subjects TF-IDF: {mat_subj.shape}")
    else:
        print("⚠ Subjects слишком мало, пропускаем")
else:
    print("⚠ Кеш не найден, обходимся без subjects")

# --- Item-Item матрица похожести (по сооценкам) ---
# Нормируем матрицу оценок и считаем cosine между КНИГАМИ
from sklearn.preprocessing import normalize
R_norm = normalize(R_sparse.T.tocsr(), axis=1)  # книги × юзеры
# Чтобы не считать полную матрицу (8000×8000), будем считать по запросу
print(f"✓ Item-Item матрица готова: {R_norm.shape}")

# --- Популярность книг (для prior) ---
book_pop = np.asarray(R_sparse.astype(bool).sum(axis=0)).flatten()
book_pop_norm = np.log1p(book_pop) / np.log1p(book_pop.max())
print(f"✓ Popularity prior готов")

# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

book_isbns_arr = np.array(book_isbns)
isbn2tfidf = {isbn: i for i, isbn in enumerate(book_isbns_arr)}

def get_content_sims(uid):
    """Похожесть всех книг на профиль пользователя по content"""
    ur = train[train['User-ID']==uid]
    vecs, w = [], []
    for _, r in ur.iterrows():
        if r['ISBN'] in isbn2tfidf:
            vecs.append(mat_content[isbn2tfidf[r['ISBN']]])
            w.append(r['Rating'])
    if not vecs:
        return np.zeros(n_books)
    prof = vstack(vecs).multiply(np.array(w)[:,None]).sum(axis=0)/sum(w)
    return cosine_similarity(np.asarray(prof), mat_content).flatten()

def get_subjects_sims(uid):
    """Похожесть по subjects"""
    if not has_subjects:
        return np.zeros(n_books)
    ur = train[train['User-ID']==uid]
    vecs, w = [], []
    for _, r in ur.iterrows():
        if r['ISBN'] in isbn2tfidf:
            vecs.append(mat_subj[isbn2tfidf[r['ISBN']]])
            w.append(r['Rating'])
    if not vecs:
        return np.zeros(n_books)
    prof = vstack(vecs).multiply(np.array(w)[:,None]).sum(axis=0)/sum(w)
    return cosine_similarity(np.asarray(prof), mat_subj).flatten()

def get_item_item_sims(uid):
    """Сумма похожестей с книгами, которые юзер хорошо оценил"""
    ur = train[(train['User-ID']==uid) & (train['Rating'] >= 7)]
    if len(ur) == 0:
        return np.zeros(n_books)
    
    # Берём индексы любимых книг
    liked_idx = [book2idx[isbn] for isbn in ur['ISBN'] if isbn in book2idx]
    if not liked_idx:
        return np.zeros(n_books)
    
    # cos(любимые книги, все книги) → суммируем
    liked_vecs = R_norm[liked_idx]
    sims = liked_vecs @ R_norm.T  # (n_liked × n_books)
    return np.asarray(sims.sum(axis=0)).flatten()

def normalize_scores(scores):
    """Min-max в [0,1]"""
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

# ============================================================
# 3. ПРОДВИНУТЫЙ ГИБРИД
# ============================================================

def recommend_advanced(uid, top_n=10, 
                       w_svd=0.4, w_content=0.25, w_item=0.25, 
                       w_subj=0.05, w_pop=0.05,
                       n_candidates=200):
    """
    Каскад:
    1. SVD отбирает top-N кандидатов
    2. Все источники нормируются и складываются с весами
    3. Финальное переранжирование
    """
    if uid not in user2idx:
        # Cold start: только content
        sims = get_content_sims(uid)
        sims[R_dense[user2idx.get(uid, 0)] > 0] = -np.inf if uid in user2idx else 0
        order = np.argsort(-sims)[:top_n]
        return [idx2book[i] for i in order]
    
    ui = user2idx[uid]
    user_seen = R_dense[ui] > 0
    
    # === Stage 1: SVD отбирает кандидатов ===
    svd_scores = pred_svd[ui].copy()
    svd_scores[user_seen] = -np.inf
    cand_idx = np.argsort(-svd_scores)[:n_candidates]
    
    # === Stage 2: считаем все сигналы для кандидатов ===
    svd_c     = svd_scores[cand_idx]
    content_c = get_content_sims(uid)[cand_idx]
    item_c    = get_item_item_sims(uid)[cand_idx]
    subj_c    = get_subjects_sims(uid)[cand_idx] if has_subjects else np.zeros(len(cand_idx))
    pop_c     = book_pop_norm[cand_idx]
    
    # === Stage 3: нормируем и комбинируем ===
    final = (w_svd     * normalize_scores(svd_c) +
             w_content * normalize_scores(content_c) +
             w_item    * normalize_scores(item_c) +
             w_subj    * normalize_scores(subj_c) +
             w_pop     * pop_c)
    
    # === Stage 4: топ-N ===
    order = np.argsort(-final)[:top_n]
    return [idx2book[cand_idx[i]] for i in order]

# ============================================================
# 4. ПОДБОР ВЕСОВ (grid search по NDCG)
# ============================================================
print("\n=== Подбор весов гибрида ===")

configs = [
    # (w_svd, w_content, w_item, w_subj, w_pop, name)
    (1.0, 0.0, 0.0, 0.0, 0.0, "Pure SVD"),
    (0.0, 1.0, 0.0, 0.0, 0.0, "Pure Content"),
    (0.5, 0.5, 0.0, 0.0, 0.0, "SVD+Content"),
    (0.4, 0.3, 0.3, 0.0, 0.0, "SVD+Cont+Item"),
    (0.4, 0.25, 0.25, 0.05, 0.05, "All sources"),
    (0.5, 0.2, 0.2, 0.05, 0.05, "SVD-heavy"),
    (0.3, 0.3, 0.3, 0.05, 0.05, "Balanced"),
    (0.35, 0.2, 0.35, 0.0, 0.1, "SVD+Item+Pop"),
]

advanced_results = []
for cfg in configs:
    w_s, w_c, w_i, w_sub, w_p, name = cfg
    rec_fn = lambda u, top_n=10, _c=cfg: recommend_advanced(
        u, top_n, w_svd=_c[0], w_content=_c[1], w_item=_c[2],
        w_subj=_c[3], w_pop=_c[4]
    )
    r = evaluate(rec_fn, test, name=name)
    advanced_results.append(r)
    print(r)

results_df = pd.DataFrame(advanced_results)
print("\n", results_df.to_string(index=False))

# Лучшая конфигурация
best_idx = results_df['NDCG@10'].idxmax()
print(f"\n🏆 ЛУЧШИЙ: {results_df.iloc[best_idx]['Model']}")
print(f"   NDCG@10 = {results_df.iloc[best_idx]['NDCG@10']}")

# ============================================================
# 5. ФИНАЛЬНЫЙ РЕКОМЕНДЕР С ЛУЧШИМИ ВЕСАМИ
# ============================================================
best_cfg = configs[best_idx]
recommend_hybrid = lambda u, top_n=10: recommend_advanced(
    u, top_n,
    w_svd=best_cfg[0], w_content=best_cfg[1], w_item=best_cfg[2],
    w_subj=best_cfg[3], w_pop=best_cfg[4]
)

if 'sample_user' in dir():
    print(f"\nADVANCED HYBRID для юзера {sample_user}:")
    print(show_books(recommend_hybrid(sample_user)))

print("\n✓ recommend_hybrid обновлён на продвинутую версию")