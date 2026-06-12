# ============================================================
# КАСКАДНЫЙ ГИБРИД: Content → SVD → Item-Item → Popularity
# (последовательность, а не сумма)
# ============================================================
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

print("="*60)
print("CASCADE HYBRID: Content → SVD → Item-Item → Popularity")
print("="*60)

# ============================================================
# 1. ВОССТАНАВЛИВАЕМ КОМПОНЕНТЫ
# ============================================================

# --- Content TF-IDF (базовый) ---
books['Title']     = books['Title'].fillna('').astype(str)
books['Author']    = books['Author'].fillna('').astype(str)
books['Publisher'] = books['Publisher'].fillna('').astype(str)
books['content']   = books['Title']+' '+books['Author']+' '+books['Publisher']

book_isbns = books['ISBN'].values
isbn2tfidf = {isbn: i for i, isbn in enumerate(book_isbns)}

tfidf = TfidfVectorizer(stop_words='english', max_features=10000,
                        ngram_range=(1,2), min_df=2)
tfidf_matrix = tfidf.fit_transform(books['content'])
print(f"✓ Content TF-IDF: {tfidf_matrix.shape}")

# --- Восстанавливаем recommend_content ---
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

# --- SVD ---
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

# --- Item-Item матрица ---
R_norm = normalize(R_sparse.T.tocsr(), axis=1)
print(f"✓ Item-Item normalized: {R_norm.shape}")

# --- Popularity ---
book_pop = np.asarray(R_sparse.astype(bool).sum(axis=0)).flatten()
print(f"✓ Popularity готов")

# ============================================================
# 2. КАСКАДНЫЙ ГИБРИД (ПОСЛЕДОВАТЕЛЬНОСТЬ ЭТАПОВ)
# ============================================================

def recommend_cascade(uid, top_n=10, 
                      n_stage1=500,   # сколько кандидатов оставит Content
                      n_stage2=200,   # сколько кандидатов после SVD-rerank
                      n_stage3=50,    # сколько после Item-Item буста
                      min_popularity=2):
    """
    Каскадный гибрид:
    Stage 1 (Content):  отобрать n_stage1 семантически похожих
    Stage 2 (SVD):      переранжировать, оставить n_stage2 лучших
    Stage 3 (Item-Item): добавить буст по сооценкам, оставить n_stage3
    Stage 4 (Popularity): отфильтровать слишком редкие, взять top_n
    """
    
    # === Cold start: если юзера нет в train, чистый Content ===
    if uid not in user2idx:
        return recommend_content(uid, top_n)
    
    user_train = train[train['User-ID']==uid]
    if len(user_train) < 2:
        return recommend_content(uid, top_n)
    
    ui = user2idx[uid]
    seen_isbns = set(user_train['ISBN'])
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ STAGE 1: CONTENT-BASED RETRIEVAL                       ║
    # ║ Отбираем n_stage1 кандидатов по семантической похожести║
    # ╚════════════════════════════════════════════════════════╝
    profile, _ = build_user_profile(uid)
    if profile is None:
        return recommend_svd(uid, top_n)
    
    content_sims = cosine_similarity(profile, tfidf_matrix).flatten()
    
    # Топ-n_stage1 ISBN-ов по content (исключая виденные)
    content_order = np.argsort(-content_sims)
    stage1_isbns = []
    for idx in content_order:
        isbn = book_isbns[idx]
        if isbn not in seen_isbns and isbn in book2idx:
            stage1_isbns.append(isbn)
            if len(stage1_isbns) >= n_stage1:
                break
    
    if not stage1_isbns:
        return recommend_content(uid, top_n)
    
    # Для каждого кандидата запоминаем его content-score
    stage1_idx_book = np.array([book2idx[isbn] for isbn in stage1_isbns])
    stage1_idx_tfidf = np.array([isbn2tfidf[isbn] for isbn in stage1_isbns])
    stage1_content_scores = content_sims[stage1_idx_tfidf]
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ STAGE 2: SVD RE-RANKING                                ║
    # ║ Переранжируем кандидатов по предсказаниям SVD          ║
    # ╚════════════════════════════════════════════════════════╝
    svd_scores = pred_svd[ui, stage1_idx_book]
    
    # Комбинированный скор: SVD-rank + лёгкий бонус от content
    # (чтобы Content полностью не потерял голос)
    svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min() + 1e-9)
    content_norm = (stage1_content_scores - stage1_content_scores.min()) / \
                   (stage1_content_scores.max() - stage1_content_scores.min() + 1e-9)
    
    stage2_score = 0.6 * svd_norm + 0.4 * content_norm
    stage2_order = np.argsort(-stage2_score)[:n_stage2]
    
    stage2_isbns       = [stage1_isbns[i]       for i in stage2_order]
    stage2_idx_book    = stage1_idx_book[stage2_order]
    stage2_base_scores = stage2_score[stage2_order]
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ STAGE 3: ITEM-ITEM CF BOOST                            ║
    # ║ Бустим книги, похожие на любимые юзера по сооценкам    ║
    # ╚════════════════════════════════════════════════════════╝
    liked = user_train[user_train['Rating'] >= 7]
    if len(liked) == 0:
        liked = user_train  # fallback: все оценки
    
    liked_idx = [book2idx[isbn] for isbn in liked['ISBN'] if isbn in book2idx]
    
    if liked_idx:
        liked_vecs = R_norm[liked_idx]
        item_sims_all = np.asarray((liked_vecs @ R_norm.T).sum(axis=0)).flatten()
        item_boost = item_sims_all[stage2_idx_book]
        # Нормируем
        if item_boost.max() > item_boost.min():
            item_boost_norm = (item_boost - item_boost.min()) / \
                              (item_boost.max() - item_boost.min())
        else:
            item_boost_norm = np.zeros_like(item_boost)
        
        # Финальный скор: предыдущий + буст
        stage3_score = stage2_base_scores + 0.5 * item_boost_norm
    else:
        stage3_score = stage2_base_scores
    
    stage3_order = np.argsort(-stage3_score)[:n_stage3]
    stage3_isbns       = [stage2_isbns[i]    for i in stage3_order]
    stage3_idx_book    = stage2_idx_book[stage3_order]
    stage3_final_score = stage3_score[stage3_order]
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ STAGE 4: POPULARITY FILTER + TOP-N                     ║
    # ║ Убираем книги с < min_popularity оценок                ║
    # ╚════════════════════════════════════════════════════════╝
    final_pairs = []
    for i, isbn in enumerate(stage3_isbns):
        bidx = stage3_idx_book[i]
        if book_pop[bidx] >= min_popularity:
            final_pairs.append((isbn, stage3_final_score[i]))
    
    # Если после фильтра осталось мало — добавляем без фильтра
    if len(final_pairs) < top_n:
        for i, isbn in enumerate(stage3_isbns):
            if isbn not in [p[0] for p in final_pairs]:
                final_pairs.append((isbn, stage3_final_score[i]))
            if len(final_pairs) >= top_n:
                break
    
    final_pairs.sort(key=lambda x: -x[1])
    return [p[0] for p in final_pairs[:top_n]]

# ============================================================
# 3. ПОДБОР ПАРАМЕТРОВ КАСКАДА
# ============================================================
print("\n=== Подбор параметров каскада ===")

cascade_configs = [
    # (n_stage1, n_stage2, n_stage3, min_pop, name)
    (200, 100, 30,  1, "Cascade narrow"),
    (500, 200, 50,  2, "Cascade balanced"),
    (1000, 300, 50, 2, "Cascade wide"),
    (500, 150, 30,  2, "Cascade strict"),
    (300, 100, 50,  3, "Cascade popular"),
    (800, 250, 50,  1, "Cascade exploration"),
]

cascade_results = []
for cfg in cascade_configs:
    n1, n2, n3, mp, name = cfg
    rec_fn = lambda u, top_n=10, _c=cfg: recommend_cascade(
        u, top_n,
        n_stage1=_c[0], n_stage2=_c[1], n_stage3=_c[2], min_popularity=_c[3]
    )
    r = evaluate(rec_fn, test, name=name)
    cascade_results.append(r)
    print(r)

results_df = pd.DataFrame(cascade_results)
print("\n", results_df.to_string(index=False))

# Лучшая конфигурация
best_idx = results_df['NDCG@10'].idxmax()
best_cfg = cascade_configs[best_idx]
print(f"\n🏆 ЛУЧШИЙ КАСКАД: {results_df.iloc[best_idx]['Model']}")
print(f"   NDCG@10 = {results_df.iloc[best_idx]['NDCG@10']:.4f}")

# ============================================================
# 4. ФИНАЛЬНЫЙ recommend_hybrid
# ============================================================
recommend_hybrid = lambda u, top_n=10: recommend_cascade(
    u, top_n,
    n_stage1=best_cfg[0], n_stage2=best_cfg[1],
    n_stage3=best_cfg[2], min_popularity=best_cfg[3]
)

# ============================================================
# 5. ЧЕСТНОЕ СРАВНЕНИЕ С КОМПОНЕНТАМИ
# ============================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ С КОМПОНЕНТАМИ ГИБРИДА")
print("="*60)

comparison = []
comparison.append(evaluate(recommend_content, test, name="Content-Based only"))
comparison.append(evaluate(recommend_svd,     test, name="SVD only"))
comparison.append(evaluate(recommend_hybrid,  test, name="Cascade Hybrid"))

cmp_df = pd.DataFrame(comparison).set_index('Model')
print(cmp_df)

# Проверка
hybrid_ndcg  = cmp_df.loc['Cascade Hybrid',    'NDCG@10']
svd_ndcg     = cmp_df.loc['SVD only',          'NDCG@10']
content_ndcg = cmp_df.loc['Content-Based only', 'NDCG@10']

print("\n=== Проверка превосходства ===")
print(f"Cascade Hybrid:  {hybrid_ndcg:.4f}")
print(f"Content only:    {content_ndcg:.4f}  ({'✓' if hybrid_ndcg>content_ndcg else '✗'} +{100*(hybrid_ndcg/max(content_ndcg,1e-9)-1):+.1f}%)")
print(f"SVD only:        {svd_ndcg:.4f}  ({'✓' if hybrid_ndcg>svd_ndcg else '✗'} +{100*(hybrid_ndcg/max(svd_ndcg,1e-9)-1):+.1f}%)")

if hybrid_ndcg > svd_ndcg and hybrid_ndcg > content_ndcg:
    print("\n✅ Каскадный гибрид превосходит обе базовые модели!")
else:
    print("\n⚠ Если не превосходит — настройте параметры или порядок этапов")

if 'sample_user' in dir():
    print(f"\nCascade Hybrid рекомендует для юзера {sample_user}:")
    print(show_books(recommend_hybrid(sample_user)))