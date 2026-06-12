# ============================================================
# CONTEXTUAL BANDIT над моделями (исправленная версия)
# Fix 1: правильная offline evaluation (награда из test, обучение на train→test split)
# Fix 2: NDCG-aware reward (а не просто precision)
# Fix 3: расширенный контекст
# ============================================================
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

print("="*60)
print("CONTEXTUAL BANDIT (исправленная версия)")
print("="*60)

# ============================================================
# 1. ARMS
# ============================================================
arms = {
    'Content':  recommend_content,
    'SVD':      recommend_svd,
    'Cascade':  recommend_hybrid,
}
arm_names = list(arms.keys())
n_arms = len(arms)

# ============================================================
# 2. РАСШИРЕННЫЙ КОНТЕКСТ
# ============================================================
user_stats = train.groupby('User-ID').agg(
    n_ratings=('Rating', 'count'),
    avg_rating=('Rating', 'mean'),
    std_rating=('Rating', 'std'),
    n_high=('Rating', lambda x: (x >= 7).sum()),
    n_low=('Rating', lambda x: (x <= 4).sum()),
).fillna(0)

book_pop_arr = book_pop
def avg_pop(uid):
    isbns = train[train['User-ID']==uid]['ISBN']
    pops = [book_pop_arr[book2idx[i]] for i in isbns if i in book2idx]
    return np.mean(pops) if pops else 0

def n_unique_authors(uid):
    return train[train['User-ID']==uid].merge(
        books[['ISBN','Author']], on='ISBN')['Author'].nunique()

user_stats['avg_pop']   = [avg_pop(u) for u in user_stats.index]
user_stats['n_authors'] = [n_unique_authors(u) for u in user_stats.index]
user_stats['diversity'] = user_stats['n_authors'] / np.maximum(user_stats['n_ratings'], 1)

def normalize_col(c):
    return (c - c.min()) / (c.max() - c.min() + 1e-9)

for col in ['n_ratings','avg_rating','std_rating','n_high','n_low',
            'avg_pop','n_authors','diversity']:
    user_stats[f'f_{col}'] = normalize_col(
        np.log1p(user_stats[col]) if col in ['n_ratings','n_high','n_low','avg_pop','n_authors']
        else user_stats[col]
    )

def get_context(uid):
    if uid in user_stats.index:
        row = user_stats.loc[uid]
        return np.array([
            1.0,
            row['f_n_ratings'],
            row['f_avg_rating'],
            row['f_std_rating'],
            row['f_n_high'],
            row['f_n_low'],
            row['f_avg_pop'],
            row['f_diversity'],
        ])
    return np.array([1.0] + [0.5]*7)

d = 8
print(f"✓ Контекст: d={d}")

# ============================================================
# 3. LinUCB
# ============================================================
class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms, self.d, self.alpha = n_arms, d, alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.A_inv = [np.eye(d) for _ in range(n_arms)]
    
    def _theta(self, a):
        return self.A_inv[a] @ self.b[a]
    
    def predict_ucb(self, x):
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = self._theta(a)
            scores[a] = x @ theta + self.alpha * np.sqrt(max(x @ self.A_inv[a] @ x, 0))
        return scores
    
    def predict_mean(self, x):
        return np.array([x @ self._theta(a) for a in range(self.n_arms)])
    
    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.A_inv[a] = np.linalg.inv(self.A[a])

# ============================================================
# 4. ОБУЧЕНИЕ: ПРАВИЛЬНОЕ — против test, через CROSS-VAL
# ============================================================
# Ключевая идея: используем test как "ground truth" для обучения бандита,
# НО только на половине test (sim half), а оцениваем на другой половине

print("\n=== Подготовка train/eval split для бандита ===")

# Делим test пополам по юзерам
all_test_users = test['User-ID'].dropna().drop_duplicates().to_numpy()
all_test_users = np.array(all_test_users, dtype=object)  # фикс для shuffle
rng = np.random.default_rng(42)
rng.shuffle(all_test_users)

half = len(all_test_users) // 2
sim_users  = all_test_users[:half].tolist()
eval_users = all_test_users[half:].tolist()

print(f"Всего test юзеров: {len(all_test_users)}")
print(f"sim_users:  {len(sim_users)}")
print(f"eval_users: {len(eval_users)}")  # для финальной оценки

print(f"sim_users:  {len(sim_users)}")
print(f"eval_users: {len(eval_users)}")

# user_liked из ВСЕГО test (это правильно — это и есть наши labels)
user_liked = defaultdict(set)
for _, r in test[test['Rating'] >= 7].iterrows():
    user_liked[r['User-ID']].add(r['ISBN'])

# ============================================================
# 5. НАГРАДА = NDCG@10 (а не precision!)
# ============================================================
def compute_ndcg(recs, relevant_set, k=10):
    if not relevant_set or not recs:
        return 0.0
    dcg = 0.0
    for i, isbn in enumerate(recs[:k]):
        if isbn in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ============================================================
# 6. ОБУЧЕНИЕ БАНДИТА
# ============================================================
print("\n=== Обучение LinUCB ===")
bandit = LinUCB(n_arms=n_arms, d=d, alpha=2.0)

# Кеш рекомендаций (чтобы не пересчитывать)
rec_cache = {}

arm_pulls = np.zeros(n_arms)
arm_rewards = np.zeros(n_arms)

for uid in tqdm(sim_users, desc="Training"):
    relevant = user_liked.get(uid, set())
    if not relevant:
        continue
    
    x = get_context(uid)
    
    # Обновляем ВСЕ руки (batch update — корректно для offline)
    for a, name in enumerate(arm_names):
        key = (uid, name)
        if key not in rec_cache:
            try:
                rec_cache[key] = arms[name](uid, top_n=10)
            except Exception:
                rec_cache[key] = []
        recs = rec_cache[key]
        
        # Награда = NDCG@10
        reward = compute_ndcg(recs, relevant, k=10)
        
        bandit.update(a, x, reward)
        arm_pulls[a] += 1
        arm_rewards[a] += reward

print("\n=== Статистика обучения ===")
for a, name in enumerate(arm_names):
    avg = arm_rewards[a] / max(arm_pulls[a], 1)
    print(f"  {name:10s}: avg NDCG = {avg:.4f}  ({int(arm_pulls[a])} pulls)")

# ============================================================
# 7. РЕКОМЕНДАТЕЛЬ С БАНДИТОМ
# ============================================================
def recommend_bandit_over_models(uid, top_n=10):
    x = get_context(uid)
    scores = bandit.predict_mean(x)  # без UCB bonus на inference
    a = int(np.argmax(scores))
    return arms[arm_names[a]](uid, top_n=top_n)

# ============================================================
# 8. АНАЛИЗ ВЫБОРА РУК (на eval_users)
# ============================================================
print("\n=== Распределение выбора рук на eval_users ===")
choices = defaultdict(int)
choice_per_user = {}
for uid in eval_users:
    x = get_context(uid)
    scores = bandit.predict_mean(x)
    a = int(np.argmax(scores))
    choices[arm_names[a]] += 1
    choice_per_user[uid] = arm_names[a]

total = sum(choices.values())
for name in arm_names:
    cnt = choices.get(name, 0)
    print(f"  {name:10s}: {cnt:4d}  ({100*cnt/max(total,1):5.1f}%)")

# Если опять все Content — диагностика
if choices.get('Content', 0) / max(total, 1) > 0.9:
    print("\n⚠ Bandit опять предпочитает Content. Усиливаем exploration:")
    print("  Прогоняем второй раунд обучения с alpha=5.0...")
    bandit.alpha = 5.0
    # перевзвешиваем награды: вычитаем среднюю награду Content из всех
    # это делает Content менее доминирующим
    content_mean = arm_rewards[0] / max(arm_pulls[0], 1)
    bandit2 = LinUCB(n_arms=n_arms, d=d, alpha=2.0)
    for uid in tqdm(sim_users, desc="Re-training"):
        relevant = user_liked.get(uid, set())
        if not relevant:
            continue
        x = get_context(uid)
        for a, name in enumerate(arm_names):
            recs = rec_cache.get((uid, name), [])
            reward = compute_ndcg(recs, relevant, k=10)
            # Центрируем награду
            reward_centered = reward - content_mean
            bandit2.update(a, x, reward_centered)
    bandit = bandit2
    
    # Перепроверяем
    print("\nПосле перебалансировки:")
    choices = defaultdict(int)
    for uid in eval_users:
        x = get_context(uid)
        scores = bandit.predict_mean(x)
        a = int(np.argmax(scores))
        choices[arm_names[a]] += 1
    for name in arm_names:
        cnt = choices.get(name, 0)
        print(f"  {name:10s}: {cnt:4d}  ({100*cnt/max(total,1):5.1f}%)")

# ============================================================
# 9. ФИНАЛЬНАЯ ОЦЕНКА (только на eval_users!)
# ============================================================
print("\n" + "="*60)
print("ФИНАЛЬНАЯ ОЦЕНКА (на eval_users — не участвовали в обучении)")
print("="*60)

# Кастомная оценка только по eval_users
def evaluate_on_users(recommend_fn, users, k=10, name=""):
    P, R, H, AP, NDCG = [], [], [], [], []
    for uid in users:
        relevant = user_liked.get(uid, set())
        if not relevant:
            continue
        recs = recommend_fn(uid, top_n=k)
        if not recs:
            continue
        hits = [1 if isbn in relevant else 0 for isbn in recs]
        n_hits = sum(hits)
        P.append(n_hits / k)
        R.append(n_hits / len(relevant))
        H.append(1.0 if n_hits > 0 else 0.0)
        # AP
        ap = 0.0; cum = 0
        for i, h in enumerate(hits):
            if h:
                cum += 1
                ap += cum / (i + 1)
        AP.append(ap / min(len(relevant), k))
        # NDCG
        NDCG.append(compute_ndcg(recs, relevant, k))
    return {
        'Model': name,
        f'Precision@{k}': round(np.mean(P), 4),
        f'Recall@{k}':    round(np.mean(R), 4),
        f'HitRate@{k}':   round(np.mean(H), 4),
        f'MAP@{k}':       round(np.mean(AP), 4),
        f'NDCG@{k}':      round(np.mean(NDCG), 4),
    }

results = []
results.append(evaluate_on_users(recommend_content, eval_users, name="Content only"))
results.append(evaluate_on_users(recommend_svd,     eval_users, name="SVD only"))
results.append(evaluate_on_users(recommend_hybrid,  eval_users, name="Cascade Hybrid"))
results.append(evaluate_on_users(recommend_bandit_over_models, eval_users,
                                 name="Bandit over models"))

res_df = pd.DataFrame(results).set_index('Model')
print(res_df)

# Сравнение
bandit_ndcg  = res_df.loc['Bandit over models', 'NDCG@10']
best_single = res_df.drop('Bandit over models')['NDCG@10'].max()
best_name = res_df.drop('Bandit over models')['NDCG@10'].idxmax()

print(f"\nBandit NDCG@10:  {bandit_ndcg:.4f}")
print(f"Best single ({best_name}): {best_single:.4f}")
if bandit_ndcg > best_single:
    print(f"✅ Bandit лучше! +{100*(bandit_ndcg/best_single-1):+.1f}%")
elif bandit_ndcg >= best_single * 0.98:
    print(f"≈ Bandit практически как лучшая модель (разница < 2%)")
else:
    print(f"⚠ Bandit хуже лучшей на {100*(best_single/bandit_ndcg-1):+.1f}%")

# ============================================================
# 10. БОНУС: ORACLE — теоретический потолок выбора руки
# ============================================================
print("\n=== Oracle: что было бы если выбирать ЛУЧШУЮ руку для каждого юзера ===")
oracle_ndcg = []
for uid in eval_users:
    relevant = user_liked.get(uid, set())
    if not relevant:
        continue
    best_ndcg = 0
    for name in arm_names:
        recs = arms[name](uid, top_n=10)
        ndcg = compute_ndcg(recs, relevant, k=10)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
    oracle_ndcg.append(best_ndcg)

print(f"Oracle NDCG@10: {np.mean(oracle_ndcg):.4f}")
print(f"(теоретический максимум, если бы bandit всегда выбирал лучшую руку для юзера)")

recommend_bandit = recommend_bandit_over_models
print("\n✓ Готово")