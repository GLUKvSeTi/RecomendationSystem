# ============================================================
# CONTEXTUAL BANDIT над моделями (LinUCB)
# Arms = {Content, SVD, Cascade Hybrid}
# Bandit учится выбирать лучшую модель для каждого юзера
# ============================================================
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

print("="*60)
print("CONTEXTUAL BANDIT OVER MODELS (LinUCB)")
print("="*60)

# ============================================================
# 1. ARMS = готовые рекомендеры
# ============================================================
arms = {
    'Content':  recommend_content,
    'SVD':      recommend_svd,
    'Cascade':  recommend_hybrid,   # каскадный гибрид
}
arm_names = list(arms.keys())
n_arms = len(arms)
print(f"✓ Arms: {arm_names}")

# ============================================================
# 2. КОНТЕКСТ ЮЗЕРА (фичи, по которым bandit решает)
# ============================================================
# Идея: для разных юзеров разные модели работают по-разному
# - Юзер с малым числом оценок → Content скорее всего лучше
# - Юзер с большой историей → SVD/Cascade
# - Юзер любит мейнстрим → SVD
# - Юзер с уникальными вкусами → Content

# Считаем фичи юзеров на train
user_stats = train.groupby('User-ID').agg(
    n_ratings=('Rating', 'count'),
    avg_rating=('Rating', 'mean'),
    std_rating=('Rating', 'std')
).fillna(0)

# Средняя популярность книг, которые юзер читал
book_pop_series = pd.Series(book_pop, index=[idx2book[i] for i in range(n_books)])
def avg_pop(uid):
    isbns = train[train['User-ID']==uid]['ISBN']
    pops = [book_pop_series.get(i, 0) for i in isbns]
    return np.mean(pops) if pops else 0

user_stats['avg_pop'] = [avg_pop(u) for u in user_stats.index]

# Нормируем фичи в [0, 1]
def normalize_col(c):
    return (c - c.min()) / (c.max() - c.min() + 1e-9)

user_stats['f_n_ratings']  = normalize_col(np.log1p(user_stats['n_ratings']))
user_stats['f_avg_rating'] = normalize_col(user_stats['avg_rating'])
user_stats['f_std_rating'] = normalize_col(user_stats['std_rating'])
user_stats['f_avg_pop']    = normalize_col(np.log1p(user_stats['avg_pop']))

def get_context(uid):
    """Вектор фичей юзера + bias"""
    if uid in user_stats.index:
        row = user_stats.loc[uid]
        return np.array([
            1.0,                    # bias
            row['f_n_ratings'],     # сколько оценок
            row['f_avg_rating'],    # средняя оценка
            row['f_std_rating'],    # разброс оценок
            row['f_avg_pop'],       # любит ли мейнстрим
        ])
    return np.array([1.0, 0.0, 0.5, 0.5, 0.5])

d = 5  # размерность контекста
print(f"✓ Размерность контекста: d={d}")

# ============================================================
# 3. LinUCB
# ============================================================
class LinUCB:
    """
    LinUCB: контекстный бандит с линейными моделями
    Для каждой руки a хранит:
      A_a = I + sum(x_t x_t^T)  — матрица контекстов
      b_a = sum(r_t * x_t)       — вектор наград
    Скор: x^T θ + α * sqrt(x^T A^{-1} x)
    где θ = A^{-1} b
    """
    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
    
    def predict(self, x):
        """Возвращает UCB-скоры для всех рук"""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = x @ theta + self.alpha * np.sqrt(x @ A_inv @ x)
        return scores
    
    def select(self, x):
        """Выбираем руку с максимальным UCB"""
        return int(np.argmax(self.predict(x)))
    
    def update(self, a, x, r):
        """Обновление параметров руки a"""
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

# ============================================================
# 4. ОБУЧЕНИЕ БАНДИТА (offline simulation)
# ============================================================
# Симулируем: для каждого юзера в train берём контекст,
# показываем рекомендации каждой руки, считаем награду
# по тому, насколько рекомендации релевантны (по test_train split)

# Делаем мини-валидацию: отделяем 20% train как симуляционный test
np.random.seed(42)
sim_users = train['User-ID'].drop_duplicates().sample(
    n=min(500, train['User-ID'].nunique()), random_state=42
).values

print(f"\n=== Обучение LinUCB на {len(sim_users)} юзерах ===")

bandit = LinUCB(n_arms=n_arms, d=d, alpha=1.5)

# Готовим релевантности из ВСЕГО train (для оценки наград)
# Награда = была ли рекомендация в "понравившихся" юзеру (rating >= 7)
user_liked = defaultdict(set)
for _, r in train[train['Rating'] >= 7].iterrows():
    user_liked[r['User-ID']].add(r['ISBN'])

# Симуляция: для каждого юзера выбираем руку и считаем награду
arm_pulls = np.zeros(n_arms)
arm_rewards = np.zeros(n_arms)

for uid in tqdm(sim_users, desc="Training bandit"):
    if uid not in user_liked or len(user_liked[uid]) == 0:
        continue
    
    x = get_context(uid)
    
    # Для обучения: пробуем ВСЕ руки, обновляем веса
    # (offline batch обучение — корректнее для оценки)
    for a, name in enumerate(arm_names):
        try:
            recs = arms[name](uid, top_n=10)
            # Награда: доля релевантных рекомендаций
            if recs:
                hits = sum(1 for isbn in recs if isbn in user_liked[uid])
                reward = hits / len(recs)
            else:
                reward = 0.0
            
            bandit.update(a, x, reward)
            arm_pulls[a] += 1
            arm_rewards[a] += reward
        except Exception:
            continue

print("\n=== Статистика рук ===")
for a, name in enumerate(arm_names):
    avg = arm_rewards[a] / max(arm_pulls[a], 1)
    print(f"  {name:10s}: avg reward = {avg:.4f}  ({int(arm_pulls[a])} pulls)")

# ============================================================
# 5. РЕКОМЕНДАТЕЛЬ С БАНДИТОМ
# ============================================================

def recommend_bandit_over_models(uid, top_n=10, explore=False):
    """
    Bandit выбирает лучшую модель для конкретного юзера
    на основе его контекстных фичей
    """
    x = get_context(uid)
    
    if explore:
        # exploration: иногда случайная рука
        if np.random.random() < 0.1:
            a = np.random.randint(n_arms)
        else:
            a = bandit.select(x)
    else:
        # exploitation: берём лучшую руку без exploration bonus
        scores = np.zeros(n_arms)
        for arm_i in range(n_arms):
            A_inv = np.linalg.inv(bandit.A[arm_i])
            theta = A_inv @ bandit.b[arm_i]
            scores[arm_i] = x @ theta  # без UCB bonus
        a = int(np.argmax(scores))
    
    chosen_arm = arm_names[a]
    return arms[chosen_arm](uid, top_n=top_n)

# ============================================================
# 6. АНАЛИЗ: КАКУЮ РУКУ БАНДИТ ВЫБИРАЕТ ДЛЯ КОГО
# ============================================================
print("\n=== Распределение выбора рук на test ===")

test_users_sample = test['User-ID'].drop_duplicates().sample(
    n=min(300, test['User-ID'].nunique()), random_state=42
).values

choices = defaultdict(int)
for uid in test_users_sample:
    x = get_context(uid)
    scores = np.zeros(n_arms)
    for arm_i in range(n_arms):
        A_inv = np.linalg.inv(bandit.A[arm_i])
        theta = A_inv @ bandit.b[arm_i]
        scores[arm_i] = x @ theta
    a = int(np.argmax(scores))
    choices[arm_names[a]] += 1

total = sum(choices.values())
for name in arm_names:
    cnt = choices.get(name, 0)
    print(f"  {name:10s}: {cnt:4d}  ({100*cnt/max(total,1):5.1f}%)")

# ============================================================
# 7. ОЦЕНКА БАНДИТА VS КОМПОНЕНТЫ
# ============================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ: BANDIT vs ОТДЕЛЬНЫЕ МОДЕЛИ")
print("="*60)

results = []
results.append(evaluate(recommend_content, test, name="Content only"))
results.append(evaluate(recommend_svd,     test, name="SVD only"))
results.append(evaluate(recommend_hybrid,  test, name="Cascade Hybrid"))
results.append(evaluate(recommend_bandit_over_models, test,
                        name="Bandit over models"))

res_df = pd.DataFrame(results).set_index('Model')
print(res_df)

# Проверка превосходства
bandit_ndcg  = res_df.loc['Bandit over models', 'NDCG@10']
best_single = res_df.drop('Bandit over models')['NDCG@10'].max()
best_name = res_df.drop('Bandit over models')['NDCG@10'].idxmax()

print(f"\nBandit NDCG@10:  {bandit_ndcg:.4f}")
print(f"Best single ({best_name}): {best_single:.4f}")
if bandit_ndcg > best_single:
    print(f"✅ Bandit лучше лучшей одиночной модели на {100*(bandit_ndcg/best_single-1):+.1f}%")
else:
    print(f"⚠ Bandit пока хуже лучшей модели. Можно увеличить alpha или число sim_users")

# ============================================================
# 8. ДЕМО
# ============================================================
if 'sample_user' in dir():
    x = get_context(sample_user)
    scores = np.zeros(n_arms)
    for arm_i in range(n_arms):
        A_inv = np.linalg.inv(bandit.A[arm_i])
        theta = A_inv @ bandit.b[arm_i]
        scores[arm_i] = x @ theta
    chosen = arm_names[int(np.argmax(scores))]
    print(f"\nДля юзера {sample_user} Bandit выбрал arm: {chosen}")
    print(f"Скоры рук: {dict(zip(arm_names, scores.round(3)))}")
    print(f"\nРекомендации:")
    print(show_books(recommend_bandit_over_models(sample_user)))

# Делаем основным
recommend_bandit = recommend_bandit_over_models
print("\n✓ recommend_bandit обновлён на Bandit-over-Models")