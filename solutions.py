# ============================================================
# OFFLINE EVALUATION FRAMEWORK
# Универсальная система для оценки и сравнения РС
# Можно подключить свою РС или РС компаньона
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json
import time

print("="*70)
print("OFFLINE EVALUATION FRAMEWORK FOR RECOMMENDER SYSTEMS")
print("="*70)

# ============================================================
# 1. СТАНДАРТ ОБМЕНА: ИНТЕРФЕЙС РЕКОМЕНДЕРА
# ============================================================
"""
СПЕЦИФИКАЦИЯ ИНТЕРФЕЙСА

Любая РС должна реализовывать функцию:
    
    def recommend(user_id, top_n=10) -> list[str]
        '''
        Возвращает упорядоченный список ISBN-ов из top_n рекомендаций
        для пользователя user_id.
        
        Args:
            user_id: ID пользователя (int или str)
            top_n: количество рекомендаций
            
        Returns:
            Список ISBN. Если рекомендации невозможны — пустой список.
            ISBN из user history (train) исключаются.
        '''

ОБЩИЕ ДАННЫЕ (одинаковые для всех РС):
    train: pd.DataFrame с колонками [User-ID, ISBN, Rating]
    test:  pd.DataFrame с колонками [User-ID, ISBN, Rating]
    books: pd.DataFrame с колонками [ISBN, Title, Author, Publisher, ...]

КРИТЕРИЙ РЕЛЕВАНТНОСТИ: Rating >= 7 в test
"""

# ============================================================
# 2. БАЗОВЫЕ МЕТРИКИ (точность)
# ============================================================

def precision_at_k(recs, relevant, k):
    if not recs: return 0.0
    return sum(1 for isbn in recs[:k] if isbn in relevant) / k

def recall_at_k(recs, relevant, k):
    if not relevant or not recs: return 0.0
    return sum(1 for isbn in recs[:k] if isbn in relevant) / len(relevant)

def hit_at_k(recs, relevant, k):
    return 1.0 if any(isbn in relevant for isbn in recs[:k]) else 0.0

def average_precision(recs, relevant, k):
    if not relevant or not recs: return 0.0
    ap, hits = 0.0, 0
    for i, isbn in enumerate(recs[:k]):
        if isbn in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / min(len(relevant), k)

def ndcg_at_k(recs, relevant, k):
    if not relevant or not recs: return 0.0
    dcg = sum(1.0/np.log2(i+2) for i, isbn in enumerate(recs[:k]) if isbn in relevant)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ============================================================
# 3. МЕТРИКИ ВНЕ ТОЧНОСТИ (важны для качества РС)
# ============================================================

def coverage(all_recs, n_books_total):
    """Catalog Coverage: какую долю каталога РС вообще рекомендует"""
    unique_recommended = set()
    for recs in all_recs:
        unique_recommended.update(recs)
    return len(unique_recommended) / n_books_total

def diversity_intra_list(recs, item_features, mat):
    """Intra-list diversity: средняя НЕпохожесть рекомендаций между собой"""
    if len(recs) < 2: return 0.0
    indices = [item_features[isbn] for isbn in recs if isbn in item_features]
    if len(indices) < 2: return 0.0
    
    from sklearn.metrics.pairwise import cosine_similarity
    vecs = mat[indices]
    sims = cosine_similarity(vecs)
    # Берём верхний треугольник без диагонали
    n = sims.shape[0]
    mean_sim = (sims.sum() - n) / (n * (n - 1))
    return 1.0 - mean_sim  # diversity = 1 - similarity

def novelty(recs, popularity_dict):
    """Novelty: -log(p(item)) — насколько редкие книги рекомендуем"""
    if not recs: return 0.0
    total = sum(popularity_dict.values())
    novelties = []
    for isbn in recs:
        p = popularity_dict.get(isbn, 1) / total
        novelties.append(-np.log2(p))
    return np.mean(novelties)

def serendipity(recs, relevant, popularity_dict):
    """
    Serendipity: релевантные И при этом непопулярные рекомендации
    (РС угадала не банальные, а оригинальные предпочтения)
    """
    if not recs: return 0.0
    pop_values = list(popularity_dict.values())
    median_pop = np.median(pop_values) if pop_values else 0
    
    score = 0
    for isbn in recs:
        if isbn in relevant and popularity_dict.get(isbn, 0) < median_pop:
            score += 1
    return score / len(recs)

# ============================================================
# 4. ГЛАВНАЯ ФУНКЦИЯ ОЦЕНИВАНИЯ
# ============================================================

# Готовим вспомогательные структуры
book_pop_dict = dict(zip(
    [idx2book[i] for i in range(n_books)],
    book_pop.tolist()
))

# Test relevance
test_relevant = defaultdict(set)
for _, r in test[test['Rating'] >= 7].iterrows():
    test_relevant[r['User-ID']].add(r['ISBN'])

# Категории юзеров по размеру истории
user_history_size = train.groupby('User-ID').size()
cold_users   = set(user_history_size[user_history_size < 5].index)
warm_users   = set(user_history_size[(user_history_size >= 5) & (user_history_size < 20)].index)
heavy_users  = set(user_history_size[user_history_size >= 20].index)


def evaluate_full(recommend_fn, name="Model", k=10, 
                  max_users=500, verbose=True):
    """
    Полная оценка РС со всеми метриками
    """
    # Юзеры из test с релевантными книгами
    eligible_users = [u for u in test_relevant if len(test_relevant[u]) > 0]
    
    np.random.seed(42)
    if len(eligible_users) > max_users:
        eligible_users = np.random.choice(eligible_users, max_users, replace=False).tolist()
    
    # Метрики
    P, R, H, AP, NDCG = [], [], [], [], []
    NOV, SER = [], []
    
    # Метрики по сегментам юзеров
    metrics_by_segment = defaultdict(lambda: {'NDCG': [], 'HR': []})
    
    all_recs = []
    inference_times = []
    failed = 0
    
    iterator = tqdm(eligible_users, desc=f"Evaluating {name}") if verbose else eligible_users
    
    for uid in iterator:
        relevant = test_relevant[uid]
        
        t0 = time.time()
        try:
            recs = recommend_fn(uid, top_n=k)
        except Exception as e:
            failed += 1
            continue
        inference_times.append(time.time() - t0)
        
        if not recs:
            failed += 1
            continue
        
        all_recs.append(recs)
        
        # Точность
        P.append(precision_at_k(recs, relevant, k))
        R.append(recall_at_k(recs, relevant, k))
        H.append(hit_at_k(recs, relevant, k))
        AP.append(average_precision(recs, relevant, k))
        NDCG.append(ndcg_at_k(recs, relevant, k))
        
        # Качество
        NOV.append(novelty(recs, book_pop_dict))
        SER.append(serendipity(recs, relevant, book_pop_dict))
        
        # По сегментам
        ndcg_val = NDCG[-1]
        hr_val   = H[-1]
        if uid in cold_users:
            metrics_by_segment['cold']['NDCG'].append(ndcg_val)
            metrics_by_segment['cold']['HR'].append(hr_val)
        elif uid in warm_users:
            metrics_by_segment['warm']['NDCG'].append(ndcg_val)
            metrics_by_segment['warm']['HR'].append(hr_val)
        elif uid in heavy_users:
            metrics_by_segment['heavy']['NDCG'].append(ndcg_val)
            metrics_by_segment['heavy']['HR'].append(hr_val)
    
    # Coverage (по всем рекомендациям)
    cov = coverage(all_recs, n_books)
    
    # Diversity (на подвыборке)
    div_scores = []
    for recs in all_recs[:100]:
        div_scores.append(diversity_intra_list(recs, isbn2tfidf, tfidf_matrix))
    
    result = {
        'Model': name,
        # ==== ТОЧНОСТЬ ====
        f'Precision@{k}':  round(np.mean(P), 4),
        f'Recall@{k}':     round(np.mean(R), 4),
        f'HitRate@{k}':    round(np.mean(H), 4),
        f'MAP@{k}':        round(np.mean(AP), 4),
        f'NDCG@{k}':       round(np.mean(NDCG), 4),
        # ==== КАЧЕСТВО ====
        'Coverage':        round(cov, 4),
        'Diversity':       round(np.mean(div_scores), 4) if div_scores else 0,
        'Novelty':         round(np.mean(NOV), 2),
        'Serendipity':     round(np.mean(SER), 4),
        # ==== СЕГМЕНТЫ ====
        'NDCG@cold':       round(np.mean(metrics_by_segment['cold']['NDCG']), 4) if metrics_by_segment['cold']['NDCG'] else 0,
        'NDCG@warm':       round(np.mean(metrics_by_segment['warm']['NDCG']), 4) if metrics_by_segment['warm']['NDCG'] else 0,
        'NDCG@heavy':      round(np.mean(metrics_by_segment['heavy']['NDCG']), 4) if metrics_by_segment['heavy']['NDCG'] else 0,
        # ==== ПРОИЗВОДИТЕЛЬНОСТЬ ====
        'Avg_time_ms':     round(np.mean(inference_times) * 1000, 1) if inference_times else 0,
        'Failed':          failed,
        'N_evaluated':     len(P),
    }
    return result

# ============================================================
# 5. РЕЕСТР МОДЕЛЕЙ (можно добавить РС компаньона)
# ============================================================

# Свои модели
my_models = {
    'Content':         recommend_content,
    'SVD':             recommend_svd,
    'Cascade Hybrid':  recommend_hybrid,
    'Bandit Ensemble': recommend_bandit,
}

# === МЕСТО ДЛЯ РС КОМПАНЬОНА ===
# Закомментируйте/раскомментируйте при обмене:
#
# def recommend_companion(user_id, top_n=10):
#     # код РС компаньона
#     return [...]
# 
# my_models['Companion RS'] = recommend_companion

# ============================================================
# 6. ЗАПУСК ОЦЕНИВАНИЯ
# ============================================================
print("\n" + "="*70)
print("ОЦЕНИВАНИЕ МОДЕЛЕЙ")
print("="*70)

all_results = []
for name, fn in my_models.items():
    print(f"\n→ {name}")
    res = evaluate_full(fn, name=name, k=10, max_users=500, verbose=False)
    all_results.append(res)
    print(f"  NDCG@10 = {res['NDCG@10']}, Coverage = {res['Coverage']}, "
          f"Diversity = {res['Diversity']}")

results_df = pd.DataFrame(all_results).set_index('Model')

# ============================================================
# 7. РЕЗУЛЬТАТЫ В НЕСКОЛЬКИХ ТАБЛИЦАХ
# ============================================================
print("\n" + "="*70)
print("ТАБЛИЦА 1: МЕТРИКИ ТОЧНОСТИ")
print("="*70)
accuracy_cols = ['Precision@10', 'Recall@10', 'HitRate@10', 'MAP@10', 'NDCG@10']
print(results_df[accuracy_cols].to_string())

print("\n" + "="*70)
print("ТАБЛИЦА 2: МЕТРИКИ КАЧЕСТВА (вне точности)")
print("="*70)
quality_cols = ['Coverage', 'Diversity', 'Novelty', 'Serendipity']
print(results_df[quality_cols].to_string())

print("\n" + "="*70)
print("ТАБЛИЦА 3: ПРОИЗВОДИТЕЛЬНОСТЬ НА СЕГМЕНТАХ ЮЗЕРОВ")
print("="*70)
segment_cols = ['NDCG@cold', 'NDCG@warm', 'NDCG@heavy']
print(results_df[segment_cols].to_string())
print(f"\nРазмеры сегментов: cold={len(cold_users)}, warm={len(warm_users)}, heavy={len(heavy_users)}")

print("\n" + "="*70)
print("ТАБЛИЦА 4: ПРОИЗВОДИТЕЛЬНОСТЬ")
print("="*70)
print(results_df[['Avg_time_ms', 'Failed', 'N_evaluated']].to_string())

# ============================================================
# 8. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График 1: метрики точности
ax = axes[0, 0]
results_df[accuracy_cols].plot(kind='bar', ax=ax)
ax.set_title('Точность по моделям', fontsize=12, fontweight='bold')
ax.set_xticklabels(results_df.index, rotation=20)
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# График 2: качество
ax = axes[0, 1]
norm_df = results_df[quality_cols].copy()
for col in norm_df.columns:
    if norm_df[col].max() > 0:
        norm_df[col] = norm_df[col] / norm_df[col].max()
norm_df.plot(kind='bar', ax=ax)
ax.set_title('Качество РС (нормализовано)', fontsize=12, fontweight='bold')
ax.set_xticklabels(results_df.index, rotation=20)
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# График 3: сегменты юзеров
ax = axes[1, 0]
results_df[segment_cols].plot(kind='bar', ax=ax)
ax.set_title('NDCG@10 по сегментам юзеров', fontsize=12, fontweight='bold')
ax.set_xticklabels(results_df.index, rotation=20)
ax.legend(['Cold (<5)', 'Warm (5-20)', 'Heavy (20+)'], fontsize=8)
ax.grid(axis='y', alpha=0.3)

# График 4: скорость
ax = axes[1, 1]
results_df['Avg_time_ms'].plot(kind='bar', ax=ax, color='coral')
ax.set_title('Время выполнения, мс/запрос', fontsize=12, fontweight='bold')
ax.set_xticklabels(results_df.index, rotation=20)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('OFFLINE EVALUATION: Сравнение РС', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 9. ИТОГИ И ПОБЕДИТЕЛЬ
# ============================================================
print("\n" + "="*70)
print("ИТОГИ")
print("="*70)

best_by_metric = {}
for col in ['NDCG@10', 'MAP@10', 'HitRate@10', 'Coverage', 'Diversity', 'Novelty']:
    if col in results_df.columns:
        best = results_df[col].idxmax()
        best_by_metric[col] = (best, results_df.loc[best, col])

print("\nЛидеры по каждой метрике:")
for metric, (model, val) in best_by_metric.items():
    print(f"  {metric:15s} → {model:20s} ({val})")

# Финальный рейтинг по NDCG@10
print("\nИТОГОВЫЙ РЕЙТИНГ ПО NDCG@10:")
ranking = results_df.sort_values('NDCG@10', ascending=False)
for i, (model, row) in enumerate(ranking.iterrows(), 1):
    medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
    print(f"  {medal} {i}. {model:25s} NDCG@10 = {row['NDCG@10']}")

# ============================================================
# 10. ЭКСПОРТ РЕЗУЛЬТАТОВ (для обмена с компаньоном)
# ============================================================
results_df.to_csv('rs_evaluation_results.csv')
results_df.to_json('rs_evaluation_results.json', orient='index', indent=2)
print("\n✓ Результаты сохранены: rs_evaluation_results.csv / .json")

# ============================================================
# 11. ИНСТРУКЦИЯ ДЛЯ КОМПАНЬОНА
# ============================================================
print("\n" + "="*70)
print("ИНСТРУКЦИЯ ДЛЯ ОБМЕНА С КОМПАНЬОНОМ")
print("="*70)
print("""
Чтобы протестировать РС компаньона:

1. Получите от него функцию следующего вида:

    def recommend_companion(user_id, top_n=10) -> list[str]:
        # ISBN в порядке убывания релевантности
        return [...]

2. Добавьте её в my_models:

    my_models['Companion RS'] = recommend_companion

3. Перезапустите ячейку. Все метрики будут посчитаны автоматически.

4. Сравните результаты в итоговой таблице.

ВАЖНО: РС компаньона должна использовать те же:
  - train (для обучения, доступны все колонки)
  - books (метаданные)
  - НЕ должна заглядывать в test (это утечка!)
""")