# -*- coding: utf-8 -*-
import time
from itertools import combinations

def join_step(L_prev, length):
    """Hàm 1: Sinh tập ứng viên (Candidate generation)."""
    candidates = set()
    L_list = list(L_prev)
    for i in range(len(L_list)):
        for j in range(i + 1, len(L_list)):
            itemset1 = sorted(L_list[i])
            itemset2 = sorted(L_list[j])
            # Chỉ ghép khi (length-2) phần tử đầu tiên giống nhau
            if itemset1[:length - 2] == itemset2[:length - 2]:
                candidate = frozenset(itemset1) | frozenset(itemset2)
                if len(candidate) == length:
                    candidates.add(candidate)
    return candidates

def prune_step(candidates, L_prev, length):
    """Hàm 2: Cắt tỉa (Pruning) bằng Tính chất Apriori."""
    pruned_candidates = set()
    for candidate in candidates:
        is_valid = True
        for subset in combinations(candidate, length - 1):
            if frozenset(subset) not in L_prev:
                is_valid = False
                break
        if is_valid:
            pruned_candidates.add(candidate)
    return pruned_candidates

def get_support(candidates, transactions, min_sup_count):
    """
    Hàm 3: Quét Database tính Độ hỗ trợ (Support) - NÚT THẮT I/O.
    """
    item_counts = {}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                item_counts[candidate] = item_counts.get(candidate, 0) + 1

    L_current = set()
    support_data = {}
    for itemset, count in item_counts.items():
        support_data[itemset] = count / len(transactions)
        if count >= min_sup_count:
            L_current.add(itemset)
    return L_current, support_data

def generate_rules(L_all, support_data, min_conf):
    """Sinh luật kết hợp (Association Rules)."""
    rules = []
    for L_k in L_all:
        for itemset in L_k:
            if len(itemset) < 2:
                continue
            for size in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, size):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent
                    if antecedent in support_data and support_data[antecedent] > 0:
                        confidence = support_data[itemset] / support_data[antecedent]
                        if confidence >= min_conf:
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support_data[itemset],
                                'confidence': confidence,
                            })
    rules.sort(key=lambda r: r['confidence'], reverse=True)
    return rules

def run_apriori(transactions, min_sup, min_conf=0.5):
    """Hàm điều phối lõi của thuật toán."""
    start_time = time.time()
    num_transactions = len(transactions)
    min_sup_count = min_sup * num_transactions

    print(f"  [-] Tiến hành quét không gian với min_sup = {min_sup}")
    
    candidates_per_level = {}
    
    C1 = {frozenset([item]) for t in transactions for item in t}
    candidates_per_level['C1'] = len(C1) # Lưu C1
    
    L_current, support_data = get_support(C1, transactions, min_sup_count)
    L_all = [L_current]
    total_candidates_seen = len(C1)

    k = 2
    while True:
        C_k_joined = join_step(L_current, k)
        if not C_k_joined:
            break
        
        candidates_per_level[f'C{k}'] = len(C_k_joined) # Lưu C2, C3, C4...

        C_k_pruned = prune_step(C_k_joined, L_current, k)
        if not C_k_pruned:
            break

        L_current, sup_data_k = get_support(C_k_pruned, transactions, min_sup_count)
        support_data.update(sup_data_k)
        total_candidates_seen += len(C_k_joined)
        
        if not L_current:
            break

        L_all.append(L_current)
        k += 1

    rules = generate_rules(L_all, support_data, min_conf)
    
    # Trả về dạng Dictionary để file main.py dễ dàng bóc tách dữ liệu
    return {
        'L_all': L_all,
        'support_data': support_data,
        'rules': rules,
        'total_candidates': total_candidates_seen,
        'candidates_per_level': candidates_per_level
    }