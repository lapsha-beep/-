"""
Программная реализация модели оценки рисков информационной безопасности
удаленного веб-сервера.

Реализуются следующие формулы.

1. Исходный риск:
Ri = Pi * Ii * Ci

2. Остаточная вероятность:
Pi' = Pi * Π(1 - rij * xj)

3. Остаточный риск:
Ri' = Pi' * Ii * Ci

4. Предотвращённый ущерб:
Ui = Ri - Ri'

5. Совокупный предотвращённый ущерб:
U(x) = Σ Ui

6. Стоимость мер защиты:
Cost(x) = Σ cj * xj

7. ROSI:
ROSI = (Prevented - Cost) / Cost
"""

import pandas as pd
import itertools


# ==============================
# Загрузка данных из Excel
# ==============================

def load_data(filepath):
    threats = pd.read_excel(filepath, sheet_name="Threats")
    protections = pd.read_excel(filepath, sheet_name="Protections")
    reduction = pd.read_excel(filepath, sheet_name="Reduction")
    return threats, protections, reduction


# ==============================
# Исходный риск
# ==============================

def calculate_initial_risk(threat):
    P = threat["P"]
    I = threat["I"]
    C = threat["C"]
    return P * I * C


# ==============================
# Остаточная вероятность
# ==============================

def residual_probability(P, reductions, x):
    prob = P
    for r, xi in zip(reductions, x):
        prob *= (1 - r * xi)
    return prob


# ==============================
# Остаточный риск
# ==============================

def residual_risk(threat, reductions, x):
    P_res = residual_probability(threat["P"], reductions, x)
    return P_res * threat["I"] * threat["C"]


# ==============================
# Стоимость набора мер
# ==============================

def calculate_cost(protections, x):
    cost = 0
    for xi, (_, protection) in zip(x, protections.iterrows()):
        cost += protection["cost"] * xi
    return cost


# ==============================
# Подробный отчёт по вероятностям
# ==============================

def detailed_probability_report(threats, protections, reduction_matrix, best_x):
    """
    Выводит:
      - исходные вероятности угроз,
      - влияние каждой отдельной меры защиты на вероятности,
      - итоговые остаточные вероятности после применения выбранного набора мер.
    """
    print("\n===== ПОДРОБНЫЙ АНАЛИЗ ВЕРОЯТНОСТЕЙ =====")

    # 1. Исходные вероятности
    print("\nИсходные вероятности угроз:")
    for idx, threat in threats.iterrows():
        print(f"  Угроза '{threat['name']}': P = {threat['P']:.4f}")

    # 2. Влияние каждой отдельной меры защиты
    print("\nВлияние каждой меры защиты на вероятности (если применена только она):")
    m = len(protections)
    for j, protection in protections.iterrows():
        x_single = [0] * m
        x_single[j] = 1
        print(f"\n  Мера '{protection['name']}':")
        for i, threat in threats.iterrows():
            reductions = reduction_matrix.iloc[i, 1:].values
            p_res = residual_probability(threat["P"], reductions, x_single)
            print(f"    Угроза '{threat['name']}': P' = {p_res:.4f} "
                  f"(снижение на {(threat['P'] - p_res):.4f})")

    # 3. Итоговые вероятности после выбранного набора мер
    print("\nИтоговые остаточные вероятности после применения ВЫБРАННОГО набора мер:")
    for i, threat in threats.iterrows():
        reductions = reduction_matrix.iloc[i, 1:].values
        p_final = residual_probability(threat["P"], reductions, best_x)
        print(f"  Угроза '{threat['name']}': P' = {p_final:.4f} "
              f"(исходная = {threat['P']:.4f})")


# ==============================
# Оптимизация выбора мер
# ==============================

def optimize_security(threats, protections, reduction_matrix, budget):
    m = len(protections)

    initial_risks = [
        calculate_initial_risk(row)
        for _, row in threats.iterrows()
    ]
    total_initial_risk = sum(initial_risks)

    best_solution = None
    best_score = -999999999

    for x in itertools.product([0, 1], repeat=m):
        cost = calculate_cost(protections, x)
        if cost > budget:
            continue

        residual_total = 0
        for i, threat in threats.iterrows():
            reductions = reduction_matrix.iloc[i, 1:].values
            r = residual_risk(threat, reductions, x)
            residual_total += r

        prevented = total_initial_risk - residual_total

        if cost == 0:
            rosi = 0
        else:
            rosi = (prevented - cost) / cost

        score = rosi

        if score > best_score:
            best_score = score
            best_solution = (x, cost, residual_total, rosi)

    return best_solution, total_initial_risk


# ==============================
# Основная программа
# ==============================

def main():
    filepath = "risk_model.xlsx"
    threats, protections, reduction = load_data(filepath)

    budget = float(input("Введите доступный бюджет: "))

    result, initial_risk = optimize_security(
        threats,
        protections,
        reduction,
        budget
    )

    if result is None:
        print("Оптимальное решение не найдено.")
        return

    x, cost, residual_risk_total, rosi = result
    prevented = initial_risk - residual_risk_total

    print("\n===== РЕЗУЛЬТАТ ОПТИМИЗАЦИИ =====\n")
    print("Исходный совокупный риск:", round(initial_risk, 2))
    print("Остаточный риск:", round(residual_risk_total, 2))
    print("Предотвращённый ущерб:", round(prevented, 2))
    print("Стоимость мер:", cost)
    print("ROSI:", round(rosi, 3))

    print("\nВыбранные меры защиты:")
    selected_protections = []
    total_cost = 0
    for xi, (_, protection) in zip(x, protections.iterrows()):
        if xi == 1:
            selected_protections.append(protection["name"])
            print(f"  - {protection['name']} (стоимость: {protection['cost']})")
            total_cost += protection['cost']

    # Дополнительная информация: вероятность до и после
    print("\n--- Сводка по вероятностям ---")
    print("Исходные вероятности угроз:")
    for _, threat in threats.iterrows():
        print(f"  {threat['name']}: P = {threat['P']:.4f}")

    print("\nОстаточные вероятности после применения выбранных мер:")
    for i, threat in threats.iterrows():
        reductions = reduction.iloc[i, 1:].values
        p_res = residual_probability(threat["P"], reductions, x)
        print(f"  {threat['name']}: P' = {p_res:.4f}")

    print(f"\nСуммарная стоимость выбранных мер: {total_cost}")

    # Вызов подробного отчёта
    detailed_probability_report(threats, protections, reduction, x)


if __name__ == "__main__":
    main()
