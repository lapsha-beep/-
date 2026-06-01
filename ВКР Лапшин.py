"""
Модель и алгоритм выбора средств защиты удалённого интернет-сервера.
Версия 2 (минимальный набор, покрывающий угрозы).

Исходные данные:
  - Threats: name, P (вероятность), L (ущерб в тысячах)
  - Protections: name, cost (стоимость в тысячах)
  - Reduction: threat_id, и столбцы с коэффициентами r_ij для каждого СЗИ

Формулы:
  - Исходный риск: R_init = P * L
  - После выбора набора X: для каждой угрозы max_r = max_{j: x_j=1} r_ij
  - Остаточный риск: R_res = R_init * (1 - max_r)
  - Цель: минимизировать сумму R_res при условии sum(cost) <= бюджет
  - Затем из полученного набора удаляются избыточные средства (те, без которых остаточный риск не меняется)

Результат: минимальный по составу и стоимости набор, дающий наименьший возможный остаточный риск в рамках бюджета.
"""

import pandas as pd
import itertools

def load_data(filepath):
    """Загрузка листов Excel."""
    threats = pd.read_excel(filepath, sheet_name="Threats")
    protections = pd.read_excel(filepath, sheet_name="Protections")
    reduction = pd.read_excel(filepath, sheet_name="Reduction")
    # Проверка наличия столбца L
    if "L" not in threats.columns:
        raise ValueError("В листе Threats отсутствует столбец L (ущерб в тысячах).")
    return threats, protections, reduction

def initial_risk(threat):
    """Риск для одной угрозы: P * L."""
    return threat["P"] * threat["L"]

def total_cost(protections, x):
    """Суммарная стоимость выбранных средств."""
    return sum(protections.iloc[j]["cost"] * x[j] for j in range(len(protections)))

def compute_residuals(threats, reduction_matrix, x):
    """
    Для заданного вектора x (список 0/1) вычисляет:
      - список остаточных рисков по угрозам,
      - список max_r по угрозам,
      - суммарный остаточный риск.
    """
    residual_risks = []
    max_rs = []
    for i in range(len(threats)):
        threat = threats.iloc[i]
        R_init = initial_risk(threat)
        reductions = reduction_matrix.iloc[i, 1:].values  # все коэффициенты для этой угрозы
        max_r = max((reductions[j] * x[j] for j in range(len(x))), default=0)
        R_res = R_init * (1 - max_r)
        residual_risks.append(R_res)
        max_rs.append(max_r)
    return residual_risks, max_rs, sum(residual_risks)

def optimize_security(threats, protections, reduction_matrix, budget):
    """
    Полный перебор: ищет набор с минимальным суммарным остаточным риском.
    Затем удаляет избыточные средства (те, без которых риск не увеличивается).
    Возвращает (x_list, cost, total_residual).
    """
    m = len(protections)
    best_solution = None
    best_residual = float('inf')
    # Перебор всех комбинаций
    for x in itertools.product([0, 1], repeat=m):
        cost = total_cost(protections, x)
        if cost > budget:
            continue
        _, _, total_res = compute_residuals(threats, reduction_matrix, x)
        if total_res < best_residual:
            best_residual = total_res
            best_solution = list(x)
    if best_solution is None:
        return None  # нет допустимых комбинаций

    # Удаление избыточных средств: если убрать j-е средство, и остаточный риск не изменится,
    # то оно не нужно.
    changed = True
    while changed:
        changed = False
        for j in range(m):
            if best_solution[j] == 0:
                continue
            # Пробуем удалить j-е средство
            x_test = best_solution.copy()
            x_test[j] = 0
            _, _, test_res = compute_residuals(threats, reduction_matrix, x_test)
            if abs(test_res - best_residual) < 1e-6:  # риск не изменился (с учётом погрешности)
                best_solution = x_test
                best_residual = test_res
                changed = True
                break  # начинаем заново, так как индексы сместились
    final_cost = total_cost(protections, best_solution)
    return best_solution, final_cost, best_residual

def detailed_report(threats, protections, reduction_matrix, x, budget):
    """Вывод подробного отчёта."""
    print("\n" + "="*70)
    print(" ВЫБОР СРЕДСТВ ЗАЩИТЫ (минимальный набор, покрывающий угрозы)")
    print("="*70)

    # Исходные риски
    total_initial = 0.0
    print("\n1. ИСХОДНЫЕ РИСКИ (без защиты):")
    for i, threat in threats.iterrows():
        R_init = initial_risk(threat)
        total_initial += R_init
        print(f"   {threat['name']}: P={threat['P']:.4f}, L={threat['L']:.2f} тыс. → R_initial = {R_init:.2f} тыс.")
    print(f"   СУММАРНЫЙ ИСХОДНЫЙ РИСК: {total_initial:.2f} тыс.")

    # Выбранные меры
    print("\n2. ВЫБРАННЫЕ СРЕДСТВА ЗАЩИТЫ (в рамках бюджета):")
    selected = []
    total_cost_val = 0.0
    for j, (_, prot) in enumerate(protections.iterrows()):
        if x[j]:
            selected.append(prot["name"])
            total_cost_val += prot["cost"]
            print(f"   - {prot['name']} (стоимость: {prot['cost']:.2f} тыс.)")
    if not selected:
        print("   (ни одно средство не выбрано)")
    print(f"   Суммарная стоимость: {total_cost_val:.2f} тыс.")
    print(f"   Бюджет: {budget:.2f} тыс.")

    # Остаточные риски
    residual_risks, max_rs, total_residual = compute_residuals(threats, reduction_matrix, x)
    print("\n3. ОСТАТОЧНЫЕ РИСКИ ПОСЛЕ ЗАЩИТЫ:")
    for i, threat in threats.iterrows():
        R_init = initial_risk(threat)
        print(f"   {threat['name']}: исходный = {R_init:.2f} тыс., max_r = {max_rs[i]:.4f}, остаточный = {residual_risks[i]:.2f} тыс.")
    print(f"\n   СУММАРНЫЙ ОСТАТОЧНЫЙ РИСК: {total_residual:.2f} тыс.")
    print(f"   ПРЕДОТВРАЩЁННЫЙ УЩЕРБ (снижение риска): {total_initial - total_residual:.2f} тыс.")

    print("\n4. ОПТИМАЛЬНОСТЬ:")
    print(f"   Выбранный набор даёт минимальный суммарный остаточный риск ({total_residual:.2f} тыс.)")
    print(f"   и не содержит избыточных средств (удаление любого увеличит риск).")
    print("="*70)

def main():
    filepath = "risk_model.xlsx"  # имя вашего файла
    try:
        threats, protections, reduction = load_data(filepath)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    try:
        budget = float(input("Введите бюджет (тыс. руб): "))
    except ValueError:
        print("Ошибка: необходимо ввести число.")
        return

    result = optimize_security(threats, protections, reduction, budget)
    if result is None:
        print("Нет допустимых комбинаций (возможно, бюджет слишком мал).")
        return

    x, cost, residual = result
    detailed_report(threats, protections, reduction, x, budget)

if __name__ == "__main__":
    main()