import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import seaborn as sns

# قراءة ملف CSV وإعداد بيانات المدخلات لـ Apriori
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# تدريب نموذج Apriori على المجموعة البيانات
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# تحويل النتائج إلى DataFrame
results_df = pd.DataFrame(columns=('Rule', 'Support', 'Confidence', 'Lift'))

for rule in rules:
    support = rule.support
    for order in rule.ordered_statistics:
        lhs = ", ".join(list(order.items_base))
        rhs = ", ".join(list(order.items_add))
        confidence = order.confidence
        lift = order.lift
        results_df.loc[len(results_df)] = [f"{lhs} => {rhs}", support, confidence, lift]

# إضافة عمود للقواعد المهمة
results_df['Interesting'] = np.where(results_df['Lift'] >= 4, '√', 'X')

# عرض النتائج
if len(results_df) > 0:
    print("النتائج:")
    print(results_df.sort_values(by=['Lift'], ascending=False)[['Rule', 'Support', 'Confidence', 'Lift', 'Interesting']])
else:
    print("لا يوجد نتائج.")

# إظهار مخطط الإحداثيات
sns.scatterplot(x='Support', y='Lift', hue='Interesting', data=results_df)

# إضافة عرض عنصر Rule لمخطط الإحداثيات
for i in range(len(results_df)):
    plt.annotate(results_df.loc[i]['Rule'], xy=(results_df.loc[i]['Support'], results_df.loc[i]['Lift']))

plt.show()