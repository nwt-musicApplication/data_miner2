#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# In[4]:


data_path = "F:\研究生课程\数据挖掘\作业4\winemag-data-130k-v2.csv"  # 数据读取路径
calendar = pd.read_csv(data_path)
new_calendar = calendar.dropna()
new_calendar = new_calendar[['country', 'province', 'region_1', 'region_2', 'variety', 'winery']] # 选择此六项进行数据分析
new_calendar.reset_index(drop=True, inplace=True)


# In[5]:


record = []   # 将数据转换为record
for index in range(len(new_calendar)):
    record.append(list(new_calendar.loc[index]))
Encoder = TransactionEncoder()
encoded_data = Encoder.fit_transform(record)   # record转换为True or False 标记的数据
new_calendar = pd.DataFrame(encoded_data, columns=Encoder.columns_)


# 频繁模式挖掘 Apriori算法   设定支持度最小阈值 0.05

# In[17]:


pd.set_option('display.max_rows',100)
frequent_item_sets = apriori(new_calendar, min_support=0.05, use_colnames=True, max_len=None).sort_values(by='support', ascending=False)
display(frequent_item_sets)


# In[31]:


sns.set(rc={'figure.figsize': (19.7, 8.27)})   #支持度top20可视化
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=8)
sns.set(font=myfont.get_name(), font_scale=1)
sns.barplot(x="itemsets", y="support", data=frequent_item_sets[:20])
plt.xticks(rotation=45)
plt.show()


# 置信度 Confidence
#     Confidence(X->Y) = Support(X∧Y) / Support(X)
# 提升度 Lift
#     Lift(X->Y)= Support(X∧Y) / (Support(X) * Support(Y))
# 杠杆率 Leverage
#     Levarage(X->Y) = Support(X∧Y) - (Support(X) * Support(Y))
# 出错率 Conviction
#     Conviction(X->Y) = (1 - Support(Y)) / (1 - Confidence(X->Y))

# 关联规则挖掘 先以置信度为衡量指标并排序 最小阈值0.8

# In[46]:


ass_rule = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.8)
ass_rule.sort_values(by='confidence', ascending=False, inplace=True)
pd.set_option('display.max_rows', 1000)
display(ass_rule)


#   处理完的数据中country项全为US,所以以consequents为US的关联规则置信度都为1，因此不具有参考价值，也说明Confidence衡量下对于某数据项都为一个值的数据不好处理。
#     
#   分析数据项使用数据表中的['country', 'province', 'region_1', 'region_2', 'variety', 'winery']进行分析
#   其中country,province,region_1,region_2都为地理位置，存在所属关系，其置信度值分析意义不大，但从另一方面证明置信度分析的有效性。
#   例如表中第三项(Russian River Valley)——>(Sonoma, California, US)，置信度值为1，实际US——>California——>Sonoma——>Russian River Valley分别属于数据中的country,province,region_2,region_1项。
#    
#   规则表中的最后一项(Oregon, Pinot Noir)——>(Willamette Valley)
#   Oregon属于province项，Willamette Valley数据region_1项，Pinot Noir属于variety项，表示俄勒冈州(Oregon)生产的黑比诺品种(Point Noir)的葡萄酒在威拉米特谷(Willamette Valley)置信度为0.920533。从网络上查找数据证实俄勒冈州(Oregon)的威拉米特谷(Willamette Valley)是黑比诺品种(Point Noir)的葡萄酒的重要原产地。

# In[47]:


import networkx as nx
node_set = set()
edge_set = set()
ass_rule.reset_index(drop=True, inplace=True)
for i in range(len(ass_rule)):
    node_set.add(str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""))
    node_set.add(str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    tuple_tmp = (str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""),
                 str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    edge_set.add(tuple_tmp)
graph = nx.DiGraph()
for i in list(node_set):
    graph.add_node(i)
for i in list(edge_set):
    graph.add_edge(i[0], i[1])
pos = nx.circular_layout(graph)
nx.draw_networkx_nodes(graph, pos)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# In[48]:


pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_size=50,)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# 关联规则挖掘 以提升度为衡量指标并排序 最小阈值0.8

# In[49]:


ass_rule = association_rules(frequent_item_sets, metric='lift', min_threshold=0.8)
ass_rule.sort_values(by='lift', ascending=False, inplace=True)
pd.set_option('display.max_rows', 1000)
display(ass_rule)


# In[50]:


import networkx as nx
node_set = set()
edge_set = set()
ass_rule.reset_index(drop=True, inplace=True)
for i in range(len(ass_rule)):
    node_set.add(str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""))
    node_set.add(str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    tuple_tmp = (str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""),
                 str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    edge_set.add(tuple_tmp)
graph = nx.DiGraph()
for i in list(node_set):
    graph.add_node(i)
for i in list(edge_set):
    graph.add_edge(i[0], i[1])
pos = nx.circular_layout(graph)
nx.draw_networkx_nodes(graph, pos)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# In[51]:


pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_size=50,)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# In[ ]:


关联规则挖掘 以杠杆率为衡量指标并排序 最小阈值0.05


# In[23]:


ass_rule = association_rules(frequent_item_sets, metric='leverage', min_threshold=0.05)
ass_rule.sort_values(by='leverage', ascending=False, inplace=True)
pd.set_option('display.max_rows', 1000)
display(ass_rule)


# In[43]:


import networkx as nx
node_set = set()
edge_set = set()
ass_rule.reset_index(drop=True, inplace=True)
for i in range(len(ass_rule)):
    node_set.add(str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""))
    node_set.add(str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    tuple_tmp = (str(ass_rule.loc[i]['antecedents']).replace("frozenset(", "").replace(")", ""),
                 str(ass_rule.loc[i]['consequents']).replace("frozenset(", "").replace(")", ""))
    edge_set.add(tuple_tmp)
graph = nx.DiGraph()
for i in list(node_set):
    graph.add_node(i)
for i in list(edge_set):
    graph.add_edge(i[0], i[1])
pos = nx.circular_layout(graph)
nx.draw_networkx_nodes(graph, pos)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# In[44]:


pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_size=50,)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
plt.show()


# In[ ]:




