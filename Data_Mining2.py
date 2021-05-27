#!usr/bin/env python
# coding:utf-8

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
lift =[]
KULC =[]
IR = []
support = []

def load_data_set():
    csv_file = csv.reader(open('winemag-data-130k-v2.csv', encoding='utf-8')) #读取数据集
    for row in csv_file:
        llist = []
        if(row[4] == 'points'): #对数据集进行处理（选取points和country为例）
            continue
        if(row[1] != ""):
            llist.append(row[1])
            if(row[4] != ""):
                llist.append(int(float(row[4])/20)) #对points进行分级
        data_set.append(llist)
    return data_set

# 通过扫描数据集创建候选1项集C1
def create_C1(data_set):
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

# 判断常用候选k项集是否满足Apriori属性
def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True

# 创建ck，一个包含所有常见的候选k项集的集合
def create_Ck(Lksub1, k):
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning(剪枝)
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck

# 通过从ck执行删除策略生成Lk。
def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

# 生成所有频繁项集
def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

# 生成关联规则并通过三种方法对关联规则进行评估
def generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        lift.append(support_data[freq_set]/(support_data[sub_set]*support_data[freq_set-sub_set]))
                        KULC.append((support_data[freq_set]/support_data[sub_set]+support_data[freq_set]/support_data[freq_set-sub_set])/2)
                        IR.append((support_data[freq_set]/support_data[freq_set-sub_set])/support_data[freq_set]/support_data[sub_set])
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

# 主函数
if __name__ == "__main__":
    data_set = []
    conf_set = []
    support_set = []
    data_set = load_data_set()

    # 这里给出最小支持度为0.05，最小置信度为0.7
    L, support_data = generate_L(data_set, k=2, min_support=0.05)
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)
    for Lk in L:
        print("*" * 50)
        print("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
        print("*" * 50)
        for freq_set in Lk:
            print(freq_set, support_data[freq_set])

    # 打印(生成generate_rule.txt)关联规则及对规则的各评价结果
    print("\t\tBig Rules")
    i = 0
    # 取出频繁2项集support值
    for Lk in L:
        if (str(len(list(Lk)[0])) == '2'):
            for freq_set in Lk:
                support_set.append(support_data[freq_set])

    file = open("generate_rules.txt", "a", encoding="utf-8")
    for item in big_rules_list:
        print(item[0], "=>", item[1], "conf: ", item[2])
        file.write(str(item[0]) + " => " + str(item[1]) + " conf: " + str(item[2]) + '\n')
        # 取出频繁2项集confidence集
        conf_set.append(item[2])
        print("lift：", lift[i])
        file.write("lift：" + str(lift[i]) + '\n')
        print("KULC：", KULC[i])
        file.write("KULC：" + str(KULC[i]) + '\n')
        print("IR：", IR[i])
        file.write("IR：" + str(IR[i]) + '\n')
        i = i+1
    file.close()

    #可视化
    # 数据读取
    wine = pd.DataFrame(pd.read_csv('winemag-data-130k-v2.csv'))

    # country属性直方图
    plt.hist(x=wine['country'].dropna(), bins=50, edgecolor='black')
    # 添加x轴和y轴标签
    plt.xlabel('country')
    plt.ylabel('frequency')
    # 添加标题
    plt.title('Wine-Country distribution')
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=6)
    plt.show()

    # points属性直方图
    plt.hist(x=wine['points'], bins=100, edgecolor='black')
    # 添加x轴和y轴标签
    plt.xlabel('points')
    plt.ylabel('frequency')
    # 添加标题
    plt.title('Wine-Points distribution')
    plt.show()

    # 画出红酒产国与红酒评分散点图
    csv_file = csv.reader(open('winemag-data-130k-v2.csv', encoding='utf-8'))  # 读取数据集
    country = []
    points = []
    for row in csv_file:
        if (row[1] != ""):
            country.append(row[1])
            if (row[4] != ""):
                points.append(row[4])
    plt.title("Country&Points - Scatter")
    plt.xlabel('country')
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=6)
    plt.ylabel('points')
    plt.scatter(country, points, s=20, c="#ff1212", marker='o')
    plt.show()

    # 画（酒国家及酒评分的）support和confidence散点图
    plt.title("Country&Points - Wine")
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.legend()
    plt.scatter(support_set, conf_set, s=20, c="#ff1212", marker='o')
    plt.show()

    # Lift评估结果与support散点图
    plt.title("Lift & Support")
    plt.xlabel('support')
    plt.ylabel('Lift')
    plt.legend()
    plt.scatter(support_set, lift, s=20, c="#ff1212", marker='o')
    plt.show()
    # Lift评估结果与confidence散点图
    plt.title("Lift & Confidence")
    plt.xlabel('confidence')
    plt.ylabel('Lift')
    plt.legend()
    plt.scatter(conf_set, lift, s=20, c="#B9D3EE", marker='o')
    plt.show()

    # KULC评估结果与support散点图
    plt.title("KULC & Support")
    plt.xlabel('support')
    plt.ylabel('KULC')
    plt.legend()
    plt.scatter(support_set, KULC, s=20, c="#ff1212", marker='o')
    plt.show()
    # KULC评估结果与confidence散点图
    plt.title("KULC & Confidence")
    plt.xlabel('confidence')
    plt.ylabel('KULC')
    plt.legend()
    plt.scatter(conf_set, KULC, s=20, c="#B9D3EE", marker='o')
    plt.show()

    # IR评估结果与support散点图
    plt.title("IR & Support")
    plt.xlabel('support')
    plt.ylabel('IR')
    plt.legend()
    plt.scatter(support_set, IR, s=20, c="#ff1212", marker='o')
    plt.show()
    # IR评估结果与confidence散点图
    plt.title("IR & Confidence")
    plt.xlabel('confidence')
    plt.ylabel('IR')
    plt.legend()
    plt.scatter(conf_set, IR, s=20, c="#B9D3EE", marker='o')
    plt.show()

    # 支持度、置信度、关联规则评价结果直方图
    name_list = ['Italy =>4', 'US => 4', 'France => 4', 'Spain => 4']
    total_width, n = 0.4, 5
    width = total_width / n
    x = list(range(len(support_set)))
    plt.bar(x, support_set, width=width, label="support", fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, conf_set, width=width, label="confidence", fc='purple')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, lift, width=width, label="lift", fc='r', tick_label=name_list)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, KULC, width=width, label="KULC", fc='blue')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, IR, width=width, label="IR", fc='g')
    plt.legend()
    plt.show()

