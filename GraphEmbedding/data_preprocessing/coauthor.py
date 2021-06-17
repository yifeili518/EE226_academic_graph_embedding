import csv
from combine import getSubLists


w_filename = 'coauthor.csv'
r_filename = 'EE226-2021spring-problem1/author_paper_all_with_year.csv'

total = []
with open(w_filename,'w', newline='') as wf:
    writer = csv.writer(wf)
    writer.writerow(['authorid','authorid'])
    with open(r_filename) as f:
        reader = csv.reader(f)
        head = next(reader)
        tmp = '0'
        coauthor = []
        data = list(reader)
        n = len(data)

        for count, row in enumerate(data):
            print(count)
            if row[1] == tmp:
                coauthor.append(row[0])
                if count == n: 
                    author_pair = getSubLists(coauthor,2)
                    for i in author_pair:
                        if i not in total:
                            total.append(i)#因为需要构建有向图且为合作关系 因此双向都要添加
            else:
                if len(coauthor) != 1: #作者与其自身不成环
                    author_pair = getSubLists(coauthor,2)#找出合作对
                    for i in author_pair:
                        if i not in total:
                            total.append(i)#因为需要构建有向图且为合作关系 因此双向都要添加    
                coauthor.clear()
                coauthor.append(row[0])
                tmp = row[1]

    for i in total:
        writer.writerow(i)



