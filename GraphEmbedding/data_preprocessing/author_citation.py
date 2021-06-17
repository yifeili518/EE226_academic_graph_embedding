'''
本代码为通过paper reference得到citation relation between authors
的数据预处理代码
'''

import csv

w_filename = 'author_citation_shuangxiang.csv'
r_filename1 = 'EE226-2021spring-problem1/author_paper_all_with_year.csv'
r_filename2 =  'EE226-2021spring-problem1/paper_reference.csv'

'''
创建author_citation.csv文件
csv格式为两列authorid 表示作者与作者之间的引用关系
同一行中前面的作者指向后面的作者 代表前面的作者引用了后面的作者（待商榷
'''



with open(w_filename,'w',newline='') as wf:
    writer = csv.writer(wf)
    writer.writerow(['authorid','authorid'])

    tmp = [] #将author_paper_all.csv中的数据整理成[['0'],['1','2'],....]的形式，
    #list下标和paperid对应，每个list元素代表这个paper中的所有作者

    with open(r_filename1) as rf1:
        reader1 = csv.reader(rf1)
        head1 = next(reader1)
        flag  = '0'
        data = list(reader1)
        coauthor = []
        count = 0
        n = len(data)
        for row in data:
            count += 1
            if row[1] == flag:
                #paper为当前flag 则为coauthor
                coauthor.append(row[0])
                if count == n: 
                    tmp_co = coauthor[:]
                    tmp.append(tmp_co)
            
            else:
                #paper非当前flag 说明检索到了下一个paper
                #将前一paper的所有作者放入tmp中 清空当前coauthor 
                #添加当前paper的第一个作者 并将flag改为当前paper
                tmp_co = coauthor[:] #这里必须复制一份 因为之后要进行clear操作
                tmp.append(tmp_co)
                coauthor.clear()
                coauthor.append(row[0])
                flag = row[1]
    #print(tmp)
    
    total = []
    with open(r_filename2) as rf2:
        reader2 = csv.reader(rf2)
        head2 = next(reader2)
        data = list(reader2)
        for count, row in enumerate(data):
            print(count)
            author_from = tmp[int(row[0])]
            author_to = tmp[int(row[1])]
            for i in author_from:
                for j in author_to:
                    if i == j:
                        continue # 排除自环的情况
                    x = [i , j]
                    y = [j , i]
                    if x in total:
                        continue # 排除重复边
                    else:
                        total.append(x)
                        total.append(y)
    
    for i in total:
        writer.writerow(i)


