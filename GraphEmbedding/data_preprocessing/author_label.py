import csv

r_filename = 'EE226-2021spring-problem1/labeled_papers_with_authors.csv'
w_filename = 'author_label.txt'

with open(r_filename) as rf:
    reader = csv.reader(rf)
    head = next(reader)
    data = list(reader)
    total = {}
    for row in data:
        print(row)
        if not row[0] in total.keys():
            total[row[0]] = [row[3]]
        else:
            if not row[3] in total[row[0]]:
                total[row[0]].append(row[3])
    
    with open(w_filename,'a+',encoding='utf-8') as wf:
        for i in total:
            s = str(i) + ' '
            for n,j in enumerate(total[i]):
                s = s + str(j)
                if n!= len(total[i]):
                    s = s + ' '
            s = s + '\n' 

            wf.write(s)   
    