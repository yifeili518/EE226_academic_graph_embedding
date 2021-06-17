import csv

r_filename = 'EE226-2021spring-problem1/labeled_papers_with_authors.csv'
w_filename = 'paper_label.txt'

with open(r_filename) as rf:
    reader = csv.reader(rf)
    head = next(reader)
    data = list(reader)
    paper_label = []
    start = '-1'
    for row in data:
        if row[1] != start:
            start = row[1]
            paper_label.append(row[3])
        else:
            continue
    
    with open(w_filename,'a+',encoding='utf-8') as wf:
        for i in range(len(paper_label)):
            s = str(i) + ' ' + paper_label[i] + '\n'
            wf.write(s)   
    