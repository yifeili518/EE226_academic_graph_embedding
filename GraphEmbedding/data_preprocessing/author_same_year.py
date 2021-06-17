import csv
from os import write
import itertools

w_filename = 'author_same_year.csv'
r_filename = 'EE226-2021spring-problem1/author_paper_all_with_year.csv'

with open(w_filename, 'w', newline='') as wf:
    writer = csv.writer(wf)
    writer.writerow(['authorid','authorid'])
    years = [[] for i in range(5)] #2016-2020
    with open(r_filename) as f:
        reader = csv.reader(f)
        head = next(reader)
        data = list(reader)
        for k,row in enumerate(data):
            print(k)
            if row[2] == '2016':
                if row[0] not in years[0]:
                    years[0].append(row[0])
            if row[2] == '2017':
                if row[0] not in years[1]:
                    years[1].append(row[0])
            if row[2] == '2018':
                if row[0] not in years[2]:
                    years[2].append(row[0])
            if row[2] == '2019':
                if row[0] not in years[3]:
                    years[3].append(row[0])
            if row[2] == '2020':
                if row[0] not in years[4]:
                    years[4].append(row[0])
        
        for year in years:
            print(years.index(year))
            print(len(year))
            tmp = list(itertools.permutations(year, 2))
            author_same_year = []
            for i in tmp:
                author_same_year.append(list(i))
            for i in author_same_year:
                writer.writerow(i)
            
