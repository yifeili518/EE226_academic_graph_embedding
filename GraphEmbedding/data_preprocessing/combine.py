def getSubLists(lis=[],m=0):
    allAns = []                    #用来存储所有递归中的子列表
    ans = [None for i in range(m)] #预先填充m个None,用来存储每次递归中的子列表    
    subLists(lis,m,ans,allAns)
    return allAns
def subLists(lis=[],m=0,ans=[],allAns=[]):
    # recursive function  codes
    if m==0:
        # m==0是某次递归返回的条件之一：子列表的第三个数已经选出。
        # 意味着已到达某个方向的最大递归深度
        #print('allAns is ',allAns,'before append ans:',ans)
        allAns.append(ans.copy()) 
        #这里有意思了，如果不用copy,那么ans即使已经存储在allAns，也会被其它递归方向中的ans刷新
        #print('allAns is ', allAns, 'after append ans:', ans)
        return
    if len(lis)<m:
        # 递归函数直接返回的条件之一：从4个数里面选5个数出来是不可能的。
        print("short list!") 
        return
    length=len(lis)
    for iter in range(length-m+1):  #可以作为子列表一员的数在lis中的index
        ans[-m]=lis[iter]           #lis[iter]作为子列表倒数第m个数
        if iter+1<length:           #可以调用递归函数的条件：保证lis[iter+1:]里面还有东东才行
            subLists(lis[iter+1:],m-1,ans,allAns)
        else:
            #print('allAns is ', allAns, 'before append ans:', ans)
            allAns.append(ans.copy())
            #print('allAns is ', allAns, 'after append ans:', ans)
            return
