#Written by Vandhana Visakamurthy for COMP9313
#z5222191
#Project 1
#Implementation of C2LSH Algorithm

def count_hashes(hashes_1,hashes_2,offset):
    counter = 0
    #iterating through both the lists simultaneously
    for i,j in zip(hashes_1,hashes_2):
        #check if the difference is lesser than equal to offset
        if(abs(i-j) <= offset):
            #increment the counter
            counter = counter + 1
    return counter

def splitting(x,query_hashes,offset):
    #obtain the keys using list comprehension
    key = x[0]
    #obtain the values using list comprehension
    value = x[1]
    #call the count_hashes function
    answer = (key,count_hashes(value,query_hashes,offset))
    return answer

def store_ids(x):
    #returns only candidate ids
    key = x[0]
    return key

########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    #set offset as 0
    offset = 0
    #candidate_set = data_hashes.filter(lambda x: x[0] == -1)
    #map the count hashes
    while True:
        #print(f'Offset: {offset}')
        #Find all collision counts and store as (candidate_id,count)
        first = data_hashes.map(lambda x: splitting(x,query_hashes,offset))
        #print(first.collect())
        #check if the sum/count is greater than or equal to alpha_m
        second = first.filter(lambda x: x[1] >= alpha_m)
        #print(second.collect())
        #append only the ids into candidate set
        third = second.map(store_ids)
        #print(third.collect())
        #candidate_set = third.union(third)
        #if the length of candidate set is less than beta_n increment offset
        if third.count() < beta_n:
            offset = offset + 1
        else:
            #break the while
            break
    #third contains the final candidate set
    return third