import pandas as p

# read
r=p.read_csv("r.csv")

t1=p.read_csv("t1.csv") # t1(i,A,K)
t2=p.read_csv("t2.csv") # t2(K,B)

#write
t_ka=t1[['K','A']]
t_ka.to_csv('t_ka.csv')

# SELECT * FROM t1 WHERE K=10;
t1.loc[t1['K']==10 ]
# SELECT * FROM t1 WHERE K=10 AND A=1;
# fails
t1.loc[t1['K']==10 & t1['A']==1]
# works
t1.loc[(t1['K']==10) & (t1['A']==1)]

# join
# SELECT * FROM t1 JOIN t2
# OO, not infix
# guessed from schemas
TJ=t1.merge(t2)
# readundant K spec
TJ=t1.merge(t2,on='K')

# pure pi
# SELECT A,B FROM t1
# subscript based
t1[0:2]
#
t1['K']
# fails
t1['K','A']
# works
t1[['K','A']]

# returned in index
t_ka.groupby(by=['K','A']).count()


#pi GROUP BY
# SELECT K,sum(A) FROM t1 GROUP BY K
t1.groupby(by='K').sum()
#sorts
t1.groupby(by='i').sum()

#join+pi
# wrong: just obj ref
tjp=t1.merge(t2).groupby(by='B')
# correct
tjp=t1.merge(t2).groupby(by='B').sum()

tjp.to_csv("tjp_out.csv")

