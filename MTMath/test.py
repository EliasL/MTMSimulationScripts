from powerlaw import Power_Law

data = [1, 2, 3, 4, 5]
dist = Power_Law(data=data, xmin=1)
print(dist.parameter1_name)
