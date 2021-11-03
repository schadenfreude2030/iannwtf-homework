list1 = [i**2 for i in range(100)]
print(list1)
list2 = [square for square in [i**2 for i in range(100)] if square % 2 == 0]
print(list2)
