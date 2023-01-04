def product(lst):
    result = 1
    for num in lst:
        result *= num
    return result


def split_list(lst, k):
    # Initialize a list of sublists
    sublists = []
    while len(lst) > 0:
        sublist = []
        for i in range(k):
            if len(lst) > 0:
                sublist.append(lst.pop(0))
        sublists.append(sublist)

    for sublist in sublists:
        print("Elements:", sublist)
        print("Product:", product(sublist))

    return sublists


quotes = sorted([1.15, 1.2, 1.4, 1.3, 1.22, 1.2, 1.35, 1.3, 1.28])
k = 3
sublists = split_list(quotes, k)
print(sublists)







