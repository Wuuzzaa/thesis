import itertools


def product(lst):
    result = 1
    for num in lst:
        result *= num
    return result


def optimize_sublists(sublists):
    best_difference = float("inf")
    best_sublists = None
    best_elements = None
    for sublist1, sublist2 in itertools.combinations(sublists, 2):
        for element1, element2 in itertools.product(sublist1, sublist2):
            temp_sublist1 = sublist1.copy()
            temp_sublist2 = sublist2.copy()
            temp_sublist1.remove(element1)
            temp_sublist1.append(element2)
            temp_sublist2.remove(element2)
            temp_sublist2.append(element1)
            difference = abs(product(temp_sublist1) - product(temp_sublist2))
            if difference < best_difference:
                best_difference = difference
                best_sublists = (temp_sublist1, temp_sublist2)
                best_elements = (element1, element2)
            if difference == 0:
                return best_sublists
    return best_sublists, best_elements


def split_list(lst, k):
    # Initialize a list of sublists
    sublists = []
    while len(lst) > 0:
        sublist = []
        for i in range(k):
            if len(lst) > 0:
                sublist.append(lst.pop(0))
        sublists.append(sublist)

    print("Sublists before optimization:")
    for i, sublist in enumerate(sublists):
        print(f"Sublist {i+1}: {sublist} (Product: {product(sublist)})")

    optimized_sublists, optimized_elements = optimize_sublists(sublists)
    if optimized_sublists:
        sublists = optimized_sublists
        print("Elements swapped:", optimized_elements)

    print("Sublists after optimization:")
    for i, sublist in enumerate(sublists):
        print(f"Sublist {i+1}: {sublist} (Product: {product(sublist)})")

    return sublists

quotes = sorted([1.15, 1.2, 1.4, 1.3, 1.22, 1.2, 1.35, 1.3, 1.28])
k = 3
sublists = split_list(quotes, k)
#print(sublists)







