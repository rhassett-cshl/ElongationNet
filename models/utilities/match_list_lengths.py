def match_list_lengths(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    
    # If the first list is shorter, extend it
    if len1 < len2:
        list1.extend([list1[-1]] * (len2 - len1))
    # If the second list is shorter, extend it
    elif len2 < len1:
        list2.extend([list2[-1]] * (len1 - len2))
    
    return list1, list2