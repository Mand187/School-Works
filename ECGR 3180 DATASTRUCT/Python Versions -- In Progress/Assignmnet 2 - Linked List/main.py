from LinkedList import LinkedList

def merge_lists(list1, list2):
    merged_list = LinkedList()  # Instantiate LinkedList

    i = 0  # Index for list1
    j = 0  # Index for list2

    # Merge while both lists have elements
    while i < list1.size() and j < list2.size():
        x = list1.at(i)  # Access element using `at` method
        y = list2.at(j)

        if x <= y:
            merged_list.push_back(x)  # Add element to merged_list
            i += 1
        else:
            merged_list.push_back(y)
            j += 1

    # If there are remaining elements in list1, append them
    while i < list1.size():
        merged_list.push_back(list1.at(i))
        i += 1

    # If there are remaining elements in list2, append them
    while j < list2.size():
        merged_list.push_back(list2.at(j))
        j += 1

    # Print merged list
    print("\nMerged List")
    merged_elements = [merged_list.at(k) for k in range(merged_list.size())]
    print("[" + ",".join(map(str, merged_elements)) + "]")

    return merged_list


def main():
    # Create the first list
    my_list1 = LinkedList()
    my_list2 = LinkedList()

    # Add elements to my_list1
    my_list1.push_back(2)
    my_list1.push_back(3)
    my_list1.push_back(4)

    print("\nList 1")
    list1_elements = [my_list1.at(k) for k in range(my_list1.size())]
    print("[" + ",".join(map(str, list1_elements)) + "]")

    # Add elements to my_list2
    my_list2.push_back(1)
    my_list2.push_back(3)
    my_list2.push_back(4)

    print("\nList 2")
    list2_elements = [my_list2.at(k) for k in range(my_list2.size())]
    print("[" + ",".join(map(str, list2_elements)) + "]")

    # Merge the two lists
    merge_lists(my_list1, my_list2)


if __name__ == "__main__":
    main()
