def modify_list(data):
    """Modifies a list in-place and returns a modified copy (has bugs).

    Args:
        data: A list of numbers.

    Returns:
        A modified copy of the list (intended behavior, but buggy).
    """
    # This function modifies the original list and also tries to return a copy
    # This will result in unexpected behavior
    for i in range(len(data)):
        data[i] *= 2  # Modifying the original list

    # Creates a new empty list (unnecessary since list is modified in-place)
    new_data = []
    for item in data:
        new_data.append(item)  # Appending to a new list (redundant)

    return new_data


# Example usage (unexpected output)
my_list = [1, 2, 3]
modified_list = modify_list(my_list)

print(f"Original list: {my_list}")  # Modified due to in-place change
print(f"Modified list: {modified_list}")  # Empty list due to redundancy
