def unique(elements):
    for i in range(0, len(elements)):
        for j in range(i + 1, len(elements)):
            if elements[i] == elements[j]:
                return False
    return True


def prod(nums):
    result = 1
    for num in nums:
        result *= num
    return result
