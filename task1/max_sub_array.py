def max_subsum_arr(arr):
    """O(n) algorithm of finding continuous subarray of integers with maximum sum"""
    maxsum, pos_cumsum, start_index, end_index = 0, 0, 0, len(arr)
    # maxsum is sum of required subarray, pos_cumsum is tmp candidate for maxsum
    # pos_cumsum increases cumulatively in array, but avoids being negative in non-negative array 
    # indexes tracks stard and end positions of subarray
    hypo_start_index = 0  # hypo_start_index is updated as cumsum < 0
    # start_index must be updated only if it_is hypo_start_index and maxsum changed
    # otherwise maxsum changed, hypo_start_index is not relevant

    for i, x in enumerate(arr):
        pos_cumsum += x
        if max(maxsum, pos_cumsum) > maxsum:
            maxsum, end_index = max(maxsum, pos_cumsum), i + 1
            start_index = hypo_start_index
        # update end_index
        # update max_sum if tmp cumsum is greater than maxsum
        # if further summation decreases cumsum, end_index will remain
        if pos_cumsum < 0:
            pos_cumsum, hypo_start_index = 0, i + 1
            # we terminate negative tmp cumsum, as keeping cumsum < 0
            # will lead to less maxsum if it goes above zero
            # start_index can't be less than at this point, consequently, update it
        if maxsum == 0 and pos_cumsum == 0 and arr[-1] < 0:  # last element < 0 is needed to avoid zeros case
            # and to track all neg case
            # by the way, take the last demand is explained by [neg, pos] array example 
            return [max(arr)]
        # in case of negatives, we return a single max of negatives
        if pos_cumsum + x == pos_cumsum:  # case of zero at the end of subset
            end_index += 1

    return arr[start_index : end_index]


'''TESTS'''
assert max_subsum_arr([-2, -1, -6, -9]) == [-1]
assert max_subsum_arr([3, 5, -10, -4, 1]) == [3, 5]
assert max_subsum_arr([3, -7, 5, -4, 3]) == [5]
assert max_subsum_arr([20, 0, 5, -9, -10, 3]) == [20, 0, 5]
assert max_subsum_arr([0, 0, 0]) == [0, 0, 0]
assert max_subsum_arr([3, 2, -6, 4, 8]) == [4, 8]
assert max_subsum_arr([3, 2, -5, 4, 8]) == [3, 2, -5, 4, 8]
assert max_subsum_arr([-4, -2, -1, 3, 2, 1]) == [3, 2, 1]
assert max_subsum_arr([-4, -2, -1, 3, 2, 1, 0]) == [3, 2, 1, 0]
assert max_subsum_arr([-4, -2, 1, 3, -1, -2, 4, 3]) == [1, 3, -1, -2, 4, 3]



if __name__ == '__main__':
    array =  [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f'found max_subsum of array {array}')
    print(max_subsum_arr(array))
