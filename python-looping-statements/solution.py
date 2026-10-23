def countdown_with_skip(start: int) -> list:
    """
    Using a while loop, count down from `start` to 1 (inclusive),
    appending each number to a list, but skip appending any
    number that is exactly 3 (use continue for this).
    Do not use break here.
    Return the resulting list.
    """
    arr=[]
    while(start>=1):
        if(start==3):
            start-=1
            continue
        arr.append(start)
        start=start-1
    return arr


def find_first_negative(numbers: list) -> int | None:
    """
    Using a while loop with an index variable, find and return
    the first negative number in `numbers`. Use break once found.
    If no negative number exists, use the loop's else clause to
    return None after the loop completes normally.
    """
    i=0
    while(i<len(numbers)):
        if(numbers[i]<0):
            return numbers[i]
            break
        i+=1
    else:
        return None


def sum_with_index(numbers: list) -> dict:
    """
    Using a for loop with enumerate(), build and return a
    dictionary mapping each index to the running sum of all
    numbers up to and including that index.

    Example: sum_with_index([10, 20, 30])
    -> {0: 10, 1: 30, 2: 60}
    """
    data = {}
    total = 0

    for ind, num in enumerate(numbers):
        total += num
        data[ind] = total

    return data


def manual_iteration_trace(items: list) -> list:
    """
    Without using a for loop, replicate what a for loop does
    internally: use iter() to get an iterator from `items`, then
    repeatedly call next() on it inside a while loop, catching
    StopIteration to know when to stop.

    Return a list of all elements retrieved this way, in order
    (it should exactly match the original `items` list).
    """
    result = []
    iterator = iter(items)

    while True:
        try:
            item = next(iterator)
            result.append(item)
        except StopIteration:
            break

    return result
    
    pass
