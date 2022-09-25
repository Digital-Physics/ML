from __future__ import annotations
# we want to use | instead of Union for "type hints"


def get_statistics(l: list[float | int]) -> dict:
    """this function returns some statistics on a list of numbers"""
    stat_dict = {}

    l.sort()

    stat_dict["mean"] = sum(l)/len(l)

    if len(l) % 2 == 1:
        stat_dict["median"] = l[len(l)//2]
    else:
        stat_dict["median"] = (l[len(l)//2] + l[len(l)//2 - 1])/2

    counts_dict = {x: l.count(x) for x in set(l)}
    most = 0
    for x in set(l):
        if counts_dict[x] > most:
            most = counts_dict[x]
            stat_dict["mode"] = x

    # an unbiased estimate of the population variance divides the sample variance by n-1, not n
    stat_dict["sample_variance"] = sum([(x - stat_dict["mean"])**2 for x in l])/(len(l) - 1)
    stat_dict["sample_standard_deviation"] = stat_dict["sample_variance"]**(1/2)
    # confidence interval for mean, not the confidence interval for a single sample
    standard_error = stat_dict["sample_standard_deviation"] / (len(l)**(1/2))
    # 95% confidence interval
    stat_dict["mean_confidence_interval"] = [stat_dict["mean"] - 1.96 * standard_error,
                                             stat_dict["mean"] + 1.96 * standard_error]

    return stat_dict


test_list = [2, 1, 3, 4, 4, 5, 6, 7]
dict_result = get_statistics(test_list)
print(dict_result)
