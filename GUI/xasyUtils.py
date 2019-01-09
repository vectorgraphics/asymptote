#!/usr/bin/env python3

import re
import typing as ty
import math
import itertools

def tuple2StrWOspaces(val: tuple) -> str:
    newStr = ','.join(['{:.6g}'.format(value) for value in val])
    return '({0})'.format(newStr)

def tryParse(val, typ=float):
    try:
        return typ(val)
    except ValueError:
        return None

def funcOnList(list1: ty.Union[ty.List, ty.Tuple], list2: ty.Union[ty.List, ty.Tuple], func: ty.Callable) -> tuple:
    """Returns [f(x[i], y[i]) : i in 1, ..., n - 1] in order with f as func 
    and x and y as list1 and 2. """

    assert len(list1) == len(list2)
    return tuple([func(list1[i], list2[i]) for i in range(len(list1))]) 


def listize(str, typ, delim='()') -> list:
    str = str.strip(delim)
    raw_elem = str.split(',')
    final_list = []
    if isinstance(typ, (list, tuple)):
        for i in range(len(raw_elem)):
            if i < len(typ):
                curr_typ = typ[i]
            else:
                curr_typ = typ[-1]
            final_list.append(curr_typ(raw_elem[i].strip()))
    else:
        for elem in raw_elem:
            final_list.append(typ(elem.strip()))
    return final_list

def twonorm(vec: ty.Iterable[ty.Union[float, int]]) -> float:
    rawSquared = sum(map(lambda x: x*x, vec))
    return math.sqrt(rawSquared)

def tryParseKey(raw_key):
    """Returns None if raw key is not in #.# format"""
    # See https://regex101.com/r/6G9MZD/1/
    # for the regex data
    return re.fullmatch(r'^(\d+)\.(\d+)$', raw_key)
