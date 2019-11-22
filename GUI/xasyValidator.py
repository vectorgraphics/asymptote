#!/usr/bin/env python3

def validateFloat(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    assert validateFloat('0.5')
    assert not validateFloat('.-')
