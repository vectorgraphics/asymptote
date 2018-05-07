def tryParse(val, typ=float):
    try:
        return typ(val)
    except ValueError:
        return None
