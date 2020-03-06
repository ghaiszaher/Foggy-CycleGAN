def print_with_timestamp(*args, **kwargs):
    from datetime import datetime
    t = "[" + str(datetime.now()).split(".")[0] + "]"
    print(t, *args, **kwargs)
