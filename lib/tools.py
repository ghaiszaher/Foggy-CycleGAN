def print_with_timestamp(*args, **kwargs):
    from datetime import datetime
    t = "[" + str(datetime.now()).split(".")[0] + "]"
    print(t, *args, **kwargs)


def create_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise Exception("Not a valid path: {}".format(path))