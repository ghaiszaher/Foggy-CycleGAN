def print_with_timestamp(*args, **kwargs):
    from datetime import datetime
    t = "[" + str(datetime.now()).split(".")[0] + "]"
    print(t, *args, **kwargs)


def create_dir(path):
    """
    Creates a path recursively if it doesn't exist
    :param path: The specified path
    :return: None
    """
    if path is None or path == '':
        return
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise Exception("Not a valid path: {}".format(path))


def df_length(df):
    """
    Returns a dataframe length.
    Reference: https://stackoverflow.com/a/15943975/11394663
    :param df: a Pandas Dataframe
    :return: int
    """
    return len(df.index)