def today(fmt='%Y%m%d'):
    """ Return the date of today as a string."""
    from datetime import datetime
    dt = datetime.today()
    return dt.strftime(fmt)


def iter_date(start, end, step=1, fmt=None, include_end=True):
    """ Iterate over days
    :param fmt: If set to None, yield datetime object, elsewise return a formatted string
    """
    from dateutil.relativedelta import relativedelta
    from dateutil import parser

    if isinstance(start, str):
        start = parser.parse(start)
    if isinstance(end, str):
        end = parser.parse(end)

    step = relativedelta(days=step)
    cursor = start
    while cursor <= end:
        yield cursor.strftime(fmt) if fmt else cursor
        cursor += step
    if include_end and cursor != end:
        yield end.strftime(fmt) if fmt else cursor