def today(fmt='%Y%m%d'):
    """ Return the date of today as a string."""
    from datetime import datetime
    dt = datetime.today()
    return dt.strftime(fmt)