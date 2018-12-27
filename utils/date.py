def today(fmt='%Y%m%d'):
    from datetime import datetime
    dt = datetime.today()
    return dt.strftime(fmt)