from writesql.utils import as_template_function


@as_template_function
def find_duplicates(table_name, col, min_count=1):
    ...


print(find_duplicates('table_name', 'cert_no', 5))
