from writesql.utils import as_template_function


@as_template_function
def find_duplicates(table_name, col, min_count=1):
    ...


if __name__ == '__main__':
    print(find_duplicates(table_name='table_name', col='cert_no', min_count=5))
