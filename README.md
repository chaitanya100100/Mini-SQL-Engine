# Mini-SQL-Engine
A mini SQL engine to parse and execute simple sql queries implemented in python

## Description
- ![files][files/] directory contains the database and its schema.
- ![files/metadata.txt][metadata.txt] contains the schema of the database. Each table's schema is given between <begin_table> and <end_table>. The first line is the name of the table and others lines are field(column) names.
- Data stored in each table is given as csv file in ![files][files/] directory of the same name as of the table. For example, the data of table 'table1' is given as 'table1.csv'.
- Main source code is in ![files/mini_sql.py][files/mini_sql.py]
- Database contains only integer data.
- Queries are entirely case insensitive.
- Only simple queries can be performed. Nested queries are not allowed.
- Error handling is implemented with sufficient error debugging details.

### Run Query
Inside ![src][src/] directory, run `python mini_sql.py "<sql_query>"`

#### Valid queries
- Normal select queries
  - `select * from table1`
  - `select a, b from table1`
- Aggregate functions like `min`, `max`, `avg`, `sum`, `count`
  - `select max(a), min(b) from table1`
- Select distinct values of a column
  - `select distinct(a) from table1`
- Conditional select with at most two conditions joined by `and` or `or`
  - `select a from table1 where a=10`
  - `select a, b from table1 where a = c or b = 5`
- Table join and aliasing
  - `select * from table1, table2`
  - `select a, t1.b, c, t2.d from table1 as t1, table2 as t2 where t1.b = t2.b and t2.d >= 0`
