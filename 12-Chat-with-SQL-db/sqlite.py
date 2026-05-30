import sqlite3

# connect to sqlite

connection = sqlite3.connect("student.db")

## create a cursor object to insert and create table
cursor = connection.cursor()

## create table
table_info = """

create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25),
MARKS INT)

"""

cursor.execute(table_info) # table create



## insert data into table

cursor.execute("""Insert Into STUDENT values('rachin','Data Science', 'A', 97)""")
cursor.execute("""Insert Into STUDENT values('Jhon','AI','D','98')""")
cursor.execute("""Insert Into STUDENT values('vector','Block chain', 'A', 87)""")
cursor.execute("""Insert Into STUDENT values('emila','AI','A','93')""")
cursor.execute("""Insert Into STUDENT values('nelson','AI','B','100')""")

## display all the data
print("The inserted records are: ")
data =  cursor.execute("""Select * from STUDENT""")
for row in data:
    print(row)

# commit your changes in the database
connection.commit()
connection.close()

# python3 Chat-with-SQL-db/sqlite.py