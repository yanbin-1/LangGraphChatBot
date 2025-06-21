import psycopg2

# 连接到PostgreSQL数据库
conn = psycopg2.connect(
    dbname='chatbot',
    user='postgres',
    password='postgres',
    host='localhost',  # 或者您的服务器IP地址
    port='5432'  # 或者您的数据库端口号
)

# 创建游标对象
cur = conn.cursor()

# 执行SQL查询
cur.execute('SELECT * FROM your_table')

# 获取查询结果
result = cur.fetchall()

# 打印结果
for row in result:
    print(row)

# 关闭游标和连接对象
cur.close()
conn.close()
ls -l /usr/share/postgresql/15/extension/vector*