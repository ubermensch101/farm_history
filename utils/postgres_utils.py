import psycopg2

class PGConn:
    def __init__(self, config):
        self.details = config.setup_details["psql"]
        self.conn = None

    def connection(self):
        """Return connection to PostgreSQL.  It does not need to be closed
        explicitly.  See the destructor definition below.

        """
        if self.conn is None:
            conn = psycopg2.connect(dbname=self.details["database"],
                                    host=self.details["host"],
                                    port=self.details["port"],
                                    user=self.details["user"],
                                    password=self.details["password"])
            self.conn = conn
            self.conn.autocommit = True
            
        return self.conn

    def __del__(self):
        """No need to explicitly close the connection.  It will be closed when
        the PGConn object is garbage collected by Python runtime.
        """
        print(self.conn)
        if self.conn is not None:
            self.conn.close()
        self.conn = None
        
def check_schema_exists(psql_conn, schema_name):
    sql_query=f"""
        select 
            schema_name 
        from 
            information_schema.schemata 
        where 
            schema_name='{schema_name}'
    """
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql_query)
        schema_exists = curr.fetchone()
    if schema_exists is not None:
        return True
    else:
        return False

def create_schema(psql_conn, schema_name, delete_original = False):
    comment_schema_drop="--"
    if delete_original and check_schema_exists(psql_conn, schema_name):
            comment_schema_drop=""
        
    sql_query=f"""
        {comment_schema_drop} drop schema {schema_name} cascade;
        create schema if not exists {schema_name};
    """
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql_query)
        
def add_column(psql_conn, table, column, type):
    sql = f'''
        alter table {table}
        drop column if exists {column};
        alter table {table}
        add column {column} {type};
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
    
def check_column_exists(psql_conn, schema, table, column):
    sql = f'''
    SELECT EXISTS (SELECT 1 
    FROM information_schema.columns 
    WHERE 
        table_schema='{schema}' 
    AND 
        table_name='{table}' 
    AND 
        column_name='{column}');
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        a = curr.fetchall()
    return a[0][0]

def find_column_geom_type(psql_conn, schema, table, column):
    sql = f'''
        select 
            type
        from 
            geometry_columns 
        where
            f_table_schema = '{schema}' 
            and 
            f_table_name = '{table}' 
            and 
            f_geometry_column = '{column}';
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        res = curr.fetchall()
    return res[0][0]

def find_dtype(psql_conn, schema, table, column):
    sql = f'''
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND 
        table_name = '{table}' AND
        column_name = '{column}';
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        res = curr.fetchall()
    return res[0][0]


def find_srid(psql_conn, schema, table, column):
    sql = f'''
        select find_srid('{schema}','{table}','{column}')
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        res = curr.fetchall()
    return int(res[0][0])


def table_exist(psql_conn, schema, table):
    sql = f'''
        SELECT EXISTS (
            SELECT 
            FROM 
                information_schema.tables 
            WHERE  
                table_schema = '{schema}'
            AND    
                table_name   = '{table}'
        );
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        res = curr.fetchall()
    if res[0][0]:
        return True
    else:
        return False
    
def number_of_entries(psql_conn, schema, table):
    sql = f'''
    select 
        count(gid)
    from 
        {schema}.{table}
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        res = curr.fetchall()
    return int(res[0][0])

def copy_table(psql_conn, input, output):
    sql = f'''
        drop table if exists {output};
        create table {output} as table {input};
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        
def update_srid(psql_conn, input_table, column, srid):
    sql = f'''
        alter table {input_table}
        alter column {column} 
        type geometry(Geometry,{srid})
        using st_transform({column},{srid});
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        
def rename_column(psql_conn, input_schema, input_table_name, original, final):
    if not check_column_exists(psql_conn, input_schema,input_table_name,original):
        return
    
    sql = f'''
        alter table {input_schema}.{input_table_name}
        rename column {original} to {final};
    '''
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        
def add_gist_index(psql_conn, schema, table, column):
    sql = f"""
        create index if not exists
            {schema}_{table}_{column}_index 
        on 
            {schema}.{table}
        using 
            GIST({column});
    """
    with psql_conn.connection().cursor() as curr:
        curr.execute(sql)
        