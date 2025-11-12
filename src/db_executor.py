import duckdb

class DBExecutor:    
    def __init__(self, sql_script_path: str):
        self.sql_script_path = sql_script_path
        self.con = duckdb.connect(database=':memory:', read_only=False)

    def run(self, params: dict):
        """
        SQL зарос.
        Args: params: словарь с параметрами для подстановки в SQL
        """
        print(f"SQL: {self.sql_script_path}")
        with open(self.sql_script_path, 'r') as f: query = f.read()
        
        for key, value in params.items(): query = query.replace(f"{{{{{key}}}}}", str(value))  
        self.con.execute(query)
        print("Готово")

if __name__ == '__main__':
    executor = DBExecutor(sql_script_path='sql/process_data.sql')
    executor.run(params={'data_path': '../data', 'output_path': '../data/processed_from_sql.csv'})
    print("Результат: ../data/processed_from_sql.csv")
