import sqlite3

conn = sqlite3.connect('prompts.db')
c = conn.cursor()


class PromptDB:
    def __init__(self):
        self.conn = sqlite3.connect('prompts.db')
        self.curs = self.conn.cursor()
        self.__create_tables()

    def __create_tables(self):
        self.curs.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
            )
        
            CREATE TABLE IF NOT EXISTS prompts (
                FOREIGN KEY (id) REFERENCES categories(id),
                prompt TEXT,
            )
        """)
