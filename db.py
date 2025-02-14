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
                name TEXT UNIQUE
            )
        """)

        self.curs.execute("""
            CREATE TABLE IF NOT EXISTS unlabeled_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT UNIQUE
            )
        """)

        self.curs.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                fid INTEGER REFERENCES categories(id),
                prompt TEXT UNIQUE NOT NULL CHECK(prompt <> '')
            );
        """)

    def get_entities(self):
        self.curs.execute("SELECT name FROM categories")
        list_of_entities = self.curs.fetchall()
        return [entity for entities in list_of_entities for entity in entities]

    def add_entity(self, entity):
        self.curs.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (entity,))
        self.conn.commit()

    def add_prompt(self, entity, prompt):
        self.curs.execute("SELECT id FROM categories WHERE name=?", (entity,))
        entity_name = self.curs.fetchone()
        self.curs.execute("INSERT OR IGNORE INTO prompts VALUES (?, ?)", (entity_name[0], prompt,))
        self.conn.commit()

    def add_unlabeled_prompt(self, prompt):
        self.curs.execute("INSERT OR IGNORE INTO unlabeled_prompts (prompt) VALUES (?)", (prompt,))
        self.conn.commit()

    def manual_label_prompt(self, entity, prompt_id):
        self.curs.execute("SELECT id FROM categories WHERE name=?", (entity,))
        entity_name = self.curs.fetchone()
        self.curs.execute("SELECT id FROM unlabeled_prompts WHERE id=?", (prompt_id,))
        prompt = self.curs.fetchone()
        self.curs.execute("INSERT INTO prompts VALUES (?,?)", (entity_name[0], prompt[0],))
        self.conn.commit()

    def close(self):
        self.conn.close()
