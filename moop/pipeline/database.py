import logging
import sqlite3 as lite
from typing import Union


class Database:

    PATH2DB = "opti.db"

    @staticmethod
    def _checkIfTableExists(tablename: str) -> bool:
        """Checks if table exists

        Args:
            tablename (str): Name of the table

        Returns:
            bool: if table exists = True
        """

        con = lite.connect(Database.PATH2DB)

        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    f"SELECT count(name) FROM sqlite_master \
                    WHERE type='table' AND name='{tablename}'"
                )
                if cur.fetchone()[0] == 1:
                    return True
        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))

        return False

    @staticmethod
    def delete_table(tablename: str) -> bool:
        """Delete table

        Args:
            tablename (str): Name of the table

        Returns:
            bool: Description
        """
        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                # From the connection, we get the cursor object. The cursor is used
                # to traverse the records from the result set. We call the execute()
                # method of the cursor and execute the SQL statement.
                cur = con.cursor()
                cur.execute("DROP TABLE IF EXISTS {}".format(tablename))
                logging.info("Deleting table {}.".format(tablename))
                return True
        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))
        return False

    @staticmethod
    def create_table(tablename: str, columnsdict: dict) -> bool:
        """Summary

        Args:
            tablename (str): Name of the table
            columnsdict (dict): Description

        Returns:
            bool: Description
        """

        columns = ["{} {}".format(key, columnsdict[key]) for key in columnsdict.keys()]

        if not Database._checkIfTableExists(tablename):
            con = lite.connect(Database.PATH2DB)
            try:
                with con:
                    # From the connection, we get the cursor object.
                    # The cursor is used to traverse the records from the
                    # result set. We call the execute() method of the cursor
                    # and execute the SQL statement.
                    cur = con.cursor()
                    cur.execute(
                        "CREATE TABLE {}({})".format(tablename, ", ".join(columns))
                    )
                    logging.info("New table {} created sucessfully.".format(tablename))
                    return True
            except lite.Error as e:
                logging.warning("{}".format(e))
            except Exception as e:
                logging.warning("{}".format(e))
        else:
            logging.info(f"Table {tablename} already exists. Reusing...")
        return False

    @staticmethod
    def insertMany(tablename: str, rows: list, columnNames: list = None) -> bool:
        """Summary

        Args:
            tablename (str): Name of the table
            rows (list): Description
            columnNames (list, optional): Description

        Returns:
            param: Description
        """

        if len(rows) == 0:
            logging.info("Nothing to insert!")
            return

        placeholder = ", ".join(len(rows[0]) * ["?"])
        con = lite.connect(Database.PATH2DB)

        try:
            with con:
                cur = con.cursor()
                if isinstance(columnNames, type(None)):
                    cur.executemany(
                        "INSERT INTO {} VALUES({})".format(tablename, placeholder), rows
                    )
                elif isinstance(columnNames, list):
                    cur.executemany(
                        "INSERT INTO {}({}) VALUES({})".format(
                            tablename, ", ".join(columnNames), placeholder
                        ),
                        rows,
                    )
                return True
        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))
        return False

    @staticmethod
    def insertManyInChunks(
        tablename: str, rows: list, columnNames: dict, chunksize: int = 200
    ) -> bool:
        """Insert many items in chunks

        Args:
            tablename (str): Description
            rows (list): Description
            columnNames (dict): Description
            chunksize (int, optional): Description

        Returns:
            param: Description
        """
        if len(rows[0]) < 999:
            logging.info("This is unnecessary! Use insertMany instead")
            return Database.insertMany(tablename, rows, columns)

        for start in range(0, len(rows[0]), chunksize):
            end = (
                start + chunksize if start + chunksize < len(rows[0]) else len(rows[0])
            )
            subrows = [x[start:end] for x in rows]
            subcolumns = columnNames[start:end]
            Database.insertMany(tablename, subrows, subcolumns)
        return True

    @staticmethod
    def insert(tablename: str, rows: list) -> bool:
        """Summary

        Args:
            tablename (str): Description
            rows (list): Description

        Returns:
            bool: Description
        """
        if len(rows) == 0:
            logging.info("Nothing to insert!")
            return

        con = lite.connect(Database.PATH2DB)
        for row in rows:
            placeholders = ", ".join(["?" for key, val in row.items()])
            vals = [val for key, val in row.items()]
            keys = list(row.keys())

            try:
                with con:
                    cur = con.cursor()
                    cur.execute(
                        "INSERT INTO {}({}) VALUES({})".format(
                            tablename, ", ".join(keys), placeholders
                        ),
                        vals,
                    )
                    return True
            except lite.Error as e:
                logging.warning("{}".format(e))
            except Exception as e:
                logging.warning("{}".format(e))
            return False

    @staticmethod
    def update(tablename: str, row: list, query: dict) -> bool:
        """Summary

        Args:
            tablename (str): Description
            row (list): Description
            query (dict): Query {"row":["=", 4]}

        Returns:
            bool: Description
        """

        qkeys, qvals = Database._queryProcessor(query)

        vals, keys = [], []
        for key, val in row.items():
            keys.append("{} = ?".format(key))
            vals.append(val)

        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "UPDATE {} SET {} WHERE {}".format(
                        tablename, ", ".join(keys), " AND ".join(qkeys)
                    ),
                    vals + qvals,
                )
                return True
        except lite.Error as e:
            logging.warning("{}".format(e))

        except Exception as e:
            logging.warning("{}".format(e))

        return False

    @staticmethod
    def getColumnNames(tablename: str) -> Union[list, None]:
        """Return column names for table

        Args:
            tablename (str): Description

        Returns:
            None: Description
        """
        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                con.row_factory = lite.Row
                cur = con.cursor()
                cur.execute("SELECT * FROM {}".format(tablename))
                return list(cur.fetchone().keys())

        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))

    @staticmethod
    def _queryProcessor(query: dict) -> Union[list, list]:
        """Processes query

        Args:
            query (dict): query in form of {<columnName>:["=", <value>]}

        Returns:
            Union[list, list]: Description
        """
        vals, keys = [], []
        for key, val in query.items():
            keys.append("({} {} ?)".format(key, val[0]))
            vals.append(val[1])
        return keys, vals

    @staticmethod
    def find(
        tablename: str,
        variables: Union[list, None] = None,
        query: Union[dict, None] = None,
        one: bool = False,
        distinct: bool = False,
    ) -> Union[list, dict, None]:
        """Find by query in tablename

        Args:
            tablename (str): Description
            variables (Union[list, None], optional): Description
            query (Union[dict, None]): Description
            one (bool, optional): Description
            distinct (bool, optional): Description

        Returns:
            Union[list, dict, None]: Description
        """
        if query is None:
            pass
        else:
            keys, vals = Database._queryProcessor(query)

        if variables is None:
            variables = ["*"]

        elif not isinstance(variables, list):
            logging.warning("variables must be a list!")
            return
        if distinct:
            distinct = "DISTINCT"
        else:
            distinct = ""

        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                con.row_factory = lite.Row
                cur = con.cursor()
                if query is not None:
                    cur.execute(
                        "SELECT {} {} FROM {} WHERE {}".format(
                            distinct,
                            ", ".join(variables),
                            tablename,
                            " AND ".join(keys),
                        ),
                        vals,
                    )
                else:
                    cur.execute(
                        "SELECT {} {} FROM {}".format(
                            distinct, ", ".join(variables), tablename
                        )
                    )
                if one:
                    return dict(cur.fetchone())
                else:
                    return list(map(dict, cur.fetchall()))

        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))

    @staticmethod
    def remove(tablename: str, query: dict) -> bool:
        """Removes query from table

        Args:
            tablename (str): Description
            query (dict): Description

        Returns:
            bool: Description
        """
        keys, vals = Database._queryProcessor(query)

        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "DELETE FROM {} WHERE {}".format(tablename, " AND ".join(keys)),
                    vals,
                )
                return True
        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))
        return False

    @staticmethod
    def getMetaData(tablename: str) -> Union[dict, None]:
        """Get Meta Data

        Args:
            tablename (str): Description

        Returns:
            Union[dict, None]: Description
        """
        con = lite.connect(Database.PATH2DB)
        try:
            with con:
                cur = con.cursor()
                cur.execute("PRAGMA table_info({})".format(tablename))
                return cur.fetchall()

        except lite.Error as e:
            logging.warning("{}".format(e))
        except Exception as e:
            logging.warning("{}".format(e))


if __name__ == "__main__":

    columns = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT",
        "price": "INT",
    }

    Database.delete_table("cars")
    Database.create_table("cars", columns)

    Database.insert(
        "cars",
        [
            {"name": "Seat123", "price": 300},
            {"name": "VW", "price": 600},
            {"name": "Skoda", "price": 10000},
            {"name": "Porsche", "price": 600},
        ],
    )

#     #Database.remove("cars", query={"name": ["=", "Audi"],
#                                    "price": ["<=", 3000]})

#     rows = Database.find("cars", query={"name": ["=", "Audi"],
#                        "price": ["<=", 3000]}, one=True)
#     print(rows)

#     rows = Database.find("cars")

#     print(rows)
