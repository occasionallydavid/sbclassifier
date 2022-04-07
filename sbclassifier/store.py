
class BaseStore:
    def get_nspam_nham(self, tokens):
        raise NotImplementedError()

    def get_token_counts(self, tokens):
        raise NotImplementedError()

    def add_spam(self, tokens):
        raise NotImplementedError()

    def add_ham(self, tokens):
        raise NotImplementedError()

    def remove_spam(self, tokens):
        raise NotImplementedError()

    def remove_ham(self, tokens):
        raise NotImplementedError()


class HeapStore(BaseStore):
    def __init__(self):
        self.spam = {}
        self.ham = {}
        self.nspam = 0
        self.nham = 0

    def _increment(self, dct, keys):
        for key in keys:
            dct[key] = dct.get(key, 0) + 1

    def _decrement(self, dct, keys):
        for key in keys:
            if key in dct:
                dct[key] = dct.get(key, 0) - 1

    def get_nspam_nham(self):
        return self.nspam, self.nham

    def get_token_counts(self, tokens):
        return {
            w: (
                self.spam.get(w, 0),
                self.ham.get(w, 0)
            )
            for w in tokens
            if w in self.spam or w in self.ham
        }

    def add_spam(self, tokens):
        self._increment(self.spam, tokens)
        self.nspam += 1

    def add_ham(self, tokens):
        self._increment(self.ham, tokens)
        self.nham += 1

    def remove_spam(self, tokens):
        self._decrement(self.spam, tokens)
        self.nspam -= 1

    def remove_ham(self, tokens):
        self._decrement(self.ham, tokens)
        self.nham -= 1


class SqliteStore(BaseStore):
    DOC_COUNT_TOKEN = ' $DOC_COUNT$ '

    def __init__(self, db):
        self.db = db
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS '
            'tokens(token PRIMARY KEY, spam, ham) '
            'WITHOUT ROWID'
        )

    def _upsert(self, tokens, spam, ham):
        self.db.executemany(
            f"INSERT INTO tokens(token, spam, ham) "
            f"VALUES (?, {spam}, {ham}) "
            f"ON CONFLICT(token) DO UPDATE "
            f"SET "
                f"spam = spam + {spam},"
                f"ham = ham + {ham}",
            ((t,) for t in tokens)
        )

    def _update(self, tokens, spam, ham):
        self.db.executemany(
            f"UPDATE tokens SET "
                f"spam = spam + {spam} "
                f"ham = ham + {ham} "
            f"WHERE token = ?",
            keys
        )

    def get_nspam_nham(self):
        counts = self.get_token_counts((self.DOC_COUNT_TOKEN,))
        dc = counts.get(self.DOC_COUNT_TOKEN)
        if dc:
            return dc
        return 0, 0

    def get_token_counts(self, tokens):
        return {
            w: row
            for w in tokens
            for row in self.db.execute(
                "SELECT max(0, spam), max(0, ham)"
                "FROM tokens "
                "WHERE token = ?",
                (w,)
            )
        }

    def add_spam(self, tokens):
        self._upsert(tokens, 1, 0)
        self._upsert([self.DOC_COUNT_TOKEN], 1, 0)

    def add_ham(self, tokens):
        self._upsert(tokens, 0, 1)
        self._upsert([self.DOC_COUNT_TOKEN], 0, 1)

    def remove_spam(self, tokens):
        self._update(tokens, -1, 0)
        self._upsert([self.DOC_COUNT_TOKEN], -1, 0)

    def remove_ham(self, tokens):
        self._update(tokens, 0, -1)
        self._upsert([self.DOC_COUNT_TOKEN], 0, -1)
