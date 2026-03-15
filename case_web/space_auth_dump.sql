PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT
            );
INSERT INTO users VALUES(1,'admin','scrypt:32768:8:1$EmUe6lrkMArVmsIL$ec8372405bd2605f38cb98a3e7d7b6f03c3a19d9ae62b76752d8172dfccbe6d12beb80bcd25ad882297af4c5158f6e50c9a566267315da88598bdb8e5fef8cac','admin','System','Admin');
INSERT INTO users VALUES(2,'tayka_jpg','scrypt:32768:8:1$mO0UqiYgBi9OT1a9$f33539e180549670deaa20466f2eeebe33874e68c35eaee435f07a08f7e3bd13a64974b3ba72f4d639ac78345dc574e4d09a725b27c20d944e4215cae9badf70','user','тайка','попка');
INSERT INTO sqlite_sequence VALUES('users',2);
COMMIT;
