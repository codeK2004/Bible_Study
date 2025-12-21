import re

BOOK_NAMES = [
    "GENESIS","EXODUS","LEVITICUS","NUMBERS","DEUTERONOMY",
    "JOSHUA","JUDGES","RUTH","1 SAMUEL","2 SAMUEL",
    "1 KINGS","2 KINGS","1 CHRONICLES","2 CHRONICLES",
    "EZRA","NEHEMIAH","ESTHER","JOB","PSALMS","PROVERBS",
    "ECCLESIASTES","SONG OF SOLOMON","ISAIAH","JEREMIAH",
    "LAMENTATIONS","EZEKIEL","DANIEL","HOSEA","JOEL","AMOS",
    "OBADIAH","JONAH","MICAH","NAHUM","HABAKKUK","ZEPHANIAH",
    "HAGGAI","ZECHARIAH","MALACHI","MATTHEW","MARK","LUKE",
    "JOHN","ACTS","ROMANS","1 CORINTHIANS","2 CORINTHIANS",
    "GALATIANS","EPHESIANS","PHILIPPIANS","COLOSSIANS",
    "1 THESSALONIANS","2 THESSALONIANS","1 TIMOTHY",
    "2 TIMOTHY","TITUS","PHILEMON","HEBREWS","JAMES",
    "1 PETER","2 PETER","1 JOHN","2 JOHN","3 JOHN",
    "JUDE","REVELATION"
]

BOOK_SET = set(BOOK_NAMES)
VERSE_RE = re.compile(r"\{?(\d+):(\d+)\}?")

def parse_bible(text: str):
    records = []
    book = chapter = verse = None
    buffer = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.upper() in BOOK_SET:
            book = line.title()
            chapter = verse = None
            continue

        m = VERSE_RE.search(line)
        if m and book:
            if verse:
                records.append({
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": buffer.strip()
                })

            chapter = int(m.group(1))
            verse = int(m.group(2))
            buffer = VERSE_RE.sub("", line).strip()
        elif verse:
            buffer += " " + line

    if verse:
        records.append({
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "text": buffer.strip()
        })

    return records
