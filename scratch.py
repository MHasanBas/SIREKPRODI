import pandas as pd
df = pd.DataFrame({"ASAL SEKOLAH": ["MAN 1 PASURUAN", "UPT SMAN 1 PASURUAN", "SMAN 1 PASURUAN", "MAN 2 PASURUAN", "SMA MANUNGGAL"]})
search_term = "man 1 pasuruan"
# Option 1: word boundaries
# Need to escape special characters in search_term? Yes.
import re
escaped_search = re.escape(search_term)
pattern = r'(?i)\b' + escaped_search + r'\b'
print(df[df["ASAL SEKOLAH"].str.contains(pattern, case=False, na=False)])

search_term2 = "man"
escaped_search2 = re.escape(search_term2)
pattern2 = r'(?i)\b' + escaped_search2 + r'\b'
print(df[df["ASAL SEKOLAH"].str.contains(pattern2, case=False, na=False)])

