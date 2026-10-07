from pathlib import Path
import urllib.request
root=Path.cwd()
urls={
 'adsblol_2025_README.md':'https://raw.githubusercontent.com/adsblol/globe_history_2025/main/README.md',
 'adsblol_2026_README.md':'https://raw.githubusercontent.com/adsblol/globe_history_2026/main/README.md',
 'adsblol_LICENSE_ODbL.txt':'https://raw.githubusercontent.com/adsblol/globe_history_2026/main/LICENSE-ODbL.txt',
 'adsblol_LICENSE_CC0.txt':'https://raw.githubusercontent.com/adsblol/globe_history_2026/main/LICENSE-cc0.txt',
 'readsb_trace_format.md':'https://raw.githubusercontent.com/wiedehopf/readsb/dev/README-json.md',
 'figuet_OSN23_README.md':'https://raw.githubusercontent.com/figuetbe/OSN23-GoA/main/README.md',
}
for name,url in urls.items():
 try:
  data=urllib.request.urlopen(url,timeout=60).read()
  (root/'docs/sources'/name).write_bytes(data)
  print(name,len(data))
 except Exception as e: print(name,str(e))
