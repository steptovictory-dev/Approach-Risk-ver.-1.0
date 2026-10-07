from pathlib import Path
import io,tarfile
root=Path.cwd()
class Parts(io.RawIOBase):
 def __init__(self,paths):
  self.streams=[p.open('rb') for p in paths]; self.index=0
 def readable(self): return True
 def read(self,size=-1):
  if size<0: raise ValueError('Bounded reads required')
  result=bytearray()
  while len(result)<size and self.index<len(self.streams):
   chunk=self.streams[self.index].read(size-len(result))
   if not chunk: self.index+=1
   else: result.extend(chunk)
  return bytes(result)
 def close(self):
  for stream in self.streams: stream.close()
  super().close()
if __name__ == '__main__':
 for year in (2025,2026):
  paths=sorted((root/f'data/raw/usa/{year}-01-15').glob('*.tar.a*'))
  paths=[p for p in paths if not p.name.endswith('.part')]
  if len(paths)!=2: continue
  with Parts(paths) as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
   for i,member in enumerate(archive):
    print(year,member.name,member.size)
    if i>=12: break
