from pathlib import Path
from collections import Counter
import gzip, json, tarfile
from inspect_us_archive import Parts
ROOT = Path(__file__).resolve().parents[1]

def main():
    reports=[]
    for year in (2025,2026):
        directory=ROOT/f'data/raw/usa/{year}-01-15'
        paths=sorted(p for p in directory.glob('*.tar.a*') if not p.name.endswith('.part'))
        if len(paths)!=2:
            raise ValueError(f'Incomplete day {year}')
        counts=Counter()
        samples=[]
        members=0
        total_payload=0
        first_names=[]
        with Parts(paths) as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
            for member in archive:
                members+=1
                total_payload+=member.size
                if len(first_names)<10: first_names.append(member.name)
                if not member.isfile(): continue
                components=member.name.strip('./').split('/')
                category='traces' if 'traces' in components else 'heatmap' if 'heatmap' in components else 'other'
                counts[category]+=1
                if category=='traces' and len(samples)<3 and 'trace_full_' in member.name:
                    if member.size>20*1024*1024:
                        raise ValueError('Unexpectedly large sample member')
                    content=archive.extractfile(member).read()
                    if content.startswith(b'\x1f\x8b'): content=gzip.decompress(content)
                    data=json.loads(content)
                    points=data.get('trace',[])
                    samples.append({'member':member.name,'icao':data.get('icao'),'timestamp':data.get('timestamp'),'points':len(points),'point_lengths':sorted(set(map(len,points))),'first_point':points[0] if points else None})
        report={'date':f'{year}-01-15','parts':[p.name for p in paths],'members':members,'payload_bytes':total_payload,'file_categories':dict(counts),'first_member_names':first_names,'json_samples':samples,'validation':'TAR headers read to end; three complete trace JSON samples checked. All trace gzip streams not individually validated.','us_airports_extracted':False,'goaround_labels_created':False}
        reports.append(report)
        print(year,'members',members,'files',dict(counts),'sample traces',len(samples),flush=True)
    (ROOT/'reports/usa_archive_validation.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf8')

if __name__=='__main__': main()
