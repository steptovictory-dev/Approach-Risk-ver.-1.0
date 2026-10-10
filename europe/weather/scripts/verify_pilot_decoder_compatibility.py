"""Проверить неизменность ранее сохраненных полей после исправления декодера."""
import pandas as pd
from audit_pilot_input import BASE
from decode_metar import decode

for name in ['04_decoded_reports.parquet','09_decoded_reports.parquet']:
    previous=pd.read_parquet(BASE/'data'/name)
    repeated=[]
    for row in previous.itertuples(index=False):
        values=decode(row.metar,row.station,row.metar_utc.to_pydatetime())
        values['archive_month']=row.archive_month
        repeated.append(values)
    frame=pd.DataFrame(repeated)
    frame['metar_utc']=pd.to_datetime(frame.metar_utc,utc=True).astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(frame,previous,check_dtype=False)
    print(name,len(frame),'all saved fields unchanged',flush=True)
