"""Проверки консервативного аудита основных полей перед масштабированием."""
import unittest
import pandas as pd
from decode_metar import decode
from core_audit import audit_core

class CoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.stamp=pd.Timestamp('2019-01-18T04:55Z')
        self.raw='EHAM 180455Z 17003KT CAVOK M02/M02 Q1017 06/1///95 NOSIG'
    def check(self,raw):
        parsed=decode(raw,'EHAM',self.stamp.to_pydatetime())
        return audit_core(raw,'EHAM',self.stamp,parsed)
    def test_unsupported_tail_does_not_validate_full_message(self):
        parsed=decode(self.raw,'EHAM',self.stamp.to_pydatetime())
        self.assertFalse(parsed['decoded_candidate'])
        self.assertTrue(self.check(self.raw)[0])
    def test_unknown_token_inside_core_is_rejected(self):
        self.assertFalse(self.check(self.raw.replace('CAVOK','INVALID CAVOK'))[0])
    def test_multiple_qnh_is_rejected(self):
        self.assertFalse(self.check(self.raw.replace('Q1017','Q1017 Q1018'))[0])
    def test_missing_temperature_is_rejected(self):
        self.assertFalse(self.check(self.raw.replace('M02/M02 ',''))[0])
    def test_field_disagreement_is_rejected(self):
        parsed=decode(self.raw,'EHAM',self.stamp.to_pydatetime())
        parsed['wind_speed_kt']=99
        self.assertFalse(audit_core(self.raw,'EHAM',self.stamp,parsed)[0])
    def test_raw_is_preserved(self):
        parsed=decode(self.raw,'EHAM',self.stamp.to_pydatetime())
        before=dict(parsed)
        audit_core(self.raw,'EHAM',self.stamp,parsed)
        self.assertEqual(parsed,before)
        self.assertEqual(parsed['metar'],self.raw)

if __name__=='__main__':
    unittest.main(verbosity=2)
