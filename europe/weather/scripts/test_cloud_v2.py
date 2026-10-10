"""Проверки смысла облачных пропусков и границы наблюдения/прогноза."""
import json
import unittest
from datetime import datetime,timezone
from decode_metar import decode as old
from decode_metar_cloud_v2 import decode

def report(body):
    raw='LFPG 191155Z AUTO '+body
    return decode(raw,'LFPG',datetime(2021,1,19,11,55,tzinfo=timezone.utc))

class CloudV2Tests(unittest.TestCase):
    def test_known_type_unknown_amount_height(self):
        for token,field in [('///CB','cumulonimbus_reported'),('///TCU','towering_cumulus_reported')]:
            row=report('24010KT 9999 '+token+' 10/08 Q1002')
            self.assertTrue(row['decoded_candidate'])
            self.assertTrue(row[field])
            self.assertTrue(row['cloud_amount_unavailable'])
            layer=json.loads(row['cloud_layers_json'])[0]
            self.assertIsNone(layer['height_ft'])
            self.assertEqual(layer['cover'],'///')
            self.assertIsNone(row['cloud_ceiling_ft'])
    def test_unknown_amount_blocks_ceiling_summary(self):
        row=report('24010KT 9999 BKN008/// ///TCU 10/08 Q1002')
        self.assertTrue(row['decoded_candidate'])
        self.assertEqual(row['lowest_known_cloud_base_ft'],800)
        self.assertIsNone(row['cloud_ceiling_ft'])
        self.assertEqual(row['cloud_ceiling_status'],'amount_unknown')
    def test_all_missing_cloud_components_preserved(self):
        row=report('24010KT //// // ///////// 10/08 Q1002')
        self.assertTrue(row['decoded_candidate'])
        self.assertEqual(row['cloud_tokens'],'/////////')
        self.assertTrue(row['cloud_amount_unavailable'])
        self.assertTrue(row['cloud_type_unavailable'])
        self.assertIsNone(row['cloud_ceiling_ft'])
        self.assertFalse(row['cavok'])
    def test_trend_does_not_create_observed_convection(self):
        raw='LFPG 191155Z AUTO 24010KT 9999 FEW008 10/08 Q1002 TEMPO ///TCU'
        stamp=datetime(2021,1,19,11,55,tzinfo=timezone.utc)
        self.assertEqual(decode(raw,'LFPG',stamp),old(raw,'LFPG',stamp))
        self.assertFalse(decode(raw,'LFPG',stamp)['towering_cumulus_reported'])
    def test_unclassified_tail_is_not_promoted_to_observation(self):
        row=report('24010KT 9999 ///TCU 10/08 Q1002 2000 SHRA BKN004')
        self.assertFalse(row['decoded_candidate'])
        self.assertEqual(row['present_weather_tokens'],'')
        self.assertTrue(row['towering_cumulus_reported'])
    def test_other_issues_are_not_suppressed(self):
        row=report('24010KT 9999 ///TCU 10/08')
        self.assertFalse(row['decoded_candidate'])
        self.assertIn('qnh_group_missing_or_multiple',row['decode_issues'])

if __name__=='__main__':unittest.main()
