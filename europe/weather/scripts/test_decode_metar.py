"""Проверки авиационной семантики, включая случаи вне текущего архива."""
from datetime import datetime, timezone
import unittest
from decode_metar import decode

TIME = datetime(2019, 1, 1, 12, 0, tzinfo=timezone.utc)


def report(groups):
    return decode('EGLL 011200Z '+groups, 'EGLL', TIME)


class DecodeSemantics(unittest.TestCase):
    def test_forecast_does_not_replace_observation(self):
        row = report('24010KT 8000 -RA BKN020 08/05 Q1013 TEMPO 25035G50KT 1000 +SN BKN002')
        self.assertEqual(row['wind_speed_kt'], 10)
        self.assertIsNone(row['wind_gust_kt'])
        self.assertEqual(row['visibility_m'], 8000)
        self.assertEqual(row['present_weather_tokens'], '-RA')
        self.assertEqual(row['cloud_ceiling_ft'], 2000)
        self.assertEqual(row['parser_status'], 'ok')

    def test_cavok_and_censored_visibility(self):
        cavok = report('VRB02KT CAVOK 08/05 Q1013 NOSIG')
        self.assertEqual(cavok['visibility_m'], 10000)
        self.assertEqual(cavok['visibility_relation'], 'ge')
        self.assertIsNone(cavok['cloud_ceiling_ft'])
        self.assertTrue(cavok['wind_variable'])
        self.assertIsNone(cavok['wind_direction_deg'])
        low = report('24010KT 0000 FG VV001 08/08 Q1013')
        self.assertEqual((low['visibility_m'], low['visibility_relation']), (50, 'lt'))
        high = report('24010KT 9999 SCT030 08/05 Q1013')
        self.assertEqual((high['visibility_m'], high['visibility_relation']), (10000, 'ge'))

    def test_missing_visibility_is_not_ten_kilometres(self):
        row = report('24010KT //// NCD 08/05 Q1013')
        self.assertIsNone(row['visibility_m'])
        self.assertEqual(row['visibility_relation'], 'missing')

    def test_missing_gust_and_calm(self):
        row = report('00000KT CAVOK 08/05 Q1013')
        self.assertTrue(row['wind_calm'])
        self.assertIsNone(row['wind_direction_deg'])
        self.assertIsNone(row['wind_gust_kt'])
        self.assertFalse(row['gust_group_reported'])

    def test_units_and_wraparound_direction(self):
        row = report('36010G20MPS 330V030 9999 BKN020 08/05 Q1013')
        self.assertAlmostEqual(row['wind_speed_kt'], 19.4384449244)
        self.assertEqual(row['wind_unit'], 'MPS')
        self.assertEqual((row['wind_variable_from_deg'], row['wind_variable_to_deg']), (330, 30))

    def test_pressure_and_negative_temperature(self):
        row = report('24010KT 9999 BKN020 M03/M05 A2992 RMK SLP123')
        self.assertAlmostEqual(row['qnh_hpa'], 1013.207489, places=4)
        self.assertEqual(row['qnh_unit'], 'inHg')
        self.assertEqual(row['temperature_c'], -3)
        self.assertNotIn('SLP123', row['observed_text'])

    def test_recent_weather_and_unknown_cloud_height(self):
        row = report('24010KT 8000 -RA BKN/// 08/05 Q1013 RETS')
        self.assertEqual(row['present_weather_tokens'], '-RA')
        self.assertEqual(row['recent_weather_tokens'], 'RETS')
        self.assertIsNone(row['cloud_ceiling_ft'])
        self.assertEqual(row['cloud_ceiling_status'], 'height_missing')

    def test_unrecognized_groups_are_audited(self):
        row = report('24010KT 8000 BKN020 08/05 Q1013 BADTOKEN')
        self.assertFalse(row['decoded_candidate'])
        self.assertEqual(row['parser_status'], 'error')
        self.assertIn('BADTOKEN', row['parser_error'])

    def test_automatic_cloud_unknowns_are_preserved(self):
        row = report('24010KT 9999 SCT020/// BKN030/// //////CB 08/05 Q1013')
        self.assertEqual(row['parser_status'], 'ok')
        self.assertTrue(row['cumulonimbus_reported'])
        self.assertTrue(row['cloud_amount_unavailable'])
        self.assertTrue(row['cloud_type_unavailable'])
        self.assertIsNone(row['cloud_ceiling_ft'])
        bare = report('24010KT 9999 ////// 08/05 Q1013')
        self.assertEqual(bare['parser_status'], 'ok')
        self.assertEqual(bare['present_weather_tokens'], '')

    def test_cloud_zero_code_is_a_bound(self):
        row = report('24010KT 0500 FG BKN000 08/08 Q1013')
        self.assertEqual(row['cloud_ceiling_ft'], 100)
        self.assertEqual(row['cloud_ceiling_relation'], 'lt')

    def test_unlabelled_post_pressure_weather_is_not_observed(self):
        row = report('VRB03KT 9999 BKN015CB 02/00 Q1000 2000 SHSN TSGS BKN004')
        self.assertFalse(row['decoded_candidate'])
        self.assertEqual(row['present_weather_tokens'], '')
        self.assertEqual(row['cloud_ceiling_ft'], 1500)
        self.assertIn('SHSN', row['unclassified_tail_text'])

    def test_unclassified_slashes_do_not_erase_known_temperature(self):
        row = report('34008KT 1800 SN BR FEW003 BKN005 00/M00 Q0989 ///// TEMPO 0800 SN')
        self.assertFalse(row['decoded_candidate'])
        self.assertEqual(row['temperature_token'], '00/M00')
        self.assertEqual(row['temperature_c'], 0)
        self.assertEqual(row['dewpoint_c'], 0)

    def test_missing_temperature_components_preserve_slash_codes(self):
        for token,temperature,dewpoint in [('/////',None,None),('///M02',None,-2),('03///',3,None)]:
            with self.subTest(token=token):
                row=report(f'31007KT 9999 FEW033 {token} Q////')
                self.assertEqual(row['temperature_token'],token)
                self.assertEqual(row['temperature_c'],temperature)
                self.assertEqual(row['dewpoint_c'],dewpoint)
                self.assertIsNone(row['qnh_hpa'])

    def test_ndv_suffix_preserves_special_visibility_bounds(self):
        for token,value,relation in [('9999NDV',10000,'ge'),('0000NDV',50,'lt'),('////NDV',None,'missing'),('8000NDV',8000,'eq')]:
            with self.subTest(token=token):
                row=report(f'15014KT {token} FEW020/// 00/M05 Q1007')
                self.assertEqual(row['visibility_token'],token)
                self.assertEqual(row['visibility_m'],value)
                self.assertEqual(row['visibility_relation'],relation)
                self.assertTrue(row['visibility_ndv'])


if __name__ == '__main__':
    unittest.main()
