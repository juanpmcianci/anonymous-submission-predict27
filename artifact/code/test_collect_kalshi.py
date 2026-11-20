"""Collector boundary tests: chronology, complete pagination and no silent truncation."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.find_spec('collect_kalshi')
if SPEC:
    import collect_kalshi as c

class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, 'collector implementation is absent')

    def market(self, **changes):
        m = dict(ticker='KXBTC15M-X', event_ticker='KXBTC15M-E', market_type='binary',
                 open_time='2026-08-02T23:45:00Z', close_time='2026-08-03T00:00:00Z',
                 status='finalized', result='yes', settlement_ts='2026-08-03T00:00:07Z')
        m.update(changes)
        return m

    def test_utc_hour_and_half_open_range(self):
        self.assertEqual(c.eligibility(self.market()), [])
        self.assertIn('not_exact_utc_hour', c.eligibility(self.market(close_time='2026-08-03T00:15:00Z')))
        self.assertIn('outside_close_range', c.eligibility(self.market(close_time='2026-09-28T00:00:00Z')))
        self.assertEqual(c.eligibility(self.market(open_time='2026-08-02T19:45:00-04:00', close_time='2026-08-02T20:00:00-04:00')), [])
        self.assertIn('not_exact_utc_hour', c.eligibility(self.market(close_time='2026-08-03T00:00:00.500Z')))

    def test_metadata_finality_without_outcome_preference(self):
        self.assertEqual(c.eligibility(self.market(result='no')), [])
        for changes, reason in [({'is_provisional': True}, 'provisional'), ({'result': ''}, 'nonbinary_result'), ({'status':'closed'}, 'not_finalized'), ({'open_time':'2026-08-02T23:46:00Z'}, 'not_900_seconds')]:
            self.assertIn(reason, c.eligibility(self.market(**changes)))

    def test_pagination_continues_after_short_page(self):
        def get(path, params):
            return {'markets':[{'ticker':'a'}], 'cursor':'next'} if 'cursor' not in params else {'markets':[{'ticker':'b'}], 'cursor':''}
        self.assertEqual([x['ticker'] for x in c.paginate(get, '/markets', {'limit':1000}, 'markets')], ['a','b'])

    def test_repeated_cursor_raises_instead_of_silent_incomplete_data(self):
        with self.assertRaises(ValueError):
            c.paginate(lambda p,q:{'markets':[], 'cursor':'repeat'}, '/markets', {}, 'markets')

    def test_duplicate_identity_conflicts_are_rejected(self):
        self.assertEqual(c.deduplicate([{'id':'a','v':1},{'id':'a','v':1}], 'id'), ([{'id':'a','v':1}], 1))
        with self.assertRaises(ValueError):
            c.deduplicate([{'id':'a','v':1},{'id':'a','v':2}], 'id')

    def test_batch_requires_all_tickers_and_unique_timestamps(self):
        with self.assertRaises(ValueError): c.parse_batch({'markets':[]}, ['a'])
        with self.assertRaises(ValueError):
            c.parse_batch({'markets':[{'market_ticker':'a','candlesticks':[{'end_period_ts':1},{'end_period_ts':1}]}]}, ['a'])
        self.assertEqual(c.parse_batch({'markets':[{'market_ticker':'a','candlesticks':[{'end_period_ts':2},{'end_period_ts':1}]}]}, ['a'])['a'][0]['end_period_ts'], 1)

    def test_batch_partition_respects_rectangular_api_cap(self):
        self.assertTrue(hasattr(c, 'candle_batches'), 'safe batch partition is absent')
        markets=[self.market(ticker=f'KXBTC15M-{hour}', open_time=f'2026-08-03T{hour-1:02d}:45:00Z', close_time=f'2026-08-03T{hour:02d}:00:00Z') for hour in range(1,24)]
        batches=c.candle_batches(markets)
        self.assertEqual([m['ticker'] for b in batches for m in b], [m['ticker'] for m in markets])
        for batch in batches:
            rectangular=len(batch)*(int((max(c.timestamp(m['close_time']) for m in batch)-min(c.timestamp(m['open_time']) for m in batch))/60)+1)
            self.assertLess(rectangular,10000)

    def test_real_probe_decimal_volume_and_full_window(self):
        p=Path(__file__).resolve().parents[1]/'revision/data_probe'
        m=json.loads((p/'market.json').read_text())['market']
        rows=json.loads((p/'candlesticks.json').read_text())['candlesticks']
        audit=c.audit_candles(m, rows)
        self.assertTrue(audit['full_window']);self.assertTrue(audit['volume_matches'])
        self.assertFalse(c.audit_candles(m, rows[:-1])['full_window'])

    def test_postclose_candle_is_audited_outside_full_window(self):
        p=Path(__file__).resolve().parents[1]/'revision/data_probe'
        m=json.loads((p/'market.json').read_text())['market']
        rows=json.loads((p/'candlesticks.json').read_text())['candlesticks']
        audit=c.audit_candles(m, rows+[dict(end_period_ts=1790631060,volume_fp='0.00')])
        self.assertTrue(audit['full_window'])
        self.assertTrue(audit['volume_matches'])
        self.assertEqual(audit['outside_window'],[1790631060])

    def test_close_query_avoids_status_and_pads_boundary(self):
        q=c.market_query('KXBTC15M')
        self.assertNotIn('status', q)
        self.assertEqual(q['min_close_ts'],1785715199)
        self.assertEqual(q['max_close_ts'],1790553600)

if __name__=='__main__': unittest.main()
