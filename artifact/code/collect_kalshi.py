"""Frozen-cohort public-data collection only. No forecasts or outcome summaries.

Run: /opt/anaconda3/bin/python3 code/collect_kalshi.py
Repeat uses checksummed raw cache. --offline forbids any new requests.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

BASE='https://external-api.kalshi.com/trade-api/v2'
SERIES=('KXBTC15M','KXETH15M')
START='2026-08-03T00:00:00Z'
STOP='2026-09-28T00:00:00Z'
ROOT=Path(__file__).resolve().parent/'empirical_data'

def timestamp(s):
    dt=datetime.fromisoformat(s.replace('Z','+00:00'))
    if dt.tzinfo is None: raise ValueError('timezone required')
    return dt.timestamp()

START_TS=int(timestamp(START)); STOP_TS=int(timestamp(STOP))

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    tmp.replace(path)

def eligibility(m):
    reasons=[]
    try:
        close=timestamp(m['close_time']); opened=timestamp(m['open_time'])
    except (KeyError,ValueError,TypeError): return ['invalid_times']
    if not START_TS<=close<STOP_TS: reasons.append('outside_close_range')
    if close%3600: reasons.append('not_exact_utc_hour')
    if close-opened!=900: reasons.append('not_900_seconds')
    if m.get('ticker','').split('-')[0] not in SERIES: reasons.append('wrong_series')
    if not m.get('event_ticker'): reasons.append('missing_event')
    if m.get('market_type')!='binary': reasons.append('nonbinary_market')
    if m.get('result') not in ('yes','no'): reasons.append('nonbinary_result')
    if m.get('status')!='finalized': reasons.append('not_finalized')
    if m.get('is_provisional',False): reasons.append('provisional')
    if not m.get('settlement_ts'): reasons.append('missing_settlement')
    return reasons

def market_query(series):
    # Pad the inclusive lower boundary because API describes min as "after".
    return dict(series_ticker=series,min_close_ts=START_TS-1,max_close_ts=STOP_TS,limit=1000)

def paginate(get,path,params,key,max_pages=100):
    rows=[]; seen=set(); q=dict(params)
    for _ in range(max_pages):
        data=get(path,q)
        if key not in data or not isinstance(data[key],list): raise ValueError('invalid page schema')
        rows.extend(data[key]); cursor=data.get('cursor')
        if not cursor: return rows
        if cursor in seen: raise ValueError('repeated pagination cursor')
        seen.add(cursor);q['cursor']=cursor
    raise ValueError('pagination safety limit; collection incomplete')

def deduplicate(rows,key):
    by={}; duplicates=0
    for row in rows:
        k=row[key]
        if k in by:
            if by[k]!=row: raise ValueError(f'conflicting duplicate {key}: {k}')
            duplicates+=1
        else: by[k]=row
    return list(by.values()),duplicates

def candle_batches(markets):
    # API charges the entire time rectangle, including times outside each life.
    batch=[]
    for m in markets:
        trial=batch+[m]
        span=(max(timestamp(x['close_time']) for x in trial)-min(timestamp(x['open_time']) for x in trial))/60
        if batch and (len(trial)>100 or len(trial)*(int(span)+1)>=10000):
            yield batch
            batch=[m]
        else: batch=trial
    if batch: yield batch

def parse_batch(data,tickers):
    rows=data['markets']; actual=[r['market_ticker'] for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=set(tickers): raise ValueError('batch ticker coverage mismatch')
    if sum(len(r['candlesticks']) for r in rows)>=10000: raise ValueError('batch hits candle cap; may be truncated')
    out={}
    for row in rows:
        candles=row['candlesticks']; ends=[x['end_period_ts'] for x in candles]
        if len(ends)!=len(set(ends)): raise ValueError('duplicate candle timestamp')
        out[row['market_ticker']]=sorted(candles,key=lambda x:x['end_period_ts'])
    return out

def audit_candles(m,rows):
    opened=int(timestamp(m['open_time']));close=int(timestamp(m['close_time']))
    expected=list(range(opened+60,close+1,60)); got=[x['end_period_ts'] for x in rows]
    in_window=[x for x in rows if opened < x['end_period_ts'] <= close]
    full=[x['end_period_ts'] for x in in_window]==expected
    volume=sum((Decimal(x['volume_fp']) for x in in_window),Decimal(0))
    return dict(ticker=m['ticker'],n_candles=len(in_window),raw_candles=len(rows),full_window=full,
                missing_end_timestamps=sorted(set(expected)-set(got)),
                outside_window=sorted(set(got)-set(expected)),
                candle_volume=str(volume),market_volume=m.get('volume_fp'),
                volume_matches=full and volume==Decimal(m['volume_fp']))

class CachedClient:
    def __init__(self,root=ROOT,offline=False,max_bytes=100_000_000):
        self.root=Path(root);self.raw=self.root/'raw';self.raw.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.root/'manifest.json';self.offline=offline;self.max_bytes=max_bytes
        self.entries=json.loads(self.manifest_path.read_text())['requests'] if self.manifest_path.exists() else []
        self.by_url={x['url']:x for x in self.entries}
        self.bytes=sum(x['bytes'] for x in self.entries)
    def get(self,path,params=None):
        url=BASE+path+(('?'+urlencode(sorted(params.items()))) if params else '')
        name=hashlib.sha256(url.encode()).hexdigest()+'.json';p=self.raw/name
        if url in self.by_url:
            entry=self.by_url[url];data=p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']: raise ValueError('raw cache checksum mismatch')
            return json.loads(data)
        if self.offline: raise RuntimeError('uncached URL requested in offline mode: '+url)
        remaining=self.max_bytes-self.bytes
        if remaining<=0: raise RuntimeError('raw byte budget exhausted')
        started=datetime.now(timezone.utc).isoformat();time.sleep(.15)
        proc=subprocess.run(['curl','--silent','--show-error','--fail','--max-time','45',
            '--retry','3','--retry-delay','1','--retry-all-errors','--max-filesize',str(min(remaining,6_000_000)),url],capture_output=True)
        if proc.returncode: raise RuntimeError(f'public request failed {path}: {proc.stderr.decode()}')
        data=proc.stdout
        if len(data)>remaining: raise RuntimeError('raw byte budget exceeded')
        obj=json.loads(data)
        if not isinstance(obj,dict) or 'error' in obj: raise ValueError('API error payload')
        p.write_bytes(data)
        entry=dict(url=url,file='raw/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
            request_started_utc=started,retrieved_utc=datetime.now(timezone.utc).isoformat())
        self.entries.append(entry);self.by_url[url]=entry;self.bytes+=len(data)
        write_json(self.manifest_path,dict(base_url=BASE,authentication='none',tls_verification=True,
            max_raw_bytes=self.max_bytes,requests=self.entries))
        return obj

def collect(client):
    root=client.root
    cutoff=client.get('/historical/cutoff')
    if timestamp(cutoff['market_settled_ts'])>START_TS or timestamp(cutoff['trades_created_ts'])>START_TS-900:
        raise RuntimeError('historical cutoff moved into cohort; explicit historical-route amendment required')
    candidates=[]
    for series in SERIES:
        rows=paginate(client.get,'/markets',market_query(series),'markets')
        candidates.extend(rows)
        print(f'Metadata {series}: {len(rows)} candidate rows; no outcome counts computed',flush=True)
    candidates,dups=deduplicate(candidates,'ticker')
    write_json(root/'candidate_markets.json',candidates)
    selected=[];ledger=[]
    for m in candidates:
        reasons=eligibility(m)
        if reasons: ledger.append(dict(ticker=m['ticker'],reasons=reasons))
        else: selected.append(m)
    selected.sort(key=lambda m:(m['close_time'],m['ticker']))
    # Ensure one contract per series/hour and an unambiguous event identity.
    slots=Counter((m['ticker'].split('-')[0],int(timestamp(m['close_time']))) for m in selected)
    if any(v!=1 for v in slots.values()): raise ValueError('multiple markets in scheduled series/hour slot')
    events=Counter(m['event_ticker'] for m in selected)
    if any(v!=1 for v in events.values()): raise ValueError('nonunique ticker/event mapping')
    expected={(series,t) for series in SERIES for t in range(START_TS,STOP_TS,3600)}
    missing=[dict(series=s,close_time=datetime.fromtimestamp(t,timezone.utc).isoformat()) for s,t in sorted(expected-set(slots))]
    write_json(root/'markets.json',selected);write_json(root/'exclusion_ledger.json',ledger)
    print(f'Eligible metadata: {len(selected)}; missing nominal slots: {len(missing)}',flush=True)
    groups=defaultdict(list)
    for m in selected:groups[datetime.fromtimestamp(timestamp(m['close_time']),timezone.utc).date().isoformat()].append(m)
    candles={}
    for date,ms in sorted(groups.items()):
        for batch in candle_batches(ms):
            params=dict(market_tickers=','.join(m['ticker'] for m in batch),
                        start_ts=int(min(timestamp(m['open_time']) for m in batch))+1,
                        end_ts=int(max(timestamp(m['close_time']) for m in batch)),period_interval=1)
            candles.update(parse_batch(client.get('/markets/candlesticks',params),[m['ticker'] for m in batch]))
        print(f'Candles {date}: {sum(len(candles[m["ticker"]]) for m in ms)} rows',flush=True)
    audits=[audit_candles(m,candles[m['ticker']]) for m in selected]
    candles={m['ticker']:[row for row in candles[m['ticker']] if timestamp(m['open_time']) < row['end_period_ts'] <= timestamp(m['close_time'])] for m in selected}
    write_json(root/'candles.json',candles)
    write_json(root/'candle_audit.json',audits)
    summary=dict(nominal_slots=len(expected),candidate_rows=len(candidates),identical_duplicate_metadata_rows=dups,
        metadata_eligible=len(selected),exclusion_reason_counts=dict(Counter(r for x in ledger for r in x['reasons'])),
        missing_nominal_slots=missing,full_window_markets=sum(x['full_window'] for x in audits),
        exact_volume_matches=sum(x['volume_matches'] for x in audits),outside_window_candles=sum(len(x['outside_window']) for x in audits),candle_rows=sum(map(len,candles.values())),
        raw_bytes=client.bytes,request_count=len(client.entries))
    write_json(root/'collection_audit.json',summary)
    # Deterministic tiny audit: earliest training contract, first minute only.
    if selected:
        m=selected[0];start=int(timestamp(m['open_time']));end=start+60
        trades=paginate(client.get,'/markets/trades',dict(ticker=m['ticker'],min_ts=start-1,max_ts=end+1,limit=1000),'trades',max_pages=20)
        trades,td=deduplicate(trades,'trade_id')
        if any(t['ticker']!=m['ticker'] for t in trades):raise ValueError('unexpected trade ticker')
        interval=[t for t in trades if start<=timestamp(t['created_time'])<end]
        vol=sum((Decimal(t['count_fp']) for t in interval),Decimal(0))
        candle=next((r for r in candles[m['ticker']] if r['end_period_ts']==end),None)
        trade_audit=dict(ticker=m['ticker'],start_ts=start,end_ts=end,interval_convention='[start,end)',
            pagination_complete=True,returned_trades=len(trades),interval_trades=len(interval),duplicate_trades=td,
            interval_volume=str(vol),candle_volume=candle['volume_fp'] if candle else None,
            volume_matches=candle is not None and vol==Decimal(candle['volume_fp']))
        write_json(root/'trade_audit.json',trade_audit)
        summary.update(raw_bytes=client.bytes,request_count=len(client.entries),trade_reconciliation=trade_audit)
        write_json(root/'collection_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('missing_nominal_slots','trade_reconciliation')},indent=2),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true');parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();collect(CachedClient(args.output,args.offline))
if __name__=='__main__':main()
