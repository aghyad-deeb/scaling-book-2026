#!/usr/bin/env python3
"""Compare two versions of a chapter and report anything a prose-only restyle must not change:
numbers, citation keys, footnote count, headings (the toc depends on them), table rows, figure
includes, details blocks, math blocks, code spans. Usage: python3 scripts/invariants.py OLD.md NEW.md"""
import re, sys
from collections import Counter

def front_matter_and_body(s):
    parts = s.split('\n---\n', 1)
    if s.startswith('---\n') and len(parts) == 2:
        return parts[0], parts[1]
    return '', s

def features(s):
    fm, body = front_matter_and_body(s)
    f = {}
    f['numbers'] = Counter(re.findall(r'(?<![\w.])\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?', body))
    f['cites'] = Counter(re.findall(r'<d-cite key="([^"]+)"', body))
    f['footnotes'] = body.count('<d-footnote>')
    f['headings'] = re.findall(r'^#{2,3} .*$', body, flags=re.M)
    f['table_rows'] = len([l for l in body.split('\n') if l.startswith('|')])
    f['figures'] = re.findall(r'\{% include figure\.liquid path="([^"]+)"', body)
    f['details'] = body.count('{% details'), body.count('{% enddetails %}')
    f['display_math'] = len(re.findall(r'\$\$', body)) // 2
    f['code_spans'] = Counter(re.findall(r'`([^`]+)`', body))
    f['takeaways'] = body.count('class="takeaway"')
    f['questions'] = len(re.findall(r'^\*\*Question', body, flags=re.M))
    f['front_matter'] = fm
    f['words'] = len(body.split())
    return f

def main(a, b):
    A, B = features(open(a).read()), features(open(b).read())
    ok = True
    def report(name, cond, detail=''):
        nonlocal ok
        print(('OK   ' if cond else 'DIFF ') + name + (': ' + detail if detail else ''))
        ok = ok and cond
    report('front matter identical', A['front_matter'] == B['front_matter'])
    report('headings identical', A['headings'] == B['headings'], '' if A['headings'] == B['headings'] else f"\n  old: {A['headings']}\n  new: {B['headings']}")
    report('figures identical', A['figures'] == B['figures'])
    report('details blocks', A['details'] == B['details'], f"{A['details']} -> {B['details']}")
    report('takeaway boxes', A['takeaways'] == B['takeaways'], f"{A['takeaways']} -> {B['takeaways']}")
    report('worked problems', A['questions'] == B['questions'], f"{A['questions']} -> {B['questions']}")
    report('footnotes', A['footnotes'] == B['footnotes'], f"{A['footnotes']} -> {B['footnotes']}")
    report('table rows', A['table_rows'] == B['table_rows'], f"{A['table_rows']} -> {B['table_rows']}")
    report('display math blocks', A['display_math'] == B['display_math'], f"{A['display_math']} -> {B['display_math']}")
    missing_cites = A['cites'] - B['cites']; added_cites = B['cites'] - A['cites']
    report('citations', not missing_cites and not added_cites, f"missing {dict(missing_cites)} added {dict(added_cites)}")
    missing_code = A['code_spans'] - B['code_spans']; added_code = B['code_spans'] - A['code_spans']
    report('code spans (backtick arithmetic)', not missing_code and not added_code, f"missing {list(missing_code)[:8]} added {list(added_code)[:8]}")
    lost = A['numbers'] - B['numbers']; gained = B['numbers'] - A['numbers']
    # tolerate small count changes only for very common small integers used in prose (e.g. "two", years)
    serious_lost = {k: v for k, v in lost.items() if not (len(k) <= 1)}
    serious_gained = {k: v for k, v in gained.items() if not (len(k) <= 1)}
    report('numbers (multiset)', not serious_lost and not serious_gained, f"lost {serious_lost} gained {serious_gained}")
    print(f"words: {A['words']} -> {B['words']} ({(B['words'] - A['words']) / A['words']:+.0%})")
    print('ALL INVARIANTS HOLD' if ok else 'INVARIANT VIOLATIONS ABOVE: each must be justified or reverted')
    return ok

if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
