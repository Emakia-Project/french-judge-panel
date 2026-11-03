import json, pandas as pd
POS = "Harassment"

j  = pd.read_csv('paper/joined_for_tables.csv', dtype={'id':str})
v1 = pd.read_csv('data/consensus_fixed_2026-09-04_0423__french2026_d2_8436__prompt-fr__rep1__k3__5judges__b4d9cce4.csv', dtype={'id':str})
v2 = pd.read_csv('data/consensus_2026-09-08_0035__all_8436_8436__prompt-fr__rep1__k3__5judges__2709a522.csv', dtype={'id':str})

vis = [json.loads(l) for l in open('runs_vision/20260904T213938Z/results.jsonl') if l.strip()]
vis = pd.DataFrame(vis); vis = vis[vis.tweet_id.notna()]
vis['tweet_id'] = vis.tweet_id.astype(str)
per = vis.groupby('tweet_id').agg(
        n_media=('media_key','size'),
        vis_pos=('verdict', lambda s:(s==POS).any())).reset_index().rename(columns={'tweet_id':'id'})

d = (v1[['id','votes','consensus']].rename(columns={'votes':'v1','consensus':'c1'})
     .merge(v2[['id','votes','consensus']].rename(columns={'votes':'v2','consensus':'c2'}), on='id')
     .merge(j[['id','stream'] + [c for c in ('x_flag','possibly_sensitive') if c in j.columns]], on='id', how='left')
     .merge(per, on='id', how='left'))
xc = 'x_flag' if 'x_flag' in d.columns else 'possibly_sensitive'
d['x'] = d[xc].fillna(False).astype(bool)
d['vis_pos'] = d.vis_pos.fillna(False).astype(bool)
d['has_media'] = d.n_media.notna()

def rate(g, col): return 100*g[col].mean() if len(g) else float('nan')

print(f"corpus: {len(d):,} | X fires {100*d.x.mean():.2f}% | "
      f"media-bearing {100*d.has_media.mean():.1f}% | vision+ {100*d.vis_pos.mean():.1f}%\n")

print("=== BY v1 VOTE BAND ===")
print(f"{'votes':>6}{'n':>7}{'X%':>7}{'media%':>8}{'vis+%':>7}{'vis+ | media%':>15}")
for v in range(6):
    g = d[d.v1==v]; m = g[g.has_media]
    print(f"{v:>6}{len(g):>7,}{rate(g,'x'):>7.2f}{100*len(m)/max(len(g),1):>8.1f}"
          f"{rate(g,'vis_pos'):>7.1f}{rate(m,'vis_pos'):>15.1f}")

print("\n=== THE HARD CASES: v1 said Harassment, did it survive v2? ===")
pos = d[d.c1.eq(POS)]
for lab, g in [("survived", pos[pos.c2.eq(POS)]), ("flipped to Neutral", pos[~pos.c2.eq(POS)])]:
    m = g[g.has_media]
    print(f"  {lab:22} n={len(g):>5,}  X {rate(g,'x'):5.2f}%  "
          f"media {100*len(m)/max(len(g),1):5.1f}%  vis+|media {rate(m,'vis_pos'):5.1f}%")

print("\n=== v1 Neutral -> v2 Harassment (the 74) ===")
g = d[~d.c1.eq(POS) & d.c2.eq(POS)]; m = g[g.has_media]
print(f"  n={len(g)}  X {rate(g,'x'):.2f}%  media {100*len(m)/max(len(g),1):.1f}%  "
      f"vis+|media {rate(m,'vis_pos'):.1f}%")

print("\n=== CONTESTED BAND (2-3 votes) — do the signals separate it? ===")
c = d[d.v1.isin([2,3])]
for lab, g in [("became v2 POS", c[c.c2.eq(POS)]), ("became v2 NEG", c[~c.c2.eq(POS)])]:
    m = g[g.has_media]
    print(f"  {lab:16} n={len(g):>5,}  X {rate(g,'x'):5.2f}%  vis+|media {rate(m,'vis_pos'):5.1f}%")
