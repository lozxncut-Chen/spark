from statistics import mean


def work_metrics(input_tokens,start,end,policy,layers):
    work=skipped=0
    for event in policy['events']:
        if event['event']=='forward':
            work+=event['q']*2*layers
            skipped+=event['q']*sum(len(v) for v in event['mask'].values())
    return dict(input_tokens=input_tokens,observation_tokens=end-start,
        observation_input_fraction=(end-start)/input_tokens,
        observation_remaining_fraction=(end-start)/(input_tokens-start),
        dense_token_sublayers=work,skipped_token_sublayers=skipped,
        executed_pruning_fraction=skipped/work if work else 0.)


def summarize(records):
    paired=[r['compensated']['prediction']==r['gold'] for r in records]
    direct=[r['direct']['prediction']==r['gold'] for r in records]
    metrics=[r['observation'] for r in records]
    work=sum(m['dense_token_sublayers'] for m in metrics)
    skipped=sum(m['skipped_token_sublayers'] for m in metrics)
    return dict(count=len(records),compensation_gain_pp=100*(mean(paired)-mean(direct)),
        rescued=sum(a and not b for a,b in zip(paired,direct)),
        harmed=sum(b and not a for a,b in zip(paired,direct)),
        mean_observation_input_fraction=mean(m['observation_input_fraction'] for m in metrics),
        mean_observation_remaining_fraction=mean(m['observation_remaining_fraction'] for m in metrics),
        min_observation_tokens=min(m['observation_tokens'] for m in metrics),
        max_observation_tokens=max(m['observation_tokens'] for m in metrics),
        executed_pruning_fraction=skipped/work if work else 0.)


def compare_masks(left,right):
    changed={'attn':0,'ffn':0}
    jaccard=[]
    for a,b in zip(left,right):
        aa,bb=a['compensated']['policy']['prefix_mask'],b['compensated']['policy']['prefix_mask']
        sa,sb=set(),set()
        for kind in changed:
            x,y=set(aa[kind]),set(bb[kind])
            changed[kind]+=x!=y
            sa.update((kind,i) for i in x)
            sb.update((kind,i) for i in y)
        jaccard.append(len(sa&sb)/len(sa|sb) if sa|sb else 1.)
    return dict(changed_requests=changed,mean_mask_jaccard=mean(jaccard))
