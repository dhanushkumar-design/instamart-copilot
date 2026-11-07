"""Part C: offline LLM ticket triage, redaction, validation and retry."""
import json,re
import pandas as pd
import mock_llm
from .foundation import BASE, RAW

EMAIL=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE=re.compile(r'(?<!\d)[6-9]\d{9}(?!\d)')
CATS=set(mock_llm.CATS)

def redact(text):
    return PHONE.sub('[PHONE]',EMAIL.sub('[EMAIL]',str(text or '')))

def fallback(text):
    words=mock_llm._words(text)
    for cat in ['fraud','technical','billing','delivery']:
        if mock_llm.KW[cat] & words:return cat
    return 'technical'

def classify(text,few_shot=True):
    if not isinstance(text,str) or not text.strip():return {'category':'none','method':'empty','tokens':0}
    safe=redact(text)[:6000]
    system='CLASSIFY. Return only JSON with category and reason. Categories: billing, delivery, technical, fraud.'
    if few_shot:
        system+=' EXAMPLES: unauthorized payment -> fraud; app crashes -> technical; charged twice -> billing; rider late -> delivery.'
    messages=[{'role':'system','content':system},{'role':'user','content':safe}]
    for attempt in range(3):
        try:
            response=mock_llm.chat(messages)
            value=json.loads(response.choices[0].message.content)
            if isinstance(value,dict) and value.get('category') in CATS:
                return {'category':value['category'],'method':'llm','tokens':response.usage.total_tokens}
            return {'category':fallback(safe),'method':'invalid_fallback','tokens':response.usage.total_tokens}
        except mock_llm.TransientLLMError:
            if attempt==2:break
        except (ValueError,TypeError,AttributeError):
            return {'category':fallback(safe),'method':'parse_fallback','tokens':0}
    return {'category':fallback(safe),'method':'error_fallback','tokens':0}

def run(few_shot=True):
    mock_llm.reset()
    tickets=pd.read_csv(RAW/'tickets.csv').fillna('')
    results=[classify(t,few_shot) for t in tickets.text]
    tickets['predicted_category']=[r['category'] for r in results]
    tickets['method']=[r['method'] for r in results]
    tickets['tokens']=[r['tokens'] for r in results]
    tickets['correct']=tickets.true_category==tickets.predicted_category
    tickets.to_csv(BASE/'reports/ticket_predictions.csv',index=False)
    summary={'accuracy':round(float(tickets.correct.mean()),4),'total_tokens':int(tickets.tokens.sum()),'fallbacks':int((tickets.method.str.contains('fallback')).sum()),'by_method':tickets.method.value_counts().to_dict()}
    (BASE/'reports/triage_metrics.json').write_text(json.dumps(summary,indent=2))
    return summary
