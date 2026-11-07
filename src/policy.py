"""Part D: simple transparent TF-IDF policy retrieval with abstention."""
from pathlib import Path
import json,re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
from .foundation import BASE, RAW

class PolicyAssistant:
    def __init__(self):
        self.chunks=[]
        for path in sorted((BASE/'data/policies').glob('*.txt')):
            content=path.read_text(encoding='utf-8')
            for sentence in re.split(r'(?<=\.)\s+',content):
                if sentence.strip():self.chunks.append((path.name,sentence.strip()))
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        self.matrix=self.vectorizer.fit_transform([x[1] for x in self.chunks])

    def retrieve(self,question,k=3):
        scores=cosine_similarity(self.vectorizer.transform([question]),self.matrix)[0]
        idx=scores.argsort()[::-1][:k]
        return [(self.chunks[i][0],self.chunks[i][1],float(scores[i])) for i in idx if scores[i]>0]

    def answer(self,question):
        found=self.retrieve(question)
        if not found or found[0][2]<0.08:
            return {'answer':'I could not find that information in the policy documents.','sources':[],'tokens':0}
        context='\n'.join(f'[{name}] {sentence}' for name,sentence,_ in found)
        messages=[{'role':'system','content':'POLICY ASSISTANT. Answer only using the provided context.'},
                  {'role':'user','content':f'CONTEXT:\n{context}\nQUESTION: {question}'}]
        resp=mock_llm.chat(messages)
        return {'answer':resp.choices[0].message.content,'sources':list(dict.fromkeys(x[0] for x in found)),'tokens':resp.usage.total_tokens}

def evaluate():
    assistant=PolicyAssistant()
    data=pd.read_csv(RAW/'policy_eval.csv').fillna('')
    rows=[]
    for r in data.itertuples(index=False):
        result=assistant.answer(r.question)
        top=assistant.retrieve(r.question,k=1)
        rows.append({'question':r.question,'expected_doc':r.expected_doc,'retrieved_doc':top[0][0] if top else 'NONE','expected_fact':r.expected_fact,'answer':result['answer'],'fact_present':r.expected_fact.lower() in result['answer'].lower() if r.expected_fact else not result['sources']})
    frame=pd.DataFrame(rows)
    frame.to_csv(BASE/'reports/policy_eval_results.csv',index=False)
    return {'fact_match_rate':round(float(frame.fact_present.mean()),4),'questions':len(frame)}
