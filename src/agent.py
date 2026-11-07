"""Part D: refund-agent simulator. No real payments are made."""
import json,re,sqlite3
import pandas as pd
import mock_llm
from .foundation import BASE, DB, RAW
from .policy import PolicyAssistant

class RefundAgent:
    def __init__(self,max_steps=6,token_budget=5000):
        self.max_steps=max_steps
        self.token_budget=token_budget
        self.policy=PolicyAssistant()
        self.refunds=set()

    def get_order(self,order_id):
        with sqlite3.connect(DB) as con:
            con.row_factory=sqlite3.Row
            row=con.execute('SELECT * FROM orders WHERE order_id=?',(order_id,)).fetchone()
        if row is None:return {'error':'order not found'}
        return {key:row[key] for key in ['order_id','customer_id','status','order_amount','delivery_minutes']}

    def tool(self,name,args,customer_id):
        if name=='get_order':
            order=self.get_order(args.get('order_id',''))
            if 'error' not in order and order['customer_id']!=customer_id:return {'error':'order not found'}
            return order
        if name=='search_policy':
            matches=self.policy.retrieve(args.get('query','refund policy'))
            return {'matches':[{'source':n,'text':s} for n,s,_ in matches]}
        if name=='calculate_refund':
            order_amount=args.get('order_amount'); minutes=args.get('delivery_minutes')
            if order_amount is None or minutes is None:return {'refund_inr':0}
            return {'refund_inr':round(float(order_amount),2) if float(minutes)>45 else 0}
        if name in ('issue_refund','escalate_to_human'):
            oid=args.get('order_id','')
            order=self.get_order(oid)
            if 'error' in order or order['customer_id']!=customer_id:return {'error':'unauthorized order'}
            amount=order['order_amount']; minutes=order['delivery_minutes']
            if order['status']!='completed' or amount is None or minutes is None or minutes<=45:
                return {'error':'not eligible'}
            if name=='issue_refund':
                if amount>1000:return {'error':'human approval required'}
                if oid in self.refunds:return {'error':'already refunded'}
                self.refunds.add(oid)
                return {'refund_inr':round(amount,2),'status':'simulated'}
            return {'refund_inr':round(amount,2),'status':'escalated'}
        return {'error':'unknown tool'}

    def run(self,customer_id,message):
        messages=[{'role':'system','content':'AGENT. Use tools to handle refund requests safely.'},{'role':'user','content':message}]
        tokens=0
        for step in range(self.max_steps):
            response=mock_llm.chat(messages,tools=[])
            tokens+=response.usage.total_tokens
            if tokens>self.token_budget:return {'outcome':'FALLBACK','text':'Token budget exceeded; hand over to human.','steps':step+1,'tokens':tokens}
            m=response.choices[0].message
            if not m.tool_calls:
                content=m.content or ''
                outcome=content.split(':',1)[0]
                return {'outcome':outcome,'text':content,'steps':step+1,'tokens':tokens}
            messages.append({'role':'assistant','content':'','tool_calls':[{'id':t.id,'name':t.name,'arguments':t.arguments} for t in m.tool_calls]})
            for call in m.tool_calls:
                args=call.arguments if isinstance(call.arguments,dict) else json.loads(call.arguments)
                observation=self.tool(call.name,args,customer_id)
                messages.append({'role':'tool','name':call.name,'tool_call_id':call.id,'content':json.dumps(observation,default=str)})
        return {'outcome':'FALLBACK','text':'Maximum steps reached; hand over to human.','steps':self.max_steps,'tokens':tokens}

def evaluate():
    agent=RefundAgent()
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv')
    results=[]
    for row in scenarios.itertuples(index=False):
        result=agent.run(row.customer_id,row.message)
        results.append({'scenario_id':row.scenario_id,'expected':row.expected_outcome,**result,'correct':row.expected_outcome==result['outcome']})
    frame=pd.DataFrame(results)
    frame.to_csv(BASE/'reports/agent_eval_results.csv',index=False)
    return {'pass_rate':round(float(frame.correct.mean()),4),'passed':int(frame.correct.sum()),'total':len(frame)}
