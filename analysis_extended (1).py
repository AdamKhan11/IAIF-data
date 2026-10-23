import re, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
clean=lambda s: re.sub(r'\s+',' ',str(s).replace('Ê','')).strip()
def skel(s): return re.sub(r'\b(log in|login|authenticate|authorize|search|delete|update|create|view|upload|download|access|register|send email|generate report|reset password|performance|security|usability|reliability|availability|scalability|uptime|backup|recovery|logging|encryption|compatibility|accessibility|response time|load time)\b','X',s.lower())
d=pd.read_csv('supp_corpus_202_with_split.csv'); d['sk']=d.text.map(skel)
X=d.text.values; y=(d.label=='Functional').astype(int).values; g=d.sk.factorize()[0]
fam=(d.sk.map(d.sk.value_counts())>=3).values; nat=~fam
trm=(d.split=='train').values; tem=~trm
models={'LR (word TF-IDF)':lambda: make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),LogisticRegression(max_iter=1000)),
'LR (char 2-5 TF-IDF)':lambda: make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True),LogisticRegression(max_iter=1000)),
'Linear SVM (word TF-IDF)':lambda: make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),LinearSVC()),
'Naive Bayes (word counts)':lambda: make_pipeline(TfidfVectorizer(ngram_range=(1,2),use_idf=False,norm=None),MultinomialNB())}
def acc(mk,a,c):
    m=mk(); m.fit(X[a],y[a]); return accuracy_score(y[c],m.predict(X[c]))
out={}
for name,mk in models.items():
    r={}
    r['fixed']=acc(mk,np.where(trm)[0],np.where(tem)[0])
    for kind in ('random','group'):
        v=[]
        for s in range(20):
            sp=(StratifiedKFold(5,shuffle=True,random_state=s).split(X,y) if kind=='random' else StratifiedGroupKFold(5,shuffle=True,random_state=s).split(X,y,g))
            v+=[acc(mk,a,c) for a,c in sp]
        r[kind]=(np.mean(v),np.std(v))
    r['nat']=acc(mk,np.where(~nat)[0],np.where(nat)[0])
    out[name]=r; print(name,{k:(np.round(v,3) if not isinstance(v,tuple) else tuple(np.round(v,3))) for k,v in r.items()})
# near-duplicate audit on fixed split
def toks(s): return set(re.findall(r'[a-z0-9]+',s.lower()))
trT=[toks(t) for t in X[trm]]; teT=[toks(t) for t in X[tem]]
mj=[max(len(a&b)/len(a|b) for b in trT) for a in teT]; mj=np.array(mj)
print('test items with max Jaccard>=0.8:',(mj>=0.8).sum(),' >=0.6:',(mj>=0.6).sum(),' <0.5:',(mj<0.5).sum(),'of',len(mj),'median',np.median(mj).round(2))
v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True).fit(X[trm]); S=cosine_similarity(v.transform(X[tem]),v.transform(X[trm]))
nn=y[trm][S.argmax(1)]; print('1-NN cosine acc on test',round(accuracy_score(y[tem],nn),3))
# correct by similarity band for LR
m=models['LR (word TF-IDF)'](); m.fit(X[trm],y[trm]); p=m.predict(X[tem]); print('LR errors',[X[tem][i] for i in np.where(p!=y[tem])[0]])
# learning curves (word LR): fraction of training data, random vs grouped test
fr=[0.2,0.4,0.6,0.8,1.0]; lc={'random':[], 'group':[]}
rng=np.random.RandomState(0)
for kind in lc:
    for f in fr:
        v=[]
        for s in range(10):
            sp=list((StratifiedKFold(5,shuffle=True,random_state=s).split(X,y) if kind=='random' else StratifiedGroupKFold(5,shuffle=True,random_state=s).split(X,y,g)))
            for a,c in sp:
                a=rng.permutation(a)[:max(10,int(len(a)*f))]
                if len(set(y[a]))<2: continue
                v.append(acc(models['LR (word TF-IDF)'],a,c))
        lc[kind].append((f,np.mean(v),np.std(v)))
print({k:[(f,round(m,3)) for f,m,_ in v] for k,v in lc.items()})
json.dump({'lc':lc,'models':{k:{kk:(list(vv) if isinstance(vv,tuple) else vv) for kk,vv in v.items()} for k,v in out.items()}},open('ext.json','w'))
# Dual statements
raw=pd.read_excel('ds.xlsx'); raw['t']=raw.Requirement_Statement.map(clean)
du=raw[raw.Requirement_Type=='Dual'].t.drop_duplicates().values
m=models['LR (word TF-IDF)'](); m.fit(X,y); pd_=m.predict(du); print('Dual n',len(du),'pred FR',int(pd_.sum()),'pred NFR',int((1-pd_).sum()))
print(du[:5])
