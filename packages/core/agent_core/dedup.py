"""Bounded multilingual retrieval and lexical overlap; retrieval is not a verdict."""
import hashlib
import math
import re
import unicodedata
import httpx

MODEL_VERSION = 'bge-m3-q8_0:9eba04c5d75b:cls:1024'


def normalized(text):
    text = re.sub(r'\[\d+\]', '', text)
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def fingerprint(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def chunks(text, size=700, overlap=100):
    # Character limits bound embedding inputs below the deployment's 2048-token limit.
    return [text[i:i+size] for i in range(0, min(len(text), 12000), size-overlap) if len(text[i:i+size].strip()) >= 40]


def overlap_score(draft, source, width=24):
    """Fraction of draft character shingles present in a source, ignoring citations."""
    a,b=normalized(draft),normalized(source)
    if len(a)<width or len(b)<width:
        return 0.0
    left={a[i:i+width] for i in range(len(a)-width+1)}
    right={b[i:i+width] for i in range(len(b)-width+1)}
    return len(left & right)/len(left)


class Embeddings:
    def __init__(self, settings):
        self.url,self.key,self.model=settings.litellm_url,settings.litellm_key,settings.embedding_model

    async def embed(self, texts):
        vectors=[]
        async with httpx.AsyncClient(timeout=90,trust_env=False) as client:
            # Sequential bounded requests avoid transient GPU allocation growth.
            for text in texts:
                r=await client.post(self.url+'/embeddings',headers={'Authorization':'Bearer '+self.key},
                    json={'model':self.model,'input':text[:1500],'encoding_format':'float'})
                r.raise_for_status()
                vector=r.json()['data'][0]['embedding']
                if len(vector)!=1024 or not all(math.isfinite(x) for x in vector):
                    raise ValueError('Invalid embedding vector')
                norm=math.sqrt(sum(x*x for x in vector))
                if norm<0.9 or norm>1.1:
                    raise ValueError('Unexpected embedding normalization')
                vectors.append([x/norm for x in vector])
        return vectors


def vector_literal(vector):
    if len(vector)!=1024 or not all(math.isfinite(x) for x in vector):
        raise ValueError('Expected finite 1024-dimensional embedding')
    return '['+','.join(str(float(x)) for x in vector)+']'
