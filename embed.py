import polars as pl
import openai
import sys
from openai.embeddings_utils import get_embedding, cosine_similarity
from get_file_data import check
from read_sources import dump
from create_dataset import parse

try:
    from constants import token
except ImportError as e:
    print("Couldn't get the credentials:", e)
    sys.exit(1)
openai.api_key = token

def make(include):
    dump(include)
    parse(include)

def embed(file, model="ada"):
    df = pl.read_csv(f"ai_generated/data/{file}.csv")
    df = df.lazy().with_column(
            pl.col("text").apply(lambda text: get_embedding(text, engine=f'text-embedding-{model}-002')).alias("embed")
        ).collect()
    df.write_json(f"ai_generated/embeds/{file}.json")
    return check()

def query(df, q):
    search_query_emb = get_embedding(q, engine=f"text-embedding-ada-002")
    df = df.lazy().with_column(
        pl.col("embed").apply(lambda emb: cosine_similarity(emb, search_query_emb)).alias("similarities")
    ).sort("similarities", reverse=True).collect()
    df = df.head(3)

    return [
        list(df["text"]),
        list(df["similarities"]),
        list(df["file_name"])
    ]