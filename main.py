from get_file_data import check
from read_sources import dump
from create_dataset import parse
from embed import embed, query
from tqdm import tqdm
import pandas as pd
import polars as pl
import sys
import os
from flask import Flask, render_template, request
app = Flask(__name__)
app.debug = True

if __name__ == '__main__':
    print(f"\n{'Project Author':^100}\n")
    need_emb = check()["regen"]

    if len(need_emb) > 0:
        parse(dump(need_emb))

        total = 0
        with tqdm(total=len(need_emb), desc="Calculating total token") as pbar:
            for file in need_emb:
                pbar.update(1)
                total += pd.read_csv(f'ai_generated/data/{file}.csv')["token_size"].sum()
        print(f"\n{total:,} tokens in total, (approx. ${total/1000 * 0.0004})")

        if input("Would you like to embed? (y/n)").lower() == 'y':
            with tqdm(total=len(need_emb), desc="Embedding files") as pbar:
                for file in need_emb:
                    pbar.update(1)
                    embed(file)
        else:
            sys.exit(1)
    dfs = []
    for root, _, files in os.walk("ai_generated/embeds"):
        for file in files:
            if file == ".gitkeep":
                continue
            dfs.append(pl.read_json(f"{root}/{file}"))
    combined_df = pl.concat(dfs)

    search_query = ""
    result = []
    @app.route("/search", methods=["POST"])
    def search():
        global search_query, result
        search_query = request.form["search"]
        result = query(combined_df, search_query)
        
        return render_template("index.html", result=result, query=search_query)

    @app.route("/")
    def index():
        global search_query, result
        return render_template("index.html", result=result, query=search_query)

    app.run()
