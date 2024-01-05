#!/usr/bin/env python3

from refactored import PaCMAP
import pandas
import sys
import sklearn.model_selection
import sklearn.feature_extraction
import matplotlib.pyplot as plt
import tqdm
import numpy
import math
from scipy.optimize import linear_sum_assignment
import os
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import time
import sqlite3
import sklearn.cluster
from openai import OpenAI
import argparse
import os

# Let's have command-line arguments so that we can run this repeatedly with
# different options.
parser = argparse.ArgumentParser()
parser.add_argument("--data-file", default=os.path.expanduser("~/Downloads/front_summary.csv"))
parser.add_argument("--skip-image", action="store_true", help="Don't bother with an image")
parser.add_argument("--random-seed", type=int, default=42, help="Reproducibility key")
args = parser.parse_args()

client = OpenAI(
    # This is the default and can be omitted
    api_key="TODO",
)

# --- LOAD DATA ---
front_support_categories = pandas.read_csv(args.data_file,
                                           na_filter=False
)
front_support_categories = front_support_categories[
    front_support_categories.tag_id.str.startswith("tag_")
    & front_support_categories.tag_id.notnull()
]
minimum_examples = 10
number_of_examples_for_tag = front_support_categories.tag_id.value_counts()

sufficient_examples = set(list((number_of_examples_for_tag >= minimum_examples).index))

front_support_categories = front_support_categories[
    front_support_categories.tag_id.isin(sufficient_examples)
]

# --------
id2tag_lookup = {
    tagid: tagname
    for (tagid, tagname) in enumerate(front_support_categories.tag_id.unique())
}
tag2id_lookup = {tagname: tagid for (tagid, tagname) in id2tag_lookup.items()}

# -----------
conn = sqlite3.connect(".embeddings_cache.sqlite")
cursor = conn.cursor()
sql = (
    """create table if not exists embeddings (
      sentence text primary key,
      """
    + (" float,\n".join([f"embedding_{x}" for x in range(1536)]))
    + """ float
    );"""
)
cursor.execute(sql)

# ------------
embedding_columns = [f"embedding_{x}" for x in range(1536)]
embedding_columns_text = ", ".join(embedding_columns)
embedding_placeholders = ",".join(["?" for x in range(1536)])


def save_embedding(text, embedding):
    if len(embedding) != 1536:
        raise ValueError
    global embedding_columns_text
    global embedding_placeholders
    global conn
    sql = f"insert into embeddings (sentence, {embedding_columns_text}) values (?,{embedding_placeholders})"
    cursor = conn.cursor()
    cursor.execute(sql, [text] + embedding)
    conn.commit()
    cursor.close()


# -----------
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    global conn
    global embedding_columns_text
    global embedding_columns
    cursor = conn.cursor()
    cursor.execute(
        f"select {embedding_columns_text} from embeddings where sentence = ?", [text]
    )
    row = cursor.fetchone()
    if row is not None:
        # pre-cached. Nice.
        return [row[x] for x in range(1536)]
    # Have to calculate it. Maybe I should keep track of the cost or something?
    response = client.embeddings.create(input=[text], model=model).data[0].embedding
    save_embedding(text, response)
    # Let's not overwhelm it
    time.sleep(0.5)
    return response


# --- PREPARE FOR PACMAP ---

# Get embeddings for the summaries
front_support_categories["embedding"] = front_support_categories.summary.apply(
    get_embedding
)
# filter NaN
front_support_categories = front_support_categories[
    front_support_categories.embedding.notnull()
]
# drop duplicate index
front_support_categories = front_support_categories.reset_index(drop=True)

# ------------------------

TOP_N_TAGS = 10
NUMBER_OF_SAMPLES_PER_TAG = 30

tag_series = front_support_categories.tag_id
# filter tag called slnib (no support action needed)
tag_series = tag_series[tag_series != "tag_slnib"]

target_tags = set(tag_series.value_counts().nlargest(TOP_N_TAGS).index)

random_selection = []
random_selection_validation = []
for tag in target_tags:
    filtered_tags = tag_series[tag_series == tag]
    sampled = list(
        tag_series[tag_series == tag]
        .sample(NUMBER_OF_SAMPLES_PER_TAG, random_state=args.random_seed)
        .index
    )

    # first 20 are training, last 10 are validation
    random_selection.extend(sampled[:20])
    random_selection_validation.extend(sampled[20:])


filtered_front_support_categories = front_support_categories.iloc[random_selection]

filtered_front_support_categories_validation = front_support_categories.iloc[
    random_selection_validation
]


# ----------------------

robert_X = numpy.array(list(filtered_front_support_categories.embedding))
robert_target_y = filtered_front_support_categories.tag_id
robert_validation_y = filtered_front_support_categories_validation.tag_id
# get rid of the tag_ prefix
# robert_target_y = robert_target_y.apply(lambda x: x[4:])

# reset the index TODO idk if this is right lol -> am i actually making sure the order / index is same as robert_X?
# GregB... not even sure it is necessary. What are we trying to fix?
robert_target_y.index = range(len(robert_target_y))

# robert_y = valid_targets_series.loc[random_selection]

# --------------
pacmapper = PaCMAP(n_components=2, verbose=True)

robert_2d_pacmap = pacmapper.fit_transform(X=robert_X, y=robert_target_y)
robert_2d_pacmap = pandas.DataFrame(robert_2d_pacmap, columns=["x0", "x1"], index=robert_target_y.index)


robert_X_validation = numpy.array(
    list(filtered_front_support_categories_validation.embedding)
)

robert_X_df_holdout_e = pacmapper.transform(X=robert_X_validation, basis=robert_X)
robert_X_df_holdout_e = pandas.DataFrame(robert_X_df_holdout_e, columns=["x0", "x1"], index=robert_validation_y.index)


if not args.skip_image:
    fig, ax = plt.subplots()

    cmap = plt.colormaps['tab10']  # Get 10 distinct colors
    colours = cmap(range(10))
    print(colours)
    
    tag_colour = {}
    for c,tag in zip(colours, robert_target_y.unique()):
        tag_colour[tag] = c
        print(f"Using color={c} for the colour of {tag}")
        robert_2d_pacmap[robert_target_y == tag].plot.scatter(
            x="x0", y="x1", color=c,
            label=tag,
            ax=ax,
            #s=10
        )

    for tag in robert_validation_y.unique():
        c = tag_colour.get(tag, "black")
        robert_X_df_holdout_e[robert_validation_y == tag].plot.scatter(
            x="x0",
            y="x1",
            color=c,
            ax=ax,
            marker="+",
            #s=1
        )

    ax.legend()
    plt.show()


nn = sklearn.neighbors.KNeighborsClassifier()
nn.fit(robert_2d_pacmap, robert_target_y)
# We should get 100% on the training data
print("Accuracy score on training data = ", sklearn.metrics.accuracy_score(robert_target_y, nn.predict(robert_2d_pacmap)))

print("Accuracy score on holdout data = ", sklearn.metrics.accuracy_score(robert_validation_y, nn.predict(robert_X_df_holdout_e)))

