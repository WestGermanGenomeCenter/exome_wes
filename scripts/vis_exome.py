#!/usr/bin/env python3

import argparse
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

###############################################################################
# Command line
###############################################################################

parser = argparse.ArgumentParser(
    description="Interactive visualization of annotated variant TSV."
)

parser.add_argument(
    "-i", "--input",
    required=True,
    help="Input annotated TSV"
)

parser.add_argument(
    "-o", "--output",
    required=True,
    help="Output HTML report"
)

args = parser.parse_args()

###############################################################################
# Read data
###############################################################################

df = pd.read_csv(args.input, sep="\t")
###############################################################################
# Additional derived columns
###############################################################################

import numpy as np

df["log_distance"] = np.log10(df["distance_to_target"] + 1)
# nicer labels
df["log_depth"] = np.log10(df["depth"] + 1)
df["Target"] = df["on_target_overlaps"].map({
    0: "Off target",
    1: "On target"
})

# hover text
df["Hover"] = (
    "<b>" + df["chrom"] + ":" + df["start"].astype(str) + "</b><br>" +
    df["ref"] + " → " + df["alt"] + "<br>" +
    "Variant: " + df["variant_type"].astype(str) + "<br>" +
    "Filter: " + df["filter"].astype(str) + "<br><br>" +
    "<b>Quality:</b> " + df["qual"].round(2).astype(str) + "<br>" +
    "<b>Depth:</b> " + df["depth"].round(2).astype(str) + "<br>" +
    "<b>Distance:</b> " + df["distance_to_target"].astype(str) + " bp<br><br>" +
    "<b>Nearest target</b><br>" +
    df["target_chrom"].astype(str) + ":" +
    df["target_start"].astype(str) + "-" +
    df["target_end"].astype(str)
)

###############################################################################
# Figure
###############################################################################





fig = make_subplots(
    rows=4,
    cols=2,
    specs=[
        [{"type": "scene"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}]
    ],




    subplot_titles=(
    "3D Scatter",
    "Quality vs Depth",
    "Depth Distribution",
    "Quality Distribution",
    "Quality vs log10(Distance)",
    "Capture Falloff",
    "log10(Depth) vs log10(Distance)",
    "Distance Distribution"
)
    
    
    
    
    
    
    
)





###############################################################################
# 3D scatter
###############################################################################

colors = {
    "On target": "#2ca02c",
    "Off target": "#d62728"
}

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Scatter3d(

            x=d["qual"],
            y=d["depth"],
            z=d["distance_to_target"],

            mode="markers",

            name=target,

            marker=dict(
                size=4,
                opacity=0.8,
                color=colors[target]
            ),

            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>"

        ),

        row=1,
        col=1

    )

###############################################################################
# Quality vs depth
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Scatter(

            x=d["qual"],
            y=d["depth"],

            mode="markers",

            marker=dict(
                color=colors[target],
                size=6,
                opacity=0.75
            ),

            name=target,
            legendgroup=target,
            showlegend=False,

            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>"

        ),

        row=1,
        col=2

    )

###############################################################################
# Violin: depth
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Violin(

            y=d["depth"],

            x=[target] * len(d),

            name=target,

            box_visible=True,
            meanline_visible=True,

            line_color=colors[target],

            showlegend=False

        ),

        row=2,
        col=1

    )

###############################################################################
# Quality vs log(distance)
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Scatter(

            x=d["log_distance"],
            y=d["qual"],

            mode="markers",

            marker=dict(
                color=colors[target],
                size=6,
                opacity=0.75
            ),

            name=target,
            legendgroup=target,
            showlegend=False,

            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>"

        ),

        row=3,
        col=1

    )

###############################################################################
# log depth vs log distance
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Scatter(

            x=d["log_distance"],
            y=d["log_depth"],

            mode="markers",

            marker=dict(
                color=colors[target],
                size=6,
                opacity=0.7
            ),

            name=target,
            legendgroup=target,
            showlegend=False,

            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>"

        ),

        row=4,
        col=1
    )


###############################################################################
# Distance distribution
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Histogram(

            x=d["log_distance"],

            name=target,

            marker_color=colors[target],

            opacity=0.6,

            nbinsx=40

        ),

        row=4,
        col=2
    )
###############################################################################
# Violin: quality
###############################################################################

for target in ["On target", "Off target"]:

    d = df[df["Target"] == target]

    fig.add_trace(

        go.Violin(

            y=d["qual"],

            x=[target] * len(d),

            name=target,

            box_visible=True,
            meanline_visible=True,

            line_color=colors[target],

            showlegend=False

        ),

        row=2,
        col=2

    )

###############################################################################
# Capture falloff
###############################################################################

bins = [
    0,
    10,
    50,
    100,
    250,
    500,
    1000,
    5000,
    10000,
    50000,
    100000,
    df["distance_to_target"].max() + 1
]

tmp = df.copy()

tmp["bin"] = pd.cut(
    tmp["distance_to_target"],
    bins=bins,
    include_lowest=True
)

grouped = (
    tmp
    .groupby(["bin", "Target"], observed=True)["depth"]
    .median()
    .reset_index()
)

for target in ["On target", "Off target"]:

    d = grouped[grouped["Target"] == target]

    fig.add_trace(

        go.Scatter(

            x=[str(x) for x in d["bin"]],
            y=d["depth"],

            mode="lines+markers",

            name=target,

            line=dict(color=colors[target], width=3),

            marker=dict(size=8),

            showlegend=False

        ),

        row=3,
        col=2

    )

###############################################################################
# Layout
###############################################################################


fig.update_layout(

    title="Variant Annotation Report",

    template="plotly_white",

    autosize=True,

    height=4000,

    margin=dict(
        l=40,
        r=40,
        t=80,
        b=40
    ),

    legend_title="Target"

)





fig.update_scenes(

    xaxis_title="Quality",
    yaxis_title="Depth",
    zaxis_title="Distance to Target (bp)"

)

fig.update_xaxes(title_text="Quality", row=1, col=2)
fig.update_yaxes(title_text="Depth", row=1, col=2)
fig.update_xaxes(
    title_text="log10(Distance + 1)",
    row=3,
    col=1
)

fig.update_yaxes(
    title_text="Quality",
    row=3,
    col=1
)

fig.update_xaxes(
    title_text="Distance bin",
    row=3,
    col=2
)

fig.update_yaxes(
    title_text="Median depth",
    row=3,
    col=2
)
fig.update_yaxes(title_text="Depth", row=2, col=1)
fig.update_yaxes(title_text="Quality", row=2, col=2)

fig.update_xaxes(
    title_text="log10(Distance + 1)",
    row=4,
    col=1
)

fig.update_yaxes(
    title_text="log10(Depth + 1)",
    row=4,
    col=1
)


fig.update_xaxes(
    title_text="log10(Distance + 1)",
    row=4,
    col=2
)

fig.update_yaxes(
    title_text="Variant count",
    row=4,
    col=2
)
###############################################################################
# Write HTML
###############################################################################


pio.write_html(
    fig,
    file=args.output,
    include_plotlyjs=True,
    full_html=True,
    config={
        "responsive": True
    }
)




print(f"Wrote {args.output}")