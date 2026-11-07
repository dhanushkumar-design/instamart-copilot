import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reports/customer_segments.csv")

segment_col = next(
    (c for c in df.columns if "segment" in c.lower() or "cluster" in c.lower()),
    None
)

if segment_col is None:
    print("Segment column not found.")
    print("Available columns:", df.columns.tolist())
else:
    counts = df[segment_col].value_counts().sort_index()

    counts.plot(kind="bar", color="skyblue", edgecolor="black")

    plt.title("Instamart Customer Segmentation")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("reports/customer_segmentation.png", dpi=300)
    plt.show()